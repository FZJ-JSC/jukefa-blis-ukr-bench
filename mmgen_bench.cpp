// A benchmark for one mmgen plan: generate a GEMM from a nest, check it,
// measure it, and print what the model needs to compare against.
//
//     mmgen_bench --size 2000 2000 2000 --nest "n:1000 k:144 m:288 n:nr m:mr"
//     mmgen_bench --size 600 600 600 --nest "n:200 k:200 | m:120 k:192 | n:nr m:mr" --verify
//     mmgen_bench ... --sum            report a hash of C, to compare harnesses
//     mmgen_bench ... --profile        report the plan's own pack/ukr split
//
// It is the GOTO bench's counterpart, not its replacement yet: no counters,
// no prefetch specs, no presets.  Those come with the compatibility mode.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "harness/blis_provider.hpp"
#include "mmgen/gemm.hpp"

namespace {

struct options
{
    long long m = 0, n = 0, k = 0;
    std::string nest, pack_at, top, beta_mode = "general", label, c_layout = "rm";
    double alpha = 2.1;
    long warmup = 2, inner = 0, reps = 0;
    double min_time = 0.3;
    int cpu = -1;
    bool verify = false, sum = false, hugepages = false, alloc_stats = false;
    bool pack_loop = false, profile = false;
};

[[noreturn]] void usage(const char* prog, const std::string& why = "")
{
    if (!why.empty()) std::fprintf(stderr, "error: %s\n\n", why.c_str());
    std::fprintf(stderr, R"(usage: %s --size M N K --nest "LOOPS" [options]

  --nest "..."        the nest, outer to inner, ending in m:mr and n:nr (either
                      order); '|' groups loops into a level; a block may be a
                      multiple of the tile, 4mr.  As model.py --nest:
                        n:1920 k:256 m:96 n:nr m:mr            Goto, A2C0
                        n:768 k:768 | m:120 k:192 | n:nr m:mr  B3A2C0
  --pack-at A:L,B:L   the loop each operand is packed in, or pre (default: the
                      loop around the first below the top that keeps it
                      invariant -- Goto's ic and pc)
  --top LABEL         outermost rung that iterates (default: the whole nest)
  --pack-kernel loop  pack with the plain copy loop, not BLIS's packm kernel
  --beta MODE|X       general (alpha 2.1, beta 1.3), one, zero, or a number
  --alpha X           (default 2.1)
  --c-layout rm|cm    C's layout (default rm)

timing
  --warmup N          untimed runs before measuring (default 2)
  --inner N           runs per measurement (default: ~200 us worth)
  --reps N            measurements (default: enough for --min-time)
  --min-time S        seconds of timed runs (default 0.3)
  --profile           also report the plan's calls, packing and per-rung trips

checking
  --verify            check the region against a plain triple loop, then exit
  --sum               print a hash of C, to compare harnesses bit for bit

system
  --cpu N             pin to this CPU
  --hugepages         allocate 2 MiB-aligned, MADV_HUGEPAGE buffers
  --alloc-stats       print what the allocator handed out
  --label TEXT        carried into the output row
  -h, --help          this text
)",
        prog);
    std::exit(2);
}

double time_ns(const std::function<void()>& f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// Deterministic contents, mean near zero, no denormals -- the same generator
// the GOTO bench uses, so the two harnesses can be compared with --sum.
double value(std::size_t i, unsigned seed)
{
    return 0.25 * static_cast<double>(static_cast<int>((i * 2654435761u + seed * 97u) % 9u) - 4);
}

void fill(double* x, std::size_t n, unsigned seed)
{
    for (std::size_t i = 0; i < n; i++) x[i] = value(i, seed);
}

std::uint64_t checksum(const double* c, std::size_t n)
{
    const std::uint64_t* p = reinterpret_cast<const std::uint64_t*>(c);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; i++)
    {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

int main(int argc, char** argv)
{
    options o;
    try
    {
        for (int i = 1; i < argc; i++)
        {
            const std::string a = argv[i];
            auto need = [&](int n) {
                if (i + n >= argc) usage(argv[0], a + " needs a value");
            };
            if (a == "-h" || a == "--help") usage(argv[0]);
            else if (a == "--size") { need(3); o.m = std::stoll(argv[++i]); o.n = std::stoll(argv[++i]); o.k = std::stoll(argv[++i]); }
            else if (a == "--nest") { need(1); o.nest = argv[++i]; }
            else if (a == "--pack-at") { need(1); o.pack_at = argv[++i]; }
            else if (a == "--top") { need(1); o.top = argv[++i]; }
            else if (a == "--beta") { need(1); o.beta_mode = argv[++i]; }
            else if (a == "--alpha") { need(1); o.alpha = std::stod(argv[++i]); }
            else if (a == "--c-layout") { need(1); o.c_layout = argv[++i];
                if (o.c_layout != "rm" && o.c_layout != "cm") usage(argv[0], "--c-layout rm|cm"); }
            else if (a == "--warmup") { need(1); o.warmup = std::stol(argv[++i]); }
            else if (a == "--inner") { need(1); o.inner = std::stol(argv[++i]); }
            else if (a == "--reps") { need(1); o.reps = std::stol(argv[++i]); }
            else if (a == "--min-time") { need(1); o.min_time = std::stod(argv[++i]); }
            else if (a == "--verify") o.verify = true;
            else if (a == "--sum") o.sum = true;
            else if (a == "--profile") o.profile = true;
            else if (a == "--pack-kernel") { need(1); const std::string v = argv[++i];
                if (v == "loop") o.pack_loop = true; else if (v != "blis") usage(argv[0], "--pack-kernel blis|loop"); }
            else if (a == "--cpu") { need(1); o.cpu = std::stoi(argv[++i]); }
            else if (a == "--hugepages") o.hugepages = true;
            else if (a == "--alloc-stats") o.alloc_stats = true;
            else if (a == "--label") { need(1); o.label = argv[++i]; }
            else usage(argv[0], "unknown option " + a);
        }
    }
    catch (const std::exception& e) { usage(argv[0], e.what()); }
    if (o.m <= 0 || o.n <= 0 || o.k <= 0) usage(argv[0], "--size M N K is required");
    if (o.nest.empty()) usage(argv[0], "--nest is required");

#if defined(__linux__)
    if (o.cpu >= 0)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(o.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            std::fprintf(stderr, "# warning: could not pin to CPU %d\n", o.cpu);
    }
#endif

    gotobench::blis_kernels ker;
    mmgen::blis_provider provider(ker);
    const mmgen::kernel_info& ki = provider.info();

    mmgen::gemm_config cfg;
    cfg.pack_at = o.pack_at;
    cfg.profile = o.profile;
    mmgen::problem p{static_cast<dim_t>(o.m), static_cast<dim_t>(o.n), static_cast<dim_t>(o.k)};

    // --top is a label: translate it to a rung index once the nest is parsed
    mmgen::aligned_allocator alloc(4096, 0, o.hugepages);
    // --pack-kernel loop: hand the plan no pack kernels at all, so its own
    // copy loop packs -- a fair look at what a kernel-less path costs
    if (o.pack_loop) provider.disable_packers();

    std::unique_ptr<mmgen::gemm> g;
    try
    {
        mmgen::nest_spec spec = mmgen::parse_nest(o.nest, ki.mr, ki.nr, o.pack_at);
        if (!o.top.empty())
        {
            const auto labels = spec.labels();
            cfg.top = -1;
            for (std::size_t i = 0; i < labels.size(); i++) if (labels[i] == o.top) cfg.top = static_cast<int>(i);
            if (cfg.top < 0) usage(argv[0], "--top " + o.top + ": no such rung");
        }
        g = std::make_unique<mmgen::gemm>(provider, spec, p, alloc, cfg);
    }
    catch (const std::exception& e) { usage(argv[0], e.what()); }

    // the matrices: page-aligned, huge pages on request, filled deterministically
    const std::size_t na = static_cast<std::size_t>(o.m) * o.k;
    const std::size_t nb = static_cast<std::size_t>(o.k) * o.n;
    const std::size_t nc = static_cast<std::size_t>(o.m) * o.n;
    double* A = static_cast<double*>(alloc.allocate(na * sizeof(double), 4096));
    double* B = static_cast<double*>(alloc.allocate(nb * sizeof(double), 4096));
    double* C = static_cast<double*>(alloc.allocate(nc * sizeof(double), 4096));
    fill(A, na, 1);
    fill(B, nb, 2);
    fill(C, nc, 3);

    mmgen::call k;
    k.a = A; k.lda = o.m;
    k.b = B; k.ldb = o.k;
    k.c = C;
    k.cs_c = 1;
    k.rs_c = o.c_layout == "cm" ? 1 : o.n;
    k.alpha = o.alpha;
    k.beta = o.beta_mode == "general" ? 1.3 : o.beta_mode == "one" ? 1.0
             : o.beta_mode == "zero" ? 0.0 : std::stod(o.beta_mode);

    // what was generated, in the model's terms
    std::printf("# nest      %s\n", g->text().c_str());
    std::printf("# loops     %s\n", g->loops().c_str());
    std::printf("# packing   %s\n", g->packing().c_str());
    std::printf("# kernel    %s  %lldx%lld\n", ki.impl.c_str(),
                static_cast<long long>(ki.mr), static_cast<long long>(ki.nr));
    std::printf("# packers   A: %zu elem/block x %zu slot(s)  B: %zu elem/block x %zu slot(s)\n",
                g->block_elems(mmgen::operand::A), g->slots(mmgen::operand::A),
                g->block_elems(mmgen::operand::B), g->slots(mmgen::operand::B));
    std::printf("# footprint %.1f MiB packed, %.1f MiB of matrices (allocator %s)\n",
                static_cast<double>(g->pack_bytes()) / (1 << 20),
                static_cast<double>(na + nb + nc) * sizeof(double) / (1 << 20), alloc.name());

    auto run_once = [&] { (*g)(k); };

    if (o.verify)
    {
        // one run from the current C, against a plain triple loop over the
        // region; nothing outside the region may change, and C is restored
        std::vector<double> c0(C, C + nc);
        run_once();
        const dim_t me = g->plan().m_exec(), ne = g->plan().n_exec(), ke = g->plan().k_exec();
        double worst = 0.0;
        std::vector<double> acc(static_cast<std::size_t>(me));
        for (dim_t j = 0; j < ne; j++)
        {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (dim_t q = 0; q < ke; q++)
            {
                const double bqj = B[q + j * k.ldb];
                const double* a = A + q * k.lda;
                for (dim_t i = 0; i < me; i++) acc[static_cast<std::size_t>(i)] += a[i] * bqj;
            }
            for (dim_t i = 0; i < me; i++)
            {
                const std::size_t ij = static_cast<std::size_t>(i * k.rs_c + j * k.cs_c);
                const double ref = k.beta * c0[ij] + k.alpha * acc[static_cast<std::size_t>(i)];
                worst = std::max(worst, std::fabs(C[ij] - ref) / (std::fabs(ref) + 1.0));
            }
        }
        for (dim_t i = 0; i < o.m; i++)
            for (dim_t j = 0; j < o.n; j++)
            {
                const std::size_t ij = static_cast<std::size_t>(i * k.rs_c + j * k.cs_c);
                if ((i >= me || j >= ne) && C[ij] != c0[ij]) worst = 1.0;
            }
        std::printf("# verify    m_exec=%lld n_exec=%lld k_exec=%lld  max relative error %.3g -> %s\n",
                    static_cast<long long>(me), static_cast<long long>(ne),
                    static_cast<long long>(ke), worst, worst < 1e-10 ? "OK" : "FAILED");
        std::copy(c0.begin(), c0.end(), C);
        if (worst >= 1e-10) return 1;
    }
    if (o.sum)
    {
        fill(C, nc, 3);
        run_once();
        std::printf("# sum       %016llx\n", static_cast<unsigned long long>(checksum(C, nc)));
        fill(C, nc, 3);
    }

    if (o.profile) g->reset_stats();
    auto region = [&] { run_once(); };
    for (long w = 0; w < o.warmup; w++) region();
    long inner = o.inner;
    if (inner <= 0)
        inner = std::max(1L, static_cast<long>(200e3 / std::max(time_ns(region), 1.0)));
    auto body = [&] { for (long i = 0; i < inner; i++) region(); };
    long reps = o.reps;
    if (reps <= 0)
    {
        const double one = std::max(time_ns(body), 1.0);
        reps = std::clamp(static_cast<long>(o.min_time * 1e9 / one) + 1, 3L, 100000L);
    }
    std::vector<double> ns;
    for (long r = 0; r < reps; r++) ns.push_back(time_ns(body) / static_cast<double>(inner));
    std::sort(ns.begin(), ns.end());
    const double ns_min = ns.front(), ns_med = ns[ns.size() / 2];
    const double gflops = g->work() / ns_min;

    if (o.alloc_stats)
        std::printf("# allocator %s: %zu buffers, %.1f MiB\n", alloc.name(), alloc.blocks(),
                    static_cast<double>(alloc.bytes()) / (1 << 20));

    std::printf("label,m,n,k,top,calls,work,reps,inner,ns_min,ns_med,gflops\n");
    std::printf("%s,%lld,%lld,%lld,%s,%.0f,%.0f,%ld,%ld,%.1f,%.1f,%.3f\n",
                o.label.empty() ? "-" : o.label.c_str(), o.m, o.n, o.k,
                o.top.empty() ? "-" : o.top.c_str(), g->calls(), g->work(), reps, inner,
                ns_min, ns_med, gflops);
    std::printf("# %.2f GFLOP/s per run, %.3f ns per micro-kernel call, %ld reps x %ld runs\n",
                gflops, ns_min / std::max(1.0, g->calls()), reps, inner);

    if (o.profile)
    {
        const mmgen::profile& s = g->stats();
        const double regions = std::max<double>(1.0, static_cast<double>(s.regions));
        std::printf("# profile   %zu region(s): %.3f ms each, pack %.1f%% (%.3f ms), "
                    "%.2f pack calls/region, %.1f Mpacked elem/region\n",
                    s.regions, s.total_ns / regions / 1e6,
                    100.0 * s.pack_ns / std::max(1.0, s.total_ns), s.pack_ns / regions / 1e6,
                    static_cast<double>(s.pack_calls) / regions,
                    static_cast<double>(s.pack_elems) / regions / 1e6);
        std::printf("#           %.1f calls/region\n",
                    static_cast<double>(s.calls) / regions);
        std::printf("#           per-rung iterations/region:");
        for (std::size_t i = 0; i < s.iteration.size(); i++)
            std::printf(" %.3g", static_cast<double>(s.iteration[i]) / regions);
        std::printf("  (%s)\n", g->loops().c_str());
    }

    alloc.release(A);
    alloc.release(B);
    alloc.release(C);
    return 0;
}
