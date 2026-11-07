// One nest, one problem, measured: the MOMMS harness.
//
// This is the composition harness -- rungs from --nest, packers on A and B,
// BLIS's kernels as the leaf -- not the GOTO bench.  It exists to show that a
// nest built out of data reproduces the hand-written one (same C, bit for bit,
// with --sum) and to time members of the family GOTO's bench cannot express.
//
//     momms_bench --size 3000 3000 3000 --nest "n:768 k:768 | m:120 k:192 | n:nr m:mr"
//     momms_bench --size 4000 4000 4000 --nest "n:144 k:144 m:288 n:nr m:mr" --verify
//     momms_bench ... --top ir --sum
//
// Everything the model predicts for a nest -- the loops, where the blocks are
// packed, how many blocks stay live -- is printed before the timing, so a run
// can be read against model.py --nest without opening the code.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "harness/blis_provider.hpp"
#include "harness/nest.hpp"

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
    bool pack_loop = false;
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
  --warmup N          untimed region runs before measuring (default 2)
  --inner N           region runs per measurement (default: ~200 us worth)
  --reps N            measurements (default: enough for --min-time)
  --min-time S        seconds of timed runs (default 0.3)

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

double time_ns(std::function<void()> f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
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
    momms::blis_gemm_ctx gctx{&ker};
    momms::ukernel ukr = momms::blis_kernel(ker, gctx);

    momms::nest_spec spec;
    try { spec = momms::parse_nest(o.nest, ker.mr, ker.nr, o.pack_at); }
    catch (const std::exception& e) { usage(argv[0], e.what()); }

    momms::nest_config cfg;
    cfg.m = o.m; cfg.n = o.n; cfg.k = o.k;
    cfg.lda = o.m; cfg.ldb = o.k;
    cfg.alpha = o.alpha;
    cfg.dt = momms::ptype::f64;
    // beta
    if (o.beta_mode == "general") cfg.beta = 1.3;
    else if (o.beta_mode == "one") cfg.beta = 1.0;
    else if (o.beta_mode == "zero") cfg.beta = 0.0;
    else cfg.beta = std::stod(o.beta_mode);
    // C's layout
    if (o.c_layout == "cm") { cfg.rs_c = 1; cfg.cs_c = o.m; }
    else                    { cfg.rs_c = o.n; cfg.cs_c = 1; }

    // --top: the index of that rung, default the whole nest
    cfg.top = 0;
    if (!o.top.empty())
    {
        cfg.top = -1;
        const auto labels = spec.labels();
        for (std::size_t i = 0; i < labels.size(); i++) if (labels[i] == o.top) cfg.top = static_cast<int>(i);
        if (cfg.top < 0) usage(argv[0], "--top " + o.top + ": no such rung");
    }

    momms::aligned_allocator alloc(o.hugepages);
    momms::blis_pack_ctx pac{&ker, true}, pbc{&ker, false};
    momms::pack_provider pa, pb;
    if (!o.pack_loop) { pa.fn = &momms::blis_pack; pa.ctx = &pac;
                        pb.fn = &momms::blis_pack; pb.ctx = &pbc; }

    std::unique_ptr<momms::nest> n;
    try
    {
        n = std::make_unique<momms::nest>(ukr, spec, cfg, alloc, pa, pb);
        n->prepack();
    }
    catch (const std::exception& e) { usage(argv[0], e.what()); }

    // what was built, in the model's terms
    std::printf("# nest      %s\n", n->text().c_str());
    std::printf("# loops     %s\n", n->loops_text().c_str());
    std::printf("# packing   %s\n", n->pack_text().c_str());
    std::printf("# kernel    %s  %lldx%lld\n", ukr.impl.c_str(),
                static_cast<long long>(ker.mr), static_cast<long long>(ker.nr));
    std::printf("# packers   A: %zu elem/block x %zu slot(s)  B: %zu elem/block x %zu slot(s)\n",
                n->block_elems(momms::OP_A), n->slots(momms::OP_A),
                n->block_elems(momms::OP_B), n->slots(momms::OP_B));
    std::printf("# footprint %.1f MiB (allocator %s, %zu buffers)\n",
                static_cast<double>(n->footprint_bytes()) / (1 << 20),
                alloc.name(), alloc.blocks());

    if (o.verify)
    {
        const double err = n->verify();
        std::printf("# verify    m_exec=%lld n_exec=%lld k_exec=%lld  max relative error %.3g -> %s\n",
                    static_cast<long long>(n->m_exec()), static_cast<long long>(n->n_exec()),
                    static_cast<long long>(n->k_exec()), err, err < 1e-10 ? "OK" : "FAILED");
        if (err >= 1e-10) return 1;
    }
    if (o.sum)
    {
        n->fill(1);
        n->run();
        std::printf("# sum       %016llx\n", static_cast<unsigned long long>(n->checksum()));
        n->fill(1);
    }

    auto region = [&] { n->run(); };
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
    const double work = n->work();
    const double gflops = work / ns_min;
    const double calls = n->calls();

    if (o.alloc_stats)
        std::printf("# allocator %s: %zu buffers, %.1f MiB\n", alloc.name(), alloc.blocks(),
                    static_cast<double>(alloc.bytes()) / (1 << 20));

    std::printf("label,m,n,k,top,calls,work,reps,inner,ns_min,ns_med,gflops\n");
    std::printf("%s,%lld,%lld,%lld,%s,%.0f,%.0f,%ld,%ld,%.1f,%.1f,%.3f\n",
                o.label.empty() ? "-" : o.label.c_str(), o.m, o.n, o.k,
                o.top.empty() ? "-" : o.top.c_str(), calls, work, reps, inner,
                ns_min, ns_med, gflops);
    std::printf("# %.2f GFLOP/s per run, %.3f ns per micro-kernel call, %ld reps x %ld runs\n",
                gflops, ns_min / std::max(1.0, calls / 1.0) / 1.0, reps, inner);
    return 0;
}
