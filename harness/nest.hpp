#pragma once
// A GEMM loop nest composed from rungs: any member of the MOMMS family
// (Smith and van de Geijn, SC19), Goto's among them.  The rungs come from
// --nest text, the same syntax model.py parses:
//
//     n:768 k:768 | m:120 k:192 | n:nr m:mr        B3A2C0
//     n:1920 k:256 m:96 n:nr m:mr                  Goto, A2C0
//
// outer to inner, ending in the two register loops (m:mr, n:nr, either order);
// '|' groups loops into a level, for the report.  A block may be a multiple of
// the tile, 4mr / 200nr.
//
// A rung may carry a packer for A or for B.  Where it sits decides only *when*
// the operand is packed -- on entry to that rung's iteration -- not what a
// packed block is: a block is always the operand cut by the innermost cache
// loops of its two dimensions, which is what the register loops read, and what
// the model calls the block resident at that level.  A packer above a deeper
// cut of its operand's dimensions therefore packs one block per such iteration,
// which is how the GOTO harness hoists packing above the timed region too; a
// packer above the region is packed once, before it.
//
// The two register loops are not walked by the recursion: the walk ends in a
// driver, one of two, that reads the packed panels and calls the micro-kernel.
// That keeps the per-tile cost at what a hand-written nest pays, while the
// cache-level composition stays data.

extern "C" {
#include <blis.h>
}

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allocator.hpp"
#include "packer.hpp"
#include "rung.hpp"

namespace momms {

/// where one operand's packed blocks come from
struct pack_provider
{
    packm_fn fn = nullptr;   // null: the packer's own copy loop
    void* ctx = nullptr;
};

struct nest_spec
{
    std::vector<loop_rung> loops;          // outer to inner, register loops last
    std::vector<std::vector<int>> groups;  // indices, per level ('|' groups)
    int pack_at[2] = {-2, -2};             // -2 unset, -1 'pre', else rung index

    std::vector<std::string> labels() const
    {
        std::vector<std::string> out;
        for (const auto& r : loops) out.push_back(r.label);
        return out;
    }
};

namespace detail {

inline dim_t block_of(const std::string& tok, dim_t mr, dim_t nr)
{
    std::string s;
    for (char c : tok) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    dim_t unit = 1;
    if (s.size() >= 2)
    {
        const std::string suf = s.substr(s.size() - 2);
        if (suf == "mr" || suf == "nr")
        {
            unit = suf == "mr" ? mr : nr;
            s.resize(s.size() - 2);
            if (s.empty()) return unit;
        }
    }
    if (s.empty()) throw std::invalid_argument("block " + tok + ": give N, Nmr or Nnr");
    std::size_t used = 0;
    long long v = 0;
    try { v = std::stoll(s, &used); } catch (const std::exception&) { used = 0; }
    if (used != s.size() || v <= 0)
        throw std::invalid_argument("block " + tok + ": give N, Nmr or Nnr");
    return static_cast<dim_t>(v) * unit;
}

inline int dim_of(const std::string& s)
{
    if (s == "m") return DIM_M;
    if (s == "n") return DIM_N;
    if (s == "k") return DIM_K;
    throw std::invalid_argument("loop " + s + ": the dimensions are m, n and k");
}

inline std::vector<std::string> words(std::string s, char sep = 0)
{
    std::vector<std::string> out;
    if (sep)
        for (char& c : s) if (c == sep) c = ' ';
    std::istringstream in(s);
    std::string t;
    while (in >> t) out.push_back(t);
    return out;
}

} // namespace detail

// Parse --nest and --pack-at.  Packing defaults to the loop around the first
// loop below the top that keeps the operand invariant -- Goto's ic and pc,
// B3A2C0's k2 and k1.
inline nest_spec parse_nest(const std::string& text, dim_t mr, dim_t nr,
                            const std::string& pack_at = "")
{
    nest_spec spec;
    std::vector<int> group;
    for (const std::string& tok : detail::words(text, ','))
    {
        if (tok == "|")
        {
            if (group.empty()) throw std::invalid_argument("--nest: empty level before |");
            spec.groups.push_back(group);
            group.clear();
            continue;
        }
        const std::size_t colon = tok.find(':');
        if (colon == std::string::npos)
            throw std::invalid_argument("loop " + tok + ": give dim:block, e.g. k:256, m:4mr or n:nr");
        loop_rung r;
        r.dim = detail::dim_of(tok.substr(0, colon));
        r.block = detail::block_of(tok.substr(colon + 1), mr, nr);
        spec.loops.push_back(r);
        group.push_back(static_cast<int>(spec.loops.size()) - 1);
    }
    if (!group.empty()) spec.groups.push_back(group);
    if (spec.loops.size() < 2)
        throw std::invalid_argument("--nest: give the loops, ending in m:mr and n:nr");
    if (spec.loops.size() > static_cast<std::size_t>(kMaxRungs))
        throw std::invalid_argument("--nest: too many loops");

    const loop_rung& a = spec.loops[spec.loops.size() - 2];
    const loop_rung& b = spec.loops[spec.loops.size() - 1];
    if (!((a.dim == DIM_M && a.block == mr && b.dim == DIM_N && b.block == nr)
          || (a.dim == DIM_N && a.block == nr && b.dim == DIM_M && b.block == mr)))
        throw std::invalid_argument("--nest: the nest must end in the register loops, m:" +
                                    std::to_string(mr) + " and n:" + std::to_string(nr) +
                                    " in either order");

    std::vector<int> cache;
    for (std::size_t i = 0; i + 2 < spec.loops.size(); i++) cache.push_back(spec.loops[i].dim);
    std::vector<std::string> labels;
    if (cache == std::vector<int>{DIM_N, DIM_K, DIM_M})
        labels = {"jc", "pc", "ic"};
    else
    {
        int seen[3] = {0, 0, 0};
        for (int d : cache) labels.push_back(std::string(dim_name(d)) + std::to_string(++seen[d]));
    }
    labels.push_back(a.dim == DIM_N ? "jr" : "ir");
    labels.push_back(b.dim == DIM_N ? "jr" : "ir");
    for (std::size_t i = 0; i < spec.loops.size(); i++) spec.loops[i].label = labels[i];

    for (const std::string& kv : detail::words(pack_at, ','))
    {
        const std::size_t c = kv.find(':');
        if (c == std::string::npos) throw std::invalid_argument("--pack-at: give A:LOOP or B:LOOP");
        std::string op = kv.substr(0, c), at = kv.substr(c + 1);
        for (char& ch : op) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        for (char& ch : at) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (op != "A" && op != "B") throw std::invalid_argument("--pack-at: only A and B are packed");
        const int which = op == "A" ? OP_A : OP_B;
        int r = -1;
        if (at != "pre")
        {
            r = -2;
            for (std::size_t i = 0; i < labels.size(); i++) if (labels[i] == at) r = static_cast<int>(i);
            if (r < 0) throw std::invalid_argument("--pack-at: no loop " + at);
            if (spec.loops[r].is_register())
                throw std::invalid_argument("--pack-at: " + at + " is a register loop");
        }
        spec.pack_at[which] = r;
    }
    for (int which = 0; which < 2; which++)
    {
        if (spec.pack_at[which] != -2) continue;
        int r = -1;    // 'pre' if no loop below the top keeps it invariant
        for (std::size_t i = 1; i < labels.size(); i++)
            if (invariant_of(spec.loops[i].dim) == which)
            {
                r = static_cast<int>(i) - 1;
                break;
            }
        // a register loop is inside the driver: packing there is 'pre'
        if (r >= 0 && spec.loops[r].is_register()) r = -1;
        spec.pack_at[which] = r;
    }
    for (int which = 0; which < 2; which++)
    {
        const int r = spec.pack_at[which];
        if (r >= 0) spec.loops[r].packs |= 1 << which;
    }
    return spec;
}

struct nest_config
{
    dim_t m = 0, n = 0, k = 0;
    dim_t lda = 0, ldb = 0;      // A is m x k, B is k x n, both column-major
    inc_t rs_c = 0, cs_c = 0;    // C, row- or column-major
    double alpha = 2.1, beta = 1.3;
    ptype dt = ptype::f64;       // the only type the bench itself fills and checks
    int top = 0;                 // outermost rung that iterates
};

class nest
{
public:
    nest(const ukernel& u, const nest_spec& spec, const nest_config& cfg,
         allocator& alloc, pack_provider pack_a = {}, pack_provider pack_b = {})
        : u_(u), spec_(spec), cfg_(cfg), alloc_(alloc)
    {
        rungs_ = spec_.loops;
        n_ = static_cast<int>(rungs_.size());
        if (n_ > kMaxRungs) throw std::invalid_argument("nest: too many rungs");
        reg_first_ = n_ - 2;
        jr_outer_ = rungs_[reg_first_].dim == DIM_N;
        idx_jr_ = index_of("jr");
        idx_ir_ = index_of("ir");

        // the loops' maximum extent and trip counts, cutting dimension by dimension
        dim_t ext[3] = {cfg_.m, cfg_.n, cfg_.k};
        trips_.assign(n_, 1);
        for (int i = 0; i < n_; i++)
        {
            const int d = rungs_[i].dim;
            trips_[i] = ndiv(ext[d], rungs_[i].block);
            ext[d] = std::min(rungs_[i].block, ext[d]);
        }
        // what one iteration moves: C, and the sources the packers read
        for (loop_rung& r : rungs_)
        {
            if (r.dim == DIM_M) r.delta_c = static_cast<std::ptrdiff_t>(r.block) * cfg_.rs_c;
            if (r.dim == DIM_N) r.delta_c = static_cast<std::ptrdiff_t>(r.block) * cfg_.cs_c;
            // A is m x k column-major: m has stride 1, k has stride lda.  B is
            // k x n column-major: k has stride 1, n has stride ldb.
            if (r.dim == DIM_M) r.src_delta[OP_A] = r.block;
            if (r.dim == DIM_K) r.src_delta[OP_A] = static_cast<std::ptrdiff_t>(r.block) * cfg_.lda;
            if (r.dim == DIM_N) r.src_delta[OP_B] = static_cast<std::ptrdiff_t>(r.block) * cfg_.ldb;
            if (r.dim == DIM_K) r.src_delta[OP_B] = r.block;
            if (r.is_register())
            {
                r.delta_c = 0;
                r.src_delta[OP_A] = r.src_delta[OP_B] = 0;
            }
        }

        // the packers: one block per deeper cache cut of the operand's dimensions
        for (int op = 0; op < 2; op++)
        {
            pack_rung_[op] = spec_.pack_at[op];
            std::vector<int> deeper;
            const int* ds = dims_of(op);
            for (int i = 0; i < reg_first_; i++)
                if (i > pack_rung_[op] && (rungs_[i].dim == ds[0] || rungs_[i].dim == ds[1]))
                    deeper.push_back(i);
            slot_rungs_[op] = deeper;
            std::size_t slots = 1;
            for (int i : deeper) slots *= static_cast<std::size_t>(trips_[i]);

            packer::shape sh;
            sh.panel = op == OP_A ? u_.mr : u_.nr;
            sh.bcast = 1;
            sh.len = deepest_block(op, DIM_K);
            sh.extent = deepest_block(op, op == OP_A ? DIM_M : DIM_N);
            sh.dt = cfg_.dt;
            sh.slots = slots;
            const pack_provider& pp = op == OP_A ? pack_a : pack_b;
            pack_[op] = std::make_unique<packer>(sh, pp.fn, pp.ctx, alloc);
        }

        // which rungs change each operand's packed block
        for (int op = 0; op < 2; op++)
            for (int i : slot_rungs_[op]) slot_touch_[op][i] = true;

        const std::size_t ea = elem_size(u_.dt_a), eb = elem_size(u_.dt_b),
                          ec = elem_size(u_.dt_c);
        a_ = alloc_.allocate(static_cast<std::size_t>(cfg_.m) * cfg_.k * ea);
        b_ = alloc_.allocate(static_cast<std::size_t>(cfg_.k) * cfg_.n * eb);
        c_ = alloc_.allocate(static_cast<std::size_t>(cfg_.m) * cfg_.n * ec);

        measure();
        fill(1);
        set_aux();
    }

    nest(const nest&) = delete;
    nest& operator=(const nest&) = delete;

    ~nest()
    {
        alloc_.release(a_);
        alloc_.release(b_);
        alloc_.release(c_);
    }

    // ------------------------------------------------------------ running ---

    /// Pack what the region will not pack itself: every packer above --top,
    /// and every 'pre' one.  Called by run() if it has not been called.
    void prepack()
    {
        for (int op = 0; op < 2; op++)
            if (pack_rung_[op] < 0) pack_all(op, initial_state());
        state s = initial_state();
        for (int i = 0; i < cfg_.top && i < reg_first_; i++)
        {
            const loop_rung& r = rungs_[i];
            s.ext[r.dim] = segment(s.ext[r.dim], r.block, 0);
            s.idx[i] = 0;
            for (int op = 0; op < 2; op++)
                if (r.packs & (1 << op)) pack_all(op, s);
        }
        prepacked_ = true;
    }

    /// One execution of the region: from --top down.
    void run()
    {
        if (!prepacked_) prepack();
        descend(0, initial_state());
    }

    // --------------------------------------------------------------- data ---

    double* a() const { return static_cast<double*>(a_); }
    double* b() const { return static_cast<double*>(b_); }
    double* c() const { return static_cast<double*>(c_); }
    const ukernel& kernel() const { return u_; }
    const nest_spec& spec() const { return spec_; }
    int rungs() const { return n_; }
    int top() const { return cfg_.top; }
    dim_t m_exec() const { return exec_[DIM_M]; }
    dim_t n_exec() const { return exec_[DIM_N]; }
    dim_t k_exec() const { return exec_[DIM_K]; }
    double calls() const { return calls_; }
    double work() const { return 2.0 * m_exec() * n_exec() * k_exec(); }
    dim_t mr() const { return u_.mr; }
    dim_t nr() const { return u_.nr; }

    /// elements of one packed block, and how many the operand keeps
    std::size_t block_elems(int op) const { return pack_[op]->block_elems(); }
    std::size_t slots(int op) const { return pack_[op]->slots(); }
    int pack_rung(int op) const { return pack_rung_[op]; }

    std::size_t footprint_bytes() const
    {
        return static_cast<std::size_t>(cfg_.m) * cfg_.k * elem_size(u_.dt_a)
             + static_cast<std::size_t>(cfg_.k) * cfg_.n * elem_size(u_.dt_b)
             + static_cast<std::size_t>(cfg_.m) * cfg_.n * elem_size(u_.dt_c)
             + pack_[OP_A]->block_bytes() * pack_[OP_A]->slots()
             + pack_[OP_B]->block_bytes() * pack_[OP_B]->slots();
    }

    void fill(unsigned seed)
    {
        if (cfg_.dt != ptype::f64 || u_.dt_a != ptype::f64 || u_.dt_b != ptype::f64
            || u_.dt_c != ptype::f64)
            throw std::invalid_argument("fill: the bench fills doubles; a mixed-precision "
                                        "nest needs its own generator");
        std::size_t na = static_cast<std::size_t>(cfg_.m) * cfg_.k;
        std::size_t nb = static_cast<std::size_t>(cfg_.k) * cfg_.n;
        std::size_t nc = static_cast<std::size_t>(cfg_.m) * cfg_.n;
        for (std::size_t i = 0; i < na; i++) a()[i] = value(i, seed);
        for (std::size_t i = 0; i < nb; i++) b()[i] = value(i, seed + 1);
        for (std::size_t i = 0; i < nc; i++) c()[i] = value(i, seed + 2);
    }

    /// A hash of C, for comparing two harnesses bit for bit.
    std::uint64_t checksum() const
    {
        const std::uint64_t* p = reinterpret_cast<const std::uint64_t*>(c());
        const std::size_t n64 = static_cast<std::size_t>(cfg_.m) * cfg_.n *
                                elem_size(u_.dt_c) / sizeof(std::uint64_t);
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < n64; i++)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    /// Run once from the current C, compare the region with
    /// beta*C0 + alpha*A*B over m_exec x n_exec x k_exec from a plain loop, and
    /// restore C; nothing outside the region may change.
    double verify()
    {
        if (cfg_.dt != ptype::f64 || u_.dt_a != ptype::f64 || u_.dt_b != ptype::f64
            || u_.dt_c != ptype::f64)
            throw std::invalid_argument("verify: the reference here is double");
        const std::size_t mn = static_cast<std::size_t>(cfg_.m) * cfg_.n;
        std::vector<double> c0(c(), c() + mn);
        run();
        const dim_t me = m_exec(), ne = n_exec(), ke = k_exec();
        double worst = 0.0;
        std::vector<double> acc(static_cast<std::size_t>(me));
        for (dim_t j = 0; j < ne; j++)
        {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (dim_t p = 0; p < ke; p++)
            {
                const double bpj = b()[p + j * cfg_.ldb];
                const double* ap = a() + p * cfg_.lda;
                for (dim_t i = 0; i < me; i++) acc[static_cast<std::size_t>(i)] += ap[i] * bpj;
            }
            for (dim_t i = 0; i < me; i++)
            {
                const std::size_t ij = static_cast<std::size_t>(i * cfg_.rs_c + j * cfg_.cs_c);
                const double ref = cfg_.beta * c0[ij] + cfg_.alpha * acc[static_cast<std::size_t>(i)];
                worst = std::max(worst, std::fabs(c()[ij] - ref) / (std::fabs(ref) + 1.0));
            }
        }
        for (dim_t i = 0; i < cfg_.m; i++)
            for (dim_t j = 0; j < cfg_.n; j++)
            {
                const std::size_t ij = static_cast<std::size_t>(i * cfg_.rs_c + j * cfg_.cs_c);
                if ((i >= me || j >= ne) && c()[ij] != c0[ij]) return 1.0;
            }
        std::copy(c0.begin(), c0.end(), c());
        return worst;
    }

    /// The nest as --nest takes it, with '|' between levels.
    std::string text() const
    {
        std::ostringstream s;
        for (int i = 0; i < n_; i++)
        {
            const loop_rung& r = rungs_[i];
            bool boundary = false;
            for (const auto& g : spec_.groups)
                if (g.size() && g[0] == i && i > 0) boundary = true;
            if (i) s << (boundary ? " | " : " ");
            s << dim_name(r.dim) << ':';
            if (r.is_register()) s << (r.dim == DIM_M ? "mr" : "nr");
            else s << r.block;
        }
        return s.str();
    }

    std::string pack_text() const
    {
        std::ostringstream s;
        for (int op = 0; op < 2; op++)
        {
            if (op) s << "  ";
            s << op_name(op) << ':';
            if (pack_rung_[op] >= 0) s << rungs_[pack_rung_[op]].label;
            else s << "pre";
        }
        return s.str();
    }

    std::string loops_text() const
    {
        std::ostringstream s;
        for (int i = 0; i < n_; i++)
        {
            if (i) s << ' ';
            s << rungs_[i].label;
        }
        return s.str();
    }

private:
    static double value(std::size_t i, unsigned seed)
    {
        return 0.25 * static_cast<double>(static_cast<int>((i * 2654435761u + seed * 97u) % 9u) - 4);
    }

    state initial_state() const
    {
        state s;
        s.ext[DIM_M] = cfg_.m;
        s.ext[DIM_N] = cfg_.n;
        s.ext[DIM_K] = cfg_.k;
        return s;
    }

    int index_of(const std::string& label) const
    {
        for (int i = 0; i < n_; i++) if (rungs_[i].label == label) return i;
        return -1;
    }

    int deepest_cut(int op, int dim) const
    {
        const int* ds = dims_of(op);
        if (ds[0] != dim && ds[1] != dim) return -1;
        int found = -1;
        for (int i = 0; i < reg_first_; i++)
            if (rungs_[i].dim == dim) found = i;
        return found;
    }

    dim_t deepest_block(int op, int dim) const
    {
        const int i = deepest_cut(op, dim);
        if (i >= 0) return rungs_[i].block;
        return dim == DIM_M ? cfg_.m : dim == DIM_N ? cfg_.n : cfg_.k;
    }

    /// which packed block the current indices select
    std::size_t slot_of(int op, const std::size_t* idx) const
    {
        std::size_t s = 0;
        for (int i : slot_rungs_[op]) s = s * static_cast<std::size_t>(trips_[i]) + idx[i];
        return s;
    }

    /// absolute element offset into the source matrix of the block at `idx`
    std::ptrdiff_t src_offset(int op, const std::size_t* idx) const
    {
        std::ptrdiff_t off = 0;
        for (int i = 0; i < reg_first_; i++)
            off += static_cast<std::ptrdiff_t>(idx[i]) * rungs_[i].src_delta[op];
        return off;
    }

    // Pack every block the packer of `op` owns for this parent: one per
    // combination of the deeper cuts of the operand's dimensions.
    void pack_all(int op, const state& s)
    {
        const std::vector<int>& deeper = slot_rungs_[op];
        const int* ds = dims_of(op);
        const int panel_dim = ds[0] == DIM_M ? DIM_M : DIM_N;
        const char* base = static_cast<const char*>(op == OP_A ? a_ : b_);
        const std::size_t es = elem_size(op == OP_A ? u_.dt_a : u_.dt_b);
        const inc_t inc = op == OP_A ? 1 : cfg_.ldb;
        const inc_t ld = op == OP_A ? cfg_.lda : 1;
        std::size_t n_comb = 1;
        for (int j : deeper) n_comb *= static_cast<std::size_t>(trips_[j]);
        state t = s;
        for (std::size_t c = 0; c < n_comb; c++)
        {
            std::size_t rest = c;      // odometer, innermost cut fastest
            for (std::size_t k = deeper.size(); k-- > 0;)
            {
                const int j = deeper[k];
                t.idx[j] = rest % static_cast<std::size_t>(trips_[j]);
                rest /= static_cast<std::size_t>(trips_[j]);
            }
            dim_t e[3] = {s.ext[0], s.ext[1], s.ext[2]};
            bool valid = true;
            for (int j = pack_rung_[op] + 1; j < reg_first_ && valid; j++)
                for (int d = 0; d < 2 && valid; d++)
                    if (rungs_[j].dim == ds[d])
                    {
                        // a partial parent has fewer blocks than the maximum
                        const dim_t cnt = ndiv(e[ds[d]], rungs_[j].block);
                        if (t.idx[j] >= static_cast<std::size_t>(cnt)) { valid = false; break; }
                        e[ds[d]] = segment(e[ds[d]], rungs_[j].block, t.idx[j]);
                    }
            if (!valid) continue;      // the slot is not used by this parent
            const std::ptrdiff_t off = src_offset(op, t.idx);
            pack_[op]->pack(slot_of(op, t.idx), e[panel_dim], e[DIM_K],
                            base + off * static_cast<std::ptrdiff_t>(es), inc, ld);
        }
    }

    // ------------------------------------------------------------- walking ---

    void descend(int i, state s, bool dry = false)
    {
        if (i == reg_first_)
        {
            drive(s, dry);
            return;
        }
        const loop_rung& r = rungs_[i];
        const dim_t trips = (i < cfg_.top) ? 1 : ndiv(s.ext[r.dim], r.block);
        for (dim_t t = 0; t < trips; t++)
        {
            state next = s;
            next.idx[i] = static_cast<std::size_t>(t);
            next.ext[r.dim] = segment(s.ext[r.dim], r.block, t);
            next.off_c = s.off_c + static_cast<std::ptrdiff_t>(t) * r.delta_c;
            if (r.dim == DIM_K && t != 0) next.first_k = false;
            for (int op = 0; op < 2; op++)
                if (slot_touch_[op][i]) next.slot[op] = slot_of(op, next.idx);
            // above the region the packers were prepacked; inside it, pack here
            if (!dry && r.packs && i >= cfg_.top)
                for (int op = 0; op < 2; op++)
                    if (r.packs & (1 << op)) pack_all(op, next);
            descend(i + 1, next, dry);
        }
    }

    // The register loops and the micro-kernel call: the leaf of the nest.
    void drive(state& s, bool dry)
    {
        const dim_t n_jr = (idx_jr_ >= cfg_.top) ? ndiv(s.ext[DIM_N], u_.nr) : 1;
        const dim_t n_ir = (idx_ir_ >= cfg_.top) ? ndiv(s.ext[DIM_M], u_.mr) : 1;
        const dim_t k = s.ext[DIM_K];
        if (dry)
        {
            calls_ += static_cast<double>(n_jr) * static_cast<double>(n_ir);
            return;
        }
        // beta multiplies C on the first k block of a tile, whichever loops cut k
        const double beta = s.first_k ? cfg_.beta : 1.0;
        double* cblk = c() + s.off_c;
        const dim_t ps_a = pack_[OP_A]->panel_stride();
        const dim_t ps_b = pack_[OP_B]->panel_stride();
        const char* a_blk = static_cast<const char*>(pack_[OP_A]->buffer(s.slot[OP_A]));
        const char* b_blk = static_cast<const char*>(pack_[OP_B]->buffer(s.slot[OP_B]));
        const std::ptrdiff_t ea = static_cast<std::ptrdiff_t>(elem_size(u_.dt_a));
        const std::ptrdiff_t eb = static_cast<std::ptrdiff_t>(elem_size(u_.dt_b));
        if (jr_outer_)
        {
            for (dim_t jr = 0; jr < n_jr; jr++)
            {
                const dim_t n_r = segment(s.ext[DIM_N], u_.nr, jr);
                const char* b1 = b_blk + static_cast<std::ptrdiff_t>(jr) * ps_b * eb;
                for (dim_t ir = 0; ir < n_ir; ir++)
                {
                    const dim_t m_r = segment(s.ext[DIM_M], u_.mr, ir);
                    const char* a1 = a_blk + static_cast<std::ptrdiff_t>(ir) * ps_a * ea;
                    const bool last_ir = ir == n_ir - 1;
                    bli_auxinfo_set_next_a(const_cast<char*>(
                        last_ir ? a_blk : a1 + ps_a * ea), &aux_);
                    bli_auxinfo_set_next_b(const_cast<char*>(last_ir
                        ? (jr == n_jr - 1 ? b_blk : b1 + ps_b * eb) : b1), &aux_);
                    double* c11 = cblk + ir * u_.mr * cfg_.rs_c + jr * u_.nr * cfg_.cs_c;
                    u_.fn(u_.ctx, m_r, n_r, k, &cfg_.alpha, a1, b1, &beta, c11,
                          cfg_.rs_c, cfg_.cs_c, &aux_);
                }
            }
        }
        else
        {
            for (dim_t ir = 0; ir < n_ir; ir++)
            {
                const dim_t m_r = segment(s.ext[DIM_M], u_.mr, ir);
                const char* a1 = a_blk + static_cast<std::ptrdiff_t>(ir) * ps_a * ea;
                for (dim_t jr = 0; jr < n_jr; jr++)
                {
                    const dim_t n_r = segment(s.ext[DIM_N], u_.nr, jr);
                    const char* b1 = b_blk + static_cast<std::ptrdiff_t>(jr) * ps_b * eb;
                    const bool last_jr = jr == n_jr - 1;
                    bli_auxinfo_set_next_b(const_cast<char*>(
                        last_jr ? b_blk : b1 + ps_b * eb), &aux_);
                    bli_auxinfo_set_next_a(const_cast<char*>(last_jr
                        ? (ir == n_ir - 1 ? a_blk : a1 + ps_a * ea) : a1), &aux_);
                    double* c11 = cblk + ir * u_.mr * cfg_.rs_c + jr * u_.nr * cfg_.cs_c;
                    u_.fn(u_.ctx, m_r, n_r, k, &cfg_.alpha, a1, b1, &beta, c11,
                          cfg_.rs_c, cfg_.cs_c, &aux_);
                }
            }
        }
    }

    void set_aux()
    {
        bli_auxinfo_set_is_a(1, &aux_);
        bli_auxinfo_set_is_b(1, &aux_);
#if defined(AOCL_BLIS)
        bli_auxinfo_set_schema_a(BLIS_PACKED_ROW_PANELS, &aux_);
        bli_auxinfo_set_schema_b(BLIS_PACKED_COL_PANELS, &aux_);
#else
        bli_auxinfo_set_schema_a(BLIS_PACKED_PANELS, &aux_);
        bli_auxinfo_set_schema_b(BLIS_PACKED_PANELS, &aux_);
#endif
        bli_auxinfo_set_ps_a(pack_[OP_A]->panel_stride(), &aux_);
        bli_auxinfo_set_ps_b(pack_[OP_B]->panel_stride(), &aux_);
    }

    void measure()
    {
        // the region covers everything at and below --top, and one iteration
        // of each loop above it: so its extents are what those cuts leave
        state s = initial_state();
        for (int i = 0; i < cfg_.top && i < reg_first_; i++)
            s.ext[rungs_[i].dim] = segment(s.ext[rungs_[i].dim], rungs_[i].block, 0);
        exec_[DIM_M] = s.ext[DIM_M];
        exec_[DIM_N] = s.ext[DIM_N];
        exec_[DIM_K] = s.ext[DIM_K];
        calls_ = 0.0;
        descend(0, initial_state(), true);
    }

    ukernel u_;
    nest_spec spec_;
    nest_config cfg_;
    allocator& alloc_;
    std::vector<loop_rung> rungs_;
    int n_ = 0, reg_first_ = 0, idx_jr_ = 0, idx_ir_ = 1;
    bool jr_outer_ = true, prepacked_ = false;
    std::vector<dim_t> trips_;
    int pack_rung_[2] = {-1, -1};
    std::vector<int> slot_rungs_[2];
    bool slot_touch_[2][kMaxRungs] = {};
    std::unique_ptr<packer> pack_[2];
    void* a_ = nullptr;
    void* b_ = nullptr;
    void* c_ = nullptr;
    dim_t exec_[3] = {0, 0, 0};
    double calls_ = 0.0;
    auxinfo_t aux_{};
};

} // namespace momms
