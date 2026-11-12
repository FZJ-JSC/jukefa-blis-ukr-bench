#pragma once
// The engine: a GEMM loop nest composed from rungs, the walk over it, the
// packers, and the micro-kernel call at the leaf.
//
// The rungs come from --nest text, the same syntax model.py parses:
//
//     n:768 k:768 | m:120 k:192 | n:nr m:mr        B3A2C0 (MOMMS)
//     n:1920 k:256 m:96 n:nr m:mr                  Goto, A2C0 (one member)
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
// cut of its operand's dimensions therefore packs one block per such iteration;
// a packer above the region is packed once, before it.
//
// The two register loops are not walked by the recursion: the walk ends in a
// driver, one of two, that reads the packed panels and calls the provider's
// micro-kernel.  That keeps the per-tile cost at what a hand-written nest pays,
// while the cache-level composition stays data.
//
// Nothing here knows about BLIS, ukrgen or any other kernel family: the leaf is
// a ukr_provider, which also owns the packed layout's shaping.
//
// The nesting is MOMMS's (Smith and van de Geijn, SC19) after the theory of
// Smith, Lowery, Langou and van de Geijn; Goto's algorithm is one member.

#include <algorithm>
#include <array>
#include <chrono>
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
#include "types.hpp"
#include "ukr_provider.hpp"

namespace mmgen {

/// the GEMM's shape
struct problem
{
    index_t m = 0, n = 0, k = 0;
};

/// One GEMM call: the matrices, how they are laid out, and the scalars.
/// A is m x k, B is k x n, C is m x n; every operand is (row, column)-strided,
/// so column-major A is (1, lda) and row-major C is (n, 1).
struct call
{
    const void* a = nullptr;
    stride_t lda = 1;
    const void* b = nullptr;
    stride_t ldb = 1;
    void* c = nullptr;
    stride_t rs_c = 0, cs_c = 1;
    double alpha = 1.0, beta = 0.0;
};

struct nest_spec
{
    std::vector<loop_rung> loops;          // outer to inner, register loops last
    std::vector<std::vector<int>> groups;  // indices, per level ('|' groups)
    std::array<int, 2> pack_at = {-2, -2}; // -2 unset, -1 'pre', else rung index

    std::vector<std::string> labels() const
    {
        std::vector<std::string> out;
        for (const auto& r : loops) out.push_back(r.label);
        return out;
    }

    /// where `op` is packed: -1 for 'pre'
    int packed_at(operand op) const noexcept { return pack_at[packed_index(op)]; }
};

/// what a plan records when it is built with profiling on
struct profile
{
    bool on = false;
    std::size_t regions = 0;            // calls to run()
    std::size_t calls = 0;              // micro-kernel calls
    std::size_t pack_calls = 0;         // packer invocations
    std::size_t pack_elems = 0;         // elements packed
    double total_ns = 0.0;              // inside run()
    double pack_ns = 0.0;               // inside the packers
    std::vector<std::size_t> iteration; // per rung, summed over regions

    void reset()
    {
        const std::size_t n = iteration.size();
        *this = profile{};
        iteration.assign(n, 0);   // one slot per rung, kept
        on = true;
    }
};

struct nest_config
{
    int top = 0;          // outermost rung that iterates
    bool profile = false;
};

namespace detail {

inline index_t block_of(const std::string& tok, index_t mr, index_t nr)
{
    std::string s;
    for (char c : tok) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    index_t unit = 1;
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
    return static_cast<index_t>(v) * unit;
}

inline dimension dimension_from_string(const std::string& s)
{
    if (s == "m") return dimension::M;
    if (s == "n") return dimension::N;
    if (s == "k") return dimension::K;
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
// B3A2C0's k2 and k1.  A register loop is inside the driver, so a default that
// would land there falls back to 'pre'.
nest_spec parse_nest(const std::string& text, index_t mr, index_t nr,
                     const std::string& pack_at = "");

class nest
{
public:
    nest(ukr_provider& ukr, const nest_spec& spec, const problem& p, buffer_allocator& alloc)
        : ukr_(ukr), spec_(spec), psize_(p), allocator_(alloc)
    {
        rungs_ = spec_.loops;
        rung_count_ = static_cast<int>(rungs_.size());
        if (rung_count_ < 2 || rung_count_ > MAX_RUNGS)
            throw std::invalid_argument("nest: bad rung count");
        reg_first_ = rung_count_ - 2;
        jr_outer_ = rungs_[reg_first_].dim == dimension::N;
        idx_jr_ = index_of("jr");
        idx_ir_ = index_of("ir");
        if (idx_jr_ < 0 || idx_ir_ < 0) throw std::invalid_argument("nest: needs jr and ir");

        // the loops' largest extent and trip counts, cutting dimension by dimension
        std::array<index_t, 3> ext = {psize_.m, psize_.n, psize_.k};
        auto dimext = [&ext](dimension dim) -> index_t&
        {
            return index_by_dim<index_t>(ext, dim);
        };
        trips_.assign(rung_count_, 1);
        for (int i = 0; i < rung_count_; i++)
        {
            const dimension d = rungs_[i].dim;
            trips_[i] = ndiv(dimext(d), rungs_[i].block);
            dimext(d) = std::min(rungs_[i].block, dimext(d));
        }

        // the packers: one block per deeper cache cut of the operand's dimensions
        const kernel_info& ki = ukr_.info();
        for (std::size_t o = 0; o + 1 < operands.size(); o++)
        {
            const operand op = operands[o];
            pack_rung_[o] = spec_.packed_at(op);
            std::vector<int> deeper;
            const auto ds = dims_of(op);
            for (int j = 0; j < reg_first_; j++)
                if (j > pack_rung_[o] && (rungs_[j].dim == ds[0] || rungs_[j].dim == ds[1]))
                    deeper.push_back(j);
            slot_rungs_[o] = deeper;
            std::size_t slots = 1;
            for (int j : deeper) slots *= static_cast<std::size_t>(trips_[j]);

            packer::shape sh;
            sh.panel = op == operand::A ? ki.mr : ki.nr;
            sh.bcast = 1;
            sh.len = deepest_block(op, dimension::K);
            sh.extent = deepest_block(op, op == operand::A ? dimension::M : dimension::N);
            sh.dt = op == operand::A ? ki.dt_a : ki.dt_b;
            sh.slots = slots;
            const pack_kernel pk = ukr_.packer(op);
            pack_[o] = std::make_unique<packer>(sh, pk.fn, pk.ctx, allocator_);
        }
        for (std::size_t o = 0; o + 1 < operands.size(); o++)
            for (int i : slot_rungs_[o]) slot_touch_[o][i] = true;

        ukr_.begin_plan(pack_[packed_index(operand::A)]->layout(),
                        pack_[packed_index(operand::B)]->layout());
        measure();
    }

    nest(const nest&) = delete;
    nest& operator=(const nest&) = delete;

    // ------------------------------------------------------------ running ---

    /// Pack what the region will not pack itself: every packer above --top, and
    /// every 'pre' one.  run() calls it when the matrices change.
    void prepack(const call& k)
    {
        set_strides(k);
        auto t0 = clock::now();
        for (std::size_t o = 0; o + 1 < operands.size(); o++)
            if (pack_rung_[o] < 0) pack_all(operands[o], initial_state(), k);
        state s = initial_state();
        for (int i = 0; i < config_.top && i < reg_first_; i++)
        {
            const loop_rung& r = rungs_[i];
            s.extent(r.dim) = segment(s.extent(r.dim), r.block, 0);
            s.idx[i] = 0;
            for (std::size_t o = 0; o + 1 < operands.size(); o++)
                if (r.packs_op(operands[o])) pack_all(operands[o], s, k);
        }
        if (config_.profile) pfstats_.pack_ns += ns_since(t0);
        prepacked_a_ = k.a;
        prepacked_b_ = k.b;
    }

    /// One execution of the region: from --top down, on these matrices.
    void run(const call& k)
    {
        set_strides(k);
        auto t0 = clock::now();
        if (prepacked_a_ != k.a || prepacked_b_ != k.b || !prepacked_)
        {
            prepack(k);
            prepacked_ = true;
        }
        descend(0, initial_state(), k);
        if (config_.profile)
        {
            pfstats_.total_ns += ns_since(t0);
            pfstats_.regions += 1;
        }
    }

    // --------------------------------------------------------------- data ---

    const kernel_info& kernel() const { return ukr_.info(); }
    const nest_spec& spec() const { return spec_; }
    const problem& size() const { return psize_; }
    int rung_count() const { return rung_count_; }
    int top() const { return config_.top; }
    void set_top(int t) { config_.top = t; measure(); }
    void set_profiling(bool on)
    {
        config_.profile = on;
        pfstats_.on = on;
        if (on && pfstats_.iteration.size() != static_cast<std::size_t>(rung_count_))
            pfstats_.iteration.assign(static_cast<std::size_t>(rung_count_), 0);
    }

    index_t m_exec() const { return index_by_dim<index_t>(exec_, dimension::M); }
    index_t n_exec() const { return index_by_dim<index_t>(exec_, dimension::N); }
    index_t k_exec() const { return index_by_dim<index_t>(exec_, dimension::K); }
    double calls() const { return calls_; }
    double work() const { return 2.0 * m_exec() * n_exec() * k_exec(); }
    index_t mr() const { return ukr_.info().mr; }
    index_t nr() const { return ukr_.info().nr; }

    /// elements of one packed block, the blocks kept per operand, and the
    /// bytes those buffers take
    std::size_t block_elems(operand op) const { return pack_[packed_index(op)]->block_elems(); }
    std::size_t slots(operand op) const { return pack_[packed_index(op)]->slots(); }
    int pack_rung(operand op) const { return pack_rung_[packed_index(op)]; }
    std::size_t pack_bytes() const
    {
        std::size_t total = 0;
        for (std::size_t o = 0; o + 1 < operands.size(); o++)
            total += pack_[o]->block_bytes() * pack_[o]->slots();
        return total;
    }

    const profile& stats() const { return pfstats_; }
    profile& stats() { return pfstats_; }

    /// The nest as --nest takes it, with '|' between levels.
    std::string text() const
    {
        std::ostringstream s;
        for (int i = 0; i < rung_count_; i++)
        {
            const loop_rung& r = rungs_[i];
            bool boundary = false;
            for (const auto& g : spec_.groups)
                if (g.size() && g[0] == i && i > 0) boundary = true;
            if (i) s << (boundary ? " | " : " ");
            s << dim_key(r.dim) << ':';
            if (r.is_register()) s << (r.dim == dimension::M ? "mr" : "nr");
            else s << r.block;
        }
        return s.str();
    }

    std::string pack_text() const
    {
        std::ostringstream s;
        for (std::size_t o = 0; o + 1 < operands.size(); o++)
        {
            if (o) s << "  ";
            s << op_name(operands[o]) << ':';
            if (pack_rung_[o] >= 0) s << rungs_[pack_rung_[o]].label;
            else s << "pre";
        }
        return s.str();
    }

    std::string loops_text() const
    {
        std::ostringstream s;
        for (int i = 0; i < rung_count_; i++)
        {
            if (i) s << ' ';
            s << rungs_[i].label;
        }
        return s.str();
    }

private:
    using clock = std::chrono::steady_clock;
    static double ns_since(clock::time_point t0)
    {
        return std::chrono::duration<double, std::nano>(clock::now() - t0).count();
    }

    state initial_state() const
    {
        state s;
        s.extent(dimension::M) = psize_.m;
        s.extent(dimension::N) = psize_.n;
        s.extent(dimension::K) = psize_.k;
        return s;
    }

    int index_of(const std::string& label) const
    {
        for (int i = 0; i < rung_count_; i++) if (rungs_[i].label == label) return i;
        return -1;
    }

    int deepest_cut(operand op, dimension dim) const
    {
        const auto ds = dims_of(op);
        if (ds[0] != dim && ds[1] != dim) return -1;
        int found = -1;
        for (int i = 0; i < reg_first_; i++)
            if (rungs_[i].dim == dim) found = i;
        return found;
    }

    index_t deepest_block(operand op, dimension dim) const
    {
        const int i = deepest_cut(op, dim);
        if (i >= 0) return rungs_[i].block;
        return select_by_dim(psize_.m, psize_.n, psize_.k, dim);
    }

    /// which packed block the current indices select
    std::size_t slot_of(operand op, const std::array<std::size_t, MAX_RUNGS>& idx) const
    {
        std::size_t s = 0;
        for (int i : slot_rungs_[packed_index(op)])
            s = s * static_cast<std::size_t>(trips_[i]) + idx[i];
        return s;
    }

    /// element offset into the source of the block at `idx`
    stride_t src_offset(operand op, const std::array<std::size_t, MAX_RUNGS>& idx) const
    {
        stride_t off = 0;
        const std::size_t o = packed_index(op);
        for (int i = 0; i < reg_first_; i++)
            off += static_cast<stride_t>(idx[i]) * rungs_[i].src_delta[o];
        return off;
    }

    /// what one iteration of each rung moves, for this call's layouts
    void set_strides(const call& k)
    {
        for (loop_rung& r : rungs_)
        {
            r.delta_c = 0;
            r.src_delta = {0, 0};
            if (r.is_register()) continue;
            if (r.dim == dimension::M) r.delta_c = static_cast<stride_t>(r.block) * k.rs_c;
            if (r.dim == dimension::N) r.delta_c = static_cast<stride_t>(r.block) * k.cs_c;
            // A is m x k column-major: m has stride 1, k has stride lda.  B is
            // k x n column-major: k has stride 1, n has stride ldb.
            if (r.dim == dimension::M) r.src_delta[packed_index(operand::A)] = r.block;
            if (r.dim == dimension::K)
                r.src_delta[packed_index(operand::A)] = static_cast<stride_t>(r.block) * k.lda;
            if (r.dim == dimension::N)
                r.src_delta[packed_index(operand::B)] = static_cast<stride_t>(r.block) * k.ldb;
            if (r.dim == dimension::K) r.src_delta[packed_index(operand::B)] = r.block;
        }
    }

    // Pack every block the packer of `op` owns for this parent: one per
    // combination of the deeper cuts of the operand's dimensions.
    void pack_all(operand op, const state& s, const call& k)
    {
        auto t0 = clock::now();
        const std::size_t o = packed_index(op);
        const std::vector<int>& deeper = slot_rungs_[o];
        const auto ds = dims_of(op);
        const dimension panel_dim = ds[0] == dimension::M ? dimension::M : dimension::N;
        const char* base = static_cast<const char*>(op == operand::A ? k.a : k.b);
        const kernel_info& ki = ukr_.info();
        const std::size_t es = elem_size(op == operand::A ? ki.dt_a : ki.dt_b);
        const stride_t inc = op == operand::A ? 1 : k.ldb;
        const stride_t ld = op == operand::A ? k.lda : 1;
        std::size_t n_comb = 1;
        for (int j : deeper) n_comb *= static_cast<std::size_t>(trips_[j]);
        state t = s;
        for (std::size_t c = 0; c < n_comb; c++)
        {
            std::size_t rest = c;      // odometer, innermost cut fastest
            for (std::size_t j = deeper.size(); j-- > 0;)
            {
                const int i = deeper[j];
                t.idx[i] = rest % static_cast<std::size_t>(trips_[i]);
                rest /= static_cast<std::size_t>(trips_[i]);
            }
            std::array<index_t, 3> e = {s.extent(dimension::M), s.extent(dimension::N),
                                        s.extent(dimension::K)};
            auto ee = [&e](dimension dim) -> index_t&
            {
                return index_by_dim<index_t>(e, dim);
            };
            bool valid = true;
            for (int j = pack_rung_[o] + 1; j < reg_first_ && valid; j++)
                for (int d = 0; d < 2 && valid; d++)
                    if (rungs_[j].dim == ds[d])
                    {
                        // a partial parent has fewer blocks than the maximum
                        const index_t cnt = ndiv(ee(ds[d]), rungs_[j].block);
                        if (t.idx[j] >= static_cast<std::size_t>(cnt)) { valid = false; break; }
                        ee(ds[d]) = segment(ee(ds[d]), rungs_[j].block, t.idx[j]);
                    }
            if (!valid) continue;      // the slot is not used by this parent
            const stride_t off = src_offset(op, t.idx);
            pack_[o]->pack(slot_of(op, t.idx), ee(panel_dim), ee(dimension::K),
                           base + off * static_cast<stride_t>(es), inc, ld);
            if (config_.profile)
            {
                pfstats_.pack_calls += 1;
                pfstats_.pack_elems += static_cast<std::size_t>(ee(panel_dim))
                                     * static_cast<std::size_t>(ee(dimension::K));
            }
        }
        if (config_.profile) pfstats_.pack_ns += ns_since(t0);
    }

    // ------------------------------------------------------------- walking ---

    void descend(int i, state s, const call& k, bool dry = false)
    {
        if (i == reg_first_)
        {
            drive(s, k, dry);
            return;
        }
        const loop_rung& r = rungs_[i];
        const index_t trips = (i < config_.top) ? 1 : ndiv(s.extent(r.dim), r.block);
        for (index_t t = 0; t < trips; t++)
        {
            state next = s;
            next.idx[i] = static_cast<std::size_t>(t);
            next.extent(r.dim) = segment(s.extent(r.dim), r.block, t);
            next.off_c = s.off_c + static_cast<stride_t>(t) * r.delta_c;
            if (r.dim == dimension::K && t != 0) next.first_k = false;
            for (std::size_t o = 0; o + 1 < operands.size(); o++)
                if (slot_touch_[o][i])
                    next.packed_slot(operands[o]) = slot_of(operands[o], next.idx);
            // above the region the packers were prepacked; inside it, pack here
            if (!dry && r.packs && i >= config_.top)
                for (std::size_t o = 0; o + 1 < operands.size(); o++)
                    if (r.packs_op(operands[o])) pack_all(operands[o], next, k);
            descend(i + 1, next, k, dry);
        }
        if (config_.profile && !dry) pfstats_.iteration[i] += static_cast<std::size_t>(trips);
    }

    // The register loops and the micro-kernel call: the leaf of the nest.
    void drive(state& s, const call& k, bool dry)
    {
        const kernel_info& ki = ukr_.info();
        const index_t n_jr = (idx_jr_ >= config_.top) ? ndiv(s.extent(dimension::N), ki.nr) : 1;
        const index_t n_ir = (idx_ir_ >= config_.top) ? ndiv(s.extent(dimension::M), ki.mr) : 1;
        if (dry)
        {
            calls_ += static_cast<double>(n_jr) * static_cast<double>(n_ir);
            return;
        }
        // beta multiplies C on the first k block of a tile, whichever loops cut k
        const double beta = s.first_k ? k.beta : 1.0;
        const double alpha = k.alpha;
        const stride_t ps_a = pack_[packed_index(operand::A)]->panel_stride();
        const stride_t ps_b = pack_[packed_index(operand::B)]->panel_stride();
        const char* a_blk = static_cast<const char*>(
            pack_[packed_index(operand::A)]->buffer(s.packed_slot(operand::A)));
        const char* b_blk = static_cast<const char*>(
            pack_[packed_index(operand::B)]->buffer(s.packed_slot(operand::B)));
        const stride_t ea = static_cast<stride_t>(elem_size(ki.dt_a));
        const stride_t eb = static_cast<stride_t>(elem_size(ki.dt_b));
        const stride_t ec = static_cast<stride_t>(elem_size(ki.dt_c));
        char* cblock = static_cast<char*>(k.c) + s.off_c * ec;
        if (config_.profile)
        {
            pfstats_.calls += static_cast<std::size_t>(n_jr) * static_cast<std::size_t>(n_ir);
            pfstats_.iteration[idx_jr_] += static_cast<std::size_t>(n_jr);
            pfstats_.iteration[idx_ir_] += static_cast<std::size_t>(n_ir);
        }
        kernel_call c;
        c.k = s.extent(dimension::K);
        c.alpha = &alpha;
        c.beta = &beta;
        c.rs_c = k.rs_c;
        c.cs_c = k.cs_c;
        if (jr_outer_)
        {
            for (index_t jr = 0; jr < n_jr; jr++)
            {
                const index_t n_r = segment(s.extent(dimension::N), ki.nr, jr);
                const char* b1 = b_blk + jr * ps_b * eb;
                for (index_t ir = 0; ir < n_ir; ir++)
                {
                    const index_t m_r = segment(s.extent(dimension::M), ki.mr, ir);
                    const char* a1 = a_blk + ir * ps_a * ea;
                    const bool last_ir = ir == n_ir - 1;
                    c.m = m_r;
                    c.n = n_r;
                    c.a = a1;
                    c.b = b1;
                    c.c = cblock + (ir * ki.mr * k.rs_c + jr * ki.nr * k.cs_c) * ec;
                    c.next_a = last_ir ? a_blk : a1 + ps_a * ea;
                    c.next_b = last_ir ? (jr == n_jr - 1 ? b_blk : b1 + ps_b * eb) : b1;
                    ukr_.gemm(c);
                }
            }
        }
        else
        {
            for (index_t ir = 0; ir < n_ir; ir++)
            {
                const index_t m_r = segment(s.extent(dimension::M), ki.mr, ir);
                const char* a1 = a_blk + ir * ps_a * ea;
                for (index_t jr = 0; jr < n_jr; jr++)
                {
                    const index_t n_r = segment(s.extent(dimension::N), ki.nr, jr);
                    const char* b1 = b_blk + jr * ps_b * eb;
                    const bool last_jr = jr == n_jr - 1;
                    c.m = m_r;
                    c.n = n_r;
                    c.a = a1;
                    c.b = b1;
                    c.c = cblock + (ir * ki.mr * k.rs_c + jr * ki.nr * k.cs_c) * ec;
                    c.next_b = last_jr ? b_blk : b1 + ps_b * eb;
                    c.next_a = last_jr ? (ir == n_ir - 1 ? a_blk : a1 + ps_a * ea) : a1;
                    ukr_.gemm(c);
                }
            }
        }
    }

    /// the region's extents and the call count, without running anything
    void measure()
    {
        // the region covers everything at and below --top, and one iteration
        // of each loop above it: so its extents are what those cuts leave
        state s = initial_state();
        for (int i = 0; i < config_.top && i < reg_first_; i++)
            s.extent(rungs_[i].dim) = segment(s.extent(rungs_[i].dim), rungs_[i].block, 0);
        exec_ = s.ext;
        calls_ = 0.0;
        call dummy;
        descend(0, initial_state(), dummy, true);
        pfstats_.iteration.assign(static_cast<std::size_t>(rung_count_), 0);
    }

    ukr_provider& ukr_;
    nest_spec spec_;
    problem psize_;
    nest_config config_;
    buffer_allocator& allocator_;
    std::vector<loop_rung> rungs_;
    int rung_count_ = 0, reg_first_ = 0, idx_jr_ = 0, idx_ir_ = 1;
    bool jr_outer_ = true, prepacked_ = false;
    const void* prepacked_a_ = nullptr;
    const void* prepacked_b_ = nullptr;
    std::vector<index_t> trips_;
    std::array<int, 2> pack_rung_ = {-1, -1};
    std::array<std::vector<int>, 2> slot_rungs_;
    std::array<std::array<bool, MAX_RUNGS>, 2> slot_touch_ = {};
    std::array<std::unique_ptr<packer>, 2> pack_;
    std::array<index_t, 3> exec_ = {0, 0, 0};
    double calls_ = 0.0;
    profile pfstats_;
};

} // namespace mmgen
