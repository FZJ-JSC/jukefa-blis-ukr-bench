// The one out-of-line piece of the mmgen library: --nest parsing, shared by
// every entry point (the bench, a preset table, a compatibility shim).

#include "mmgen/gemm.hpp"

namespace mmgen {

nest_spec parse_nest(const std::string& text, index_t mr, index_t nr,
                     const std::string& pack_at)
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
        r.dim = detail::dimension_from_string(tok.substr(0, colon));
        r.block = detail::block_of(tok.substr(colon + 1), mr, nr);
        spec.loops.push_back(r);
        group.push_back(static_cast<int>(spec.loops.size()) - 1);
    }
    if (!group.empty()) spec.groups.push_back(group);
    if (spec.loops.size() < 2)
        throw std::invalid_argument("--nest: give the loops, ending in m:mr and n:nr");
    if (spec.loops.size() > static_cast<std::size_t>(MAX_RUNGS))
        throw std::invalid_argument("--nest: too many loops");

    const loop_rung& a = spec.loops[spec.loops.size() - 2];
    const loop_rung& b = spec.loops[spec.loops.size() - 1];
    if (!((a.dim == dimension::M && a.block == mr && b.dim == dimension::N && b.block == nr)
          || (a.dim == dimension::N && a.block == nr && b.dim == dimension::M && b.block == mr)))
        throw std::invalid_argument("--nest: the nest must end in the register loops, m:" +
                                    std::to_string(mr) + " and n:" + std::to_string(nr) +
                                    " in either order");

    // labels: Goto's five keep their names, anything else is dimension and count
    std::vector<dimension> cache;
    for (std::size_t i = 0; i + 2 < spec.loops.size(); i++) cache.push_back(spec.loops[i].dim);
    std::vector<std::string> labels;
    if (cache == std::vector<dimension>{dimension::N, dimension::K, dimension::M})
        labels = {"jc", "pc", "ic"};
    else
    {
        std::array<int, 3> seen = {0, 0, 0};
        for (dimension d : cache)
            labels.push_back(std::string(1, dim_name(d)) + std::to_string(++seen[dim_index(d)]));
    }
    labels.push_back(a.dim == dimension::N ? "jr" : "ir");
    labels.push_back(b.dim == dimension::N ? "jr" : "ir");
    for (std::size_t i = 0; i < spec.loops.size(); i++) spec.loops[i].label = labels[i];

    for (const std::string& kv : detail::words(pack_at, ','))
    {
        const std::size_t c = kv.find(':');
        if (c == std::string::npos) throw std::invalid_argument("--pack-at: give A:LOOP or B:LOOP");
        std::string op = kv.substr(0, c), at = kv.substr(c + 1);
        for (char& ch : op) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        for (char& ch : at) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (op != "A" && op != "B") throw std::invalid_argument("--pack-at: only A and B are packed");
        const operand which = op == "A" ? operand::A : operand::B;
        int r = -1;
        if (at != "pre")
        {
            r = -2;
            for (std::size_t i = 0; i < labels.size(); i++) if (labels[i] == at) r = static_cast<int>(i);
            if (r < 0) throw std::invalid_argument("--pack-at: no loop " + at);
            if (spec.loops[r].is_register())
                throw std::invalid_argument("--pack-at: " + at + " is a register loop");
        }
        spec.pack_at[packed_index(which)] = r;
    }
    for (std::size_t o = 0; o + 1 < operands.size(); o++)
    {
        const operand which = operands[o];
        if (spec.pack_at[o] != -2) continue;
        int r = -1;    // 'pre' if no loop below the top keeps it invariant
        for (std::size_t i = 1; i < labels.size(); i++)
            if (invariant_of(spec.loops[i].dim) == which)
            {
                r = static_cast<int>(i) - 1;
                break;
            }
        // a register loop is inside the driver: packing there is 'pre'
        if (r >= 0 && spec.loops[r].is_register()) r = -1;
        spec.pack_at[o] = r;
    }
    for (std::size_t o = 0; o + 1 < operands.size(); o++)
    {
        const int r = spec.pack_at[o];
        if (r >= 0) spec.loops[r].packs |= 1 << o;
    }
    return spec;
}

} // namespace mmgen
