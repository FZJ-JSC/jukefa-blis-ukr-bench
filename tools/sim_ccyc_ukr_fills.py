#!/usr/bin/env python3
"""
Fills and residency of a C-cycling SME micro-kernel: an LRU simulation.

`model.py` cannot express this structure (it has one C invariant, the `pc`
rung sized m*n_c, and no rung below the register loops), so the L1 behaviour
of a micro-kernel that cycles ZA tiles through a staged C is simulated here
instead: a line-level LRU cache driven by the loop structure, which yields
the fills/cycle, the L1 and L2 bandwidth per cycle, and the number of misses
in flight a given L2 latency would need.

The kernel this models
----------------------
Three shapes, and the names matter because two of them coincide on SME:

  nano-kernel   one ZA array of C: m_za x n_za doubles (16x32 for fp64 at
                SVL=512, i.e. 4 KiB), accumulated over k_mu k-steps
  logical tile  rows x cols nano-kernels (--shape M_T N_T), the tile one
                micro-kernel call computes when it is bigger than ZA -- which
                is what forces the C cycling
  blocking      mc, nc, kc: the packed panels' depth and the cache blocks

m_za x n_za IS BLIS's MR x NR whenever the micro-kernel is exactly one ZA
array, because MR/NR mean the register-resident tile and ZA is where an SME
accumulator lives; the logical tile is then the caller's tile and has no BLIS
name of its own.  If a port instead defines MR/NR as a whole number of ZA
tiles, then --shape plays the role of MR/NR and the packdim should follow the
PACKED width (MR/NR), not the nano width -- set --packdim-a/-b explicitly,
since they default to m_za/n_za.

One call computes a logical tile of ``rows x cols`` ZA tiles.  Inside the
call a k-chunk loop of depth ``k_mu`` interleaves them::

    for each k-chunk of depth k_mu:
      for each ZA tile (r, c) of the logical tile:
        ldr za  <- C tile (r, c)              # m_za*n_za*esize bytes
        for k in chunk:
          fmopa into ZA from A strip(r) and B strip(c)
        str za  -> C tile (r, c)

What is modelled
----------------
  * one cache (size, line, associativity) in front of the k-loop, LRU
    replacement, line granularity;
  * a line of A holds ``line/esize`` consecutive k of one row (row-major with
    stride ``lda_a``), a line of B holds ``line/esize`` consecutive k of one
    column (stride ``ldb_b``), a line of C holds ``line/esize`` doubles of one
    ZA tile -- so a line is touched once per *octet* of k, not once per
    k-step.  This is what turns m*k_mu elements into m*k_mu*esize/line lines;
  * the accumulator is loaded at the start of its pass and stored at the end
    (``--touch both``, the default) -- this dominates every other ordering
    choice, see --touch;
  * successive chunks with k advancing, so A/B lines are new data each chunk
    while the C lines are the same ones re-touched.  Reported figures are the
    steady-state mean after ``--warmup`` chunks;
  * a fill is any touch of a non-resident line, loads and stores alike; with
    --count-wb a dirty line's eviction is counted as a write-back too.

What is NOT modelled
--------------------
  * the hardware prefetcher, which in a real kernel would pull A/B ahead and
    can displace the accumulator -- the effect --touch is fighting;
  * any translation of the ZA load/store into slices (its cost is the C
    bandwidth term, not the line count);
  * partial tiles (blocks are assumed to be whole multiples of the tiles);
  * latency, MSHR occupancy and time: only the miss *rate* is produced, then
    converted to in-flight misses = rate x latency.  The model's fill-buffer
    test is richer than that product; treat it as first order.

Usage
-----
    python tools/sim_ccyc_ukr_fills.py --shape 32 64 --k-mu 16
    python tools/sim_ccyc_ukr_fills.py --shape 32 32 --k-mu 8,16,32,48
    python tools/sim_ccyc_ukr_fills.py --shape 32 64 --k-mu 16 --touch end
    python tools/sim_ccyc_ukr_fills.py --shape 32 64 --k-mu 16 --assoc 8
    python tools/sim_ccyc_ukr_fills.py --sweep-shapes --k-mu 16
    python tools/sim_ccyc_ukr_fills.py --selftest

Columns: kb is the live set (A/B strips plus the accumulator), comp is the
compulsory rate for that configuration, fills/cy is what the simulation
measures (1 = every line touch misses), and ms/max is the in-flight misses at
each --l2-latency as a percentage of --mshr entries.
"""
from __future__ import annotations
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import OrderedDict
from dataclasses import dataclass


# ---------------------------------------------------------------- config ---

@dataclass
class cfg:
    """
    One simulated configuration.

    :param m_za: rows of one ZA tile (16 for fp64 at SVL=512)
    :param n_za: columns of one ZA tile (32)
    :param rows: ZA tiles stacked along m in the logical tile
    :param cols: ZA tiles stacked along n
    :param k_mu: k-steps per ZA pass
    :param esize: element size in bytes
    :param line: cache line size in bytes
    :param size: cache capacity in bytes
    :param assoc: ways per set; 0 or the whole cache means fully associative
    :param touch: when the accumulator is touched: 'both' (load at the start
                  of the pass and store at the end), 'start', 'end' or 'none'
                  (an accumulator that persists across chunks)
    :param ab_order: 'k' interleaves A and B line by line within a k-step,
                     'strip' reads the whole A strip then the whole B strip
    :param pass_order: 'row' visits the tiles tile-major in m, 'col' in n
    :param kc: k-steps in the packed A/B block the chunks walk along; the
               default covers every simulated chunk, so a chunk's A/B lines
               are always new data
    :param layout: 'packed' (the physical one: A stored as m_r-wide panels of
                   kc x packdim_a doubles, B as n_r-wide panels of kc x
                   packdim_b) or 'row' (row-major A, column-major B, k
                   contiguous -- the shape the first version of this
                   simulation used)
    :param packdim_a: stored rows of one packed A panel, in elements.  It
                      follows MR, and in a C-cycling micro-kernel MR is the
                      LOGICAL tile, not one ZA array -- the accumulator is only
                      one ZA array at a time, so nothing smaller is resident.
                      Default: m_t.  Pass m_za to model a non-cycling kernel.
    :param packdim_b: stored columns of one packed B panel; default n_t
    :param panel_pad: bytes added to each packed panel's stride, which breaks
                      panel aliasing WITHOUT disturbing the k stride inside a
                      panel (padding packdim does both)
    :param lda_a: row stride in bytes, 'row' layout only (default kc*esize)
    :param ldb_b: column stride in bytes, 'row' layout only (default kc*esize)
    :param chunks: chunks simulated
    :param warmup: chunks discarded before averaging
    :param dc_zva: the first accumulation of a packed block needs no prior AB
                   (it starts at zero), so its accumulator load can be skipped;
                   a dc zva on those lines also avoids a read-for-ownership if
                   the staging store would otherwise fetch them
    :param epilogue: model the end-of-kc-block epilogue: per ZA tile, read the
                     staged AB (4 KiB, an L1 hit), read C (4 KiB, a miss
                     stream -- RAM or L2 if prefetched) and write C back.  The
                     AB staging traffic in the k-loop is what gives the C load
                     something to hide behind; the C read and write are new
                     traffic that only happens once per kc block.
    :param count_wb: count dirty evictions as write-backs
    :param peak: FLOP/cycle in the k-loop (128 = 1 FMOPA/cy at 16x32 fp64)
    :param l1_bw: L1 bytes/cycle for LOADS (the read-only port); stores use the
                  second port, which the loads can also use when it is free,
                  so the combined traffic is also capped by l1_bw
    :param l1_st_bw: L1 bytes/cycle for STORES (the shared port alone)
    :param mshr: L1 fill-buffer entries, for the in-flight column
    :param latency: L2 latencies in cycles to report in-flight misses for
    """
    m_za: int = 16
    n_za: int = 32
    rows: int = 2
    cols: int = 1
    k_mu: int = 32
    esize: int = 8
    line: int = 64
    size: int = 32 * 1024
    assoc: int = 0
    touch: str = 'both'
    ab_order: str = 'k'
    pass_order: str = 'row'
    kc: int = 0
    layout: str = 'packed'
    packdim_a: int = 0
    packdim_b: int = 0
    panel_pad: int = 0
    lda_a: int = 0
    ldb_b: int = 0
    chunks: int = 60
    warmup: int = 10
    dc_zva: bool = False
    epilogue: bool = False
    count_wb: bool = False
    peak: float = 128.0
    l1_bw: float = 128.0
    l1_st_bw: float = 64.0
    mshr: int = 16
    latency: tuple = (11, 16, 21, 25)

    # ---- derived quantities -------------------------------------------

    @property
    def t(self) -> int:
        """:return: ZA tiles per logical tile"""
        return self.rows * self.cols

    @property
    def m_t(self) -> int:
        """:return: rows of the logical tile"""
        return self.rows * self.m_za

    @property
    def n_t(self) -> int:
        """:return: columns of the logical tile"""
        return self.cols * self.n_za

    @property
    def per_line(self) -> int:
        """:return: elements of k in one line"""
        return self.line // self.esize

    @property
    def cyc_per_kstep(self) -> float:
        """:return: cycles one k-step of one ZA tile takes at --peak"""
        return self.m_za * self.n_za * 2.0 / self.peak

    @property
    def cyc_per_chunk(self) -> float:
        """:return: cycles one k-chunk takes"""
        return self.t * self.k_mu * self.cyc_per_kstep

    @property
    def tile_bytes(self) -> int:
        """:return: bytes of one C accumulator tile"""
        return self.m_za * self.n_za * self.esize

    def tile_order(self) -> list:
        """:return: the (r, c) ZA tiles in the order the passes visit them"""
        if self.pass_order == 'row':
            return [(r, c) for r in range(self.rows) for c in range(self.cols)]
        return [(r, c) for c in range(self.cols) for r in range(self.rows)]


# ----------------------------------------------------------------- cache ---

class cache:
    """
    Set-associative LRU cache of lines, with optional dirty tracking.

    :param size: capacity in bytes
    :param line: line size in bytes
    :param assoc: ways per set; 0 or size//line means fully associative
    """

    def __init__(self, size : int, line : int, assoc : int = 0):
        self.line = line
        self.nline = max(1, size // line)
        if assoc <= 0 or assoc >= self.nline:      # one set of all the lines
            self.ways, self.nsets = self.nline, 1
        else:
            self.nsets = max(1, self.nline // assoc)
            self.ways = self.nline // self.nsets
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.seen = set()          # distinct lines ever touched
        self.used = set()          # sets ever touched
        self.fills = 0
        self.wb = 0
        self.by = {}

    def access(self, addr : int, dirty : bool = False, tag : str = '') -> bool:
        """
        Touch one line.

        :param addr: byte address inside the line
        :param dirty: whether this access leaves the line dirty
        :return: True if the line was not resident (a fill)
        """
        n = addr // self.line
        s = self.sets[n % self.nsets]
        key = n // self.nsets
        if key in s:
            s.move_to_end(key)
            s[key] = s[key] or dirty
            return False
        self.fills += 1
        self.by[tag] = self.by.get(tag, 0) + 1
        s[key] = dirty
        self.used.add(n % self.nsets)
        self.seen.add(n)
        while len(s) > self.ways:
            _, was = s.popitem(last=False)
            if was:
                self.wb += 1
        return True

    @property
    def refills(self) -> int:
        """:return: fills beyond one per distinct line (i.e. lines fetched twice)"""
        return max(0, self.fills - len(self.seen))

    def clear_counts(self):
        """Reset the counters, keeping residency (for a warm start)."""
        self.fills = self.wb = 0


# ---------------------------------------------------------------- stream ---

A_BASE, B_BASE, C_BASE = 0, 1 << 24, 2 << 24


def addr_a(c : cfg, row : int, k : int) -> int:
    """
    :param c: configuration
    :param row: row of the logical tile, within its own micro-panel
    :param k: k-step within the packed block
    :return: byte address of the line holding that element
    """
    if c.layout == 'row':
        lda = c.lda_a or (c.kc or c.chunks * c.k_mu) * c.esize
        return A_BASE + row * lda + (k // c.per_line) * c.line
    pd = c.packdim_a or c.m_t
    panel = row // c.m_za
    kc = c.kc or c.chunks * c.k_mu
    return (A_BASE + panel * (kc * pd * c.esize + c.panel_pad)
            + (k * pd + row % c.m_za) * c.esize)


def addr_b(c : cfg, col : int, k : int) -> int:
    """
    :param c: configuration
    :param col: column of the logical tile, within its own micro-panel
    :param k: k-step within the packed block
    :return: byte address of the line holding that element
    """
    if c.layout == 'row':
        ldb = c.ldb_b or (c.kc or c.chunks * c.k_mu) * c.esize
        return B_BASE + col * ldb + (k // c.per_line) * c.line
    pd = c.packdim_b or c.n_t
    panel = col // c.n_za
    kc = c.kc or c.chunks * c.k_mu
    return (B_BASE + panel * (kc * pd * c.esize + c.panel_pad)
            + (k * pd + col % c.n_za) * c.esize)


def touches(c : cfg, chunk : int):
    """
    Yield (address, dirty) for every line touched in one k-chunk, in program
    order.

    :param c: configuration
    :param chunk: which chunk, for the k range
    :return: generator of (byte address, dirty)
    """
    k0 = chunk * c.k_mu
    c_lines = c.tile_bytes // c.line
    kc = c.kc or c.chunks * c.k_mu
    first = c.dc_zva and (chunk * c.k_mu) % kc == 0
    rows = range(0, c.m_t) if c.ab_order == 'strip' else None
    for (r, col) in c.tile_order():
        tile = r * c.cols + col
        cbase = C_BASE + tile * c.tile_bytes
        if c.touch in ('start', 'both') and not first:
            for l in range(c_lines):
                yield cbase + l * c.line, False, 'C'
        if c.ab_order == 'k':
            for k in range(k0, k0 + c.k_mu):
                for row in range(r * c.m_za, (r + 1) * c.m_za):
                    yield addr_a(c, row, k), False, 'AB'
                for cc in range(col * c.n_za, (col + 1) * c.n_za):
                    yield addr_b(c, cc, k), False, 'AB'
        else:                                   # whole A strip, then whole B
            for row in range(r * c.m_za, (r + 1) * c.m_za):
                for k in range(k0, k0 + c.k_mu):
                    yield addr_a(c, row, k), False, 'AB'
            for cc in range(col * c.n_za, (col + 1) * c.n_za):
                for k in range(k0, k0 + c.k_mu):
                    yield addr_b(c, cc, k), False, 'AB'
        if c.touch in ('end', 'both'):
            for l in range(c_lines):
                yield cbase + l * c.line, True, 'C'


# ------------------------------------------------------------------- run ---

def run(c : cfg) -> dict:
    """
    Simulate one configuration.

    :param c: configuration
    :return: dict of measurements, per cycle where that makes sense
    """
    l1 = cache(c.size, c.line, c.assoc)
    per, wb, re, ab = [], [], [], []
    for chunk in range(c.chunks):
        l1.fills = l1.wb = 0
        l1.by = {}
        for addr, dirty, tag in touches(c, chunk):
            l1.access(addr, dirty and c.count_wb, tag)
        per.append(l1.fills)
        re.append(l1.by.get('C', 0))                   # fills paid by the C stream
        ab.append(l1.by.get('AB', 0))                  # ... and by the A/B strips
        wb.append(l1.wb if c.count_wb else 0)
    steady = per[c.warmup:]
    fills = sum(steady) / max(1, len(steady))
    # what each stream costs per chunk: 0 C fills means the accumulator survived
    rs = re[c.warmup:]
    as_ = ab[c.warmup:]
    c_fills = sum(rs) / max(1, len(rs))
    ab_fills = sum(as_) / max(1, len(as_))
    ab_unique = (c.m_t + c.n_t) * c.k_mu * c.esize / c.line
    wbs = sum(wb[c.warmup:]) / max(1, len(wb[c.warmup:]))

    cyc = c.cyc_per_chunk
    ab_b = c.t * c.k_mu * (c.m_za + c.n_za) * c.esize
    c_moves = 2 if c.touch == 'both' else 1 if c.touch in ('start', 'end') else 0
    c_b = c.t * c.tile_bytes * c_moves
    c_rd = c_b / 2 if c_moves == 2 else c_b
    c_wr = c_b - c_rd
    if c.dc_zva and c.touch != 'none':
        # one accumulator load per packed block is skipped, so the steady-state
        # per-chunk read falls by 1/nchunks of the staged read
        nch = max(1.0, (c.kc or c.chunks * c.k_mu) / c.k_mu)
        c_rd -= (c.t * c.tile_bytes / nch)
    ab_bytes = (c.m_t + c.n_t) * c.k_mu * c.esize       # strips, counted once
    live = ab_bytes + (0 if c.touch == 'none' else c.t * c.tile_bytes)
    comp = (c.m_t + c.n_t) * c.k_mu * c.esize / c.line      # A/B first touch
    comp += 0 if c.touch == 'none' else c.t * c.tile_bytes / c.line

    rd_bw = (ab_b + c_rd) / cyc
    wr_bw = c_wr / cyc
    epi_rd = epi_wr = epi_fills = 0.0
    if c.epilogue and c.touch != 'none':
        kc = c.kc or c.chunks * c.k_mu
        per_kc = kc / c.k_mu                    # chunks per packed block
        # per logical tile per block: the LAST tile's AB is still in ZA and a
        # MOVA to Z costs no memory, so only (t-1) AB tiles are read from L1;
        # every tile reads 4 KiB of C and writes 4 KiB back
        epi_rd = (2 * c.t - 1) * c.tile_bytes / (per_kc * cyc)
        epi_wr = c.t * c.tile_bytes / (per_kc * cyc)
        epi_fills = c.t * (c.tile_bytes / c.line) / per_kc
        rd_bw += epi_rd
        wr_bw += epi_wr
    return dict(fills=fills, fills_cy=fills / cyc, l2_bw=fills / cyc * c.line,
                rd_bw=rd_bw, wr_bw=wr_bw, epi_rd=epi_rd, epi_wr=epi_wr,
                epi_fills=epi_fills,
                c_fills=c_fills, ab_fills=ab_fills, ab_unique=ab_unique,
                sets_used=len(l1.used), nsets=l1.nsets,
                l1_ab_bw=ab_b / cyc, l1_bw=(ab_b + c_b) / cyc, wb=wbs,
                live_kb=live / 1024, comp_cy=comp / cyc,
                inflow=[fills / cyc * L for L in c.latency],
                util=[100 * fills / cyc * L / c.mshr for L in c.latency])


def fmt(c : cfg, m : dict) -> str:
    """
    :param c: configuration
    :param m: run() output
    :return: one table row
    """
    shape = f"{c.m_t}x{c.n_t}"
    infl = " ".join(f"{x:4.1f}" for x in m['inflow'])
    util = " ".join(f"{x:4.0f}" for x in m['util'])
    flag = []
    if m['rd_bw'] > c.l1_bw:
        flag.append("load port over")
    if m['wr_bw'] > c.l1_st_bw:
        flag.append("store port over")
    if m['rd_bw'] + m['wr_bw'] > c.l1_bw:
        flag.append("ports over")
    if m['live_kb'] * 1024 > c.size:
        flag.append("live > cache")
    if m['fills_cy'] > m['comp_cy'] * 1.02:
        flag.append("above compulsory")
    sets = (f"{m['sets_used']:>3}/{m['nsets']:<3} "
            f"{m['ab_fills']:>5.0f}/{m['ab_unique']:>4.0f} {m['c_fills']:>4.0f}")
    if m['c_fills'] > 0.0 and c.touch != 'none':
        flag.append("C re-fetched")
    if m['ab_fills'] > m['ab_unique'] * 1.02:
        flag.append("A/B re-fetched")
    return (f"{shape:>7} {c.t:>2} {c.k_mu:>5} {m['live_kb']:>6.0f}K {m['comp_cy']:>7.3f} "
            f"{m['fills_cy']:>7.3f} {m['l2_bw']:>6.1f} "
            f"{m['rd_bw']:>5.0f}/{m['wr_bw']:<4.0f} {sets} "
            f"{infl} | {util} {' '.join(flag)}")


def header(c : cfg) -> str:
    lat = " ".join(f"{L:>4}" for L in c.latency)
    return (f"{'shape':>7} {'t':>2} {'k_mu':>5} {'live':>7} {'comp':>7} "
            f"{'fills':>7} {'L2 B/cy':>6} {'L1 rd/wr':>9} {'sets  ABf/uniq  Cf':>18} "
            f"{'in-flight @ ' + lat:>21} | {'% of ' + str(c.mshr) + ' MSHRs':>17}")


# -------------------------------------------------------------- selftest ---

def selftest() -> int:
    """
    Check the numbers this simulation has published.

    :return: exit status
    """
    bad = 0

    def chk(label, got, want, tol=1e-3):
        nonlocal bad
        ok = abs(got - want) <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got:.3f} (want {want:.3f})")
        bad += 0 if ok else 1

    # the 32x64/k_mu=32 point is over L1 capacity, so its exact figure moves
    # with the layout (0.844 row-major, 0.871 packed); only the ordering holds
    for (mr, nr, k, want) in ((32, 64, 16, 0.375), (32, 64, 32, 0.871),
                              (32, 32, 32, 0.500), (16, 32, 32, 0.750),
                              (16, 64, 16, 0.625)):
        c = cfg(rows=mr // 16, cols=nr // 32, k_mu=k)
        chk(f"{mr}x{nr} k_mu={k} fills/cy", run(c)['fills_cy'], want)
    c = cfg(rows=2, cols=2, k_mu=16, touch='end')
    chk("32x64 k_mu=16 with a single C touch", run(c)['fills_cy'], 0.750)
    c = cfg(rows=2, cols=2, k_mu=16, touch='none')
    chk("32x64 k_mu=16 with a persistent accumulator", run(c)['fills_cy'], 0.375)
    c = cfg(rows=2, cols=1, k_mu=32, pass_order='col')
    chk("32x32 k_mu=32 tile-major in n", run(c)['fills_cy'], 0.500)
    # assoc: an 8-way 32 KiB L1 changes the answer, and that is a finding, not
    # noise -- only the low-occupancy points keep their compulsory rate
    def chk_range(label, got, lo, hi):
        nonlocal bad
        ok = lo <= got <= hi
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got:.3f} "
              f"(want {lo:.3f}..{hi:.3f})")
        bad += 0 if ok else 1

    # associativity: packed panels keep the compulsory rates, but only for a
    # friendly panel stride -- that is why kc and packdim have to be swept
    chk_range("32x32 k_mu=16, 8-way, packed (robust)",
              run(cfg(rows=2, cols=1, k_mu=16, assoc=8))['fills_cy'], 0.49, 0.51)
    # in a C-cycling micro-kernel MR/NR is the LOGICAL tile, so packdim follows
    # --shape and the strips skip sets: 0.625, not the 0.375 of a nano packdim
    chk_range("32x64 k_mu=16, 8-way, packdim = MR (32/64), kc=192",
              run(cfg(rows=2, cols=2, k_mu=16, assoc=8, kc=192))['fills_cy'], 0.60, 0.65)
    chk_range("32x64 k_mu=16, 8-way, packdim = nano (16/32), kc=192",
              run(cfg(rows=2, cols=2, k_mu=16, assoc=8, kc=192,
                      packdim_a=16, packdim_b=32))['fills_cy'], 0.37, 0.38)
    chk_range("32x64 k_mu=16, 8-way, row layout, stride 1024 (thrash)",
              run(cfg(rows=2, cols=2, k_mu=16, assoc=8, layout='row',
                      lda_a=1024, ldb_b=1024))['fills_cy'], 5.0, 7.0)
    print(f"selftest: {'PASS' if not bad else str(bad) + ' FAILED'}")
    return 0 if not bad else 1


# ------------------------------------------------------------------ main ---

def parse_kmu(text : str) -> list:
    """
    :param text: '16', '8,16,32' or '8:48:8'
    :return: list of k_mu values
    """
    if ',' in text:
        return [int(x) for x in text.split(',')]
    if ':' in text:
        a, b, s = (int(x) for x in text.split(':'))
        return list(range(a, b + 1, s))
    return [int(text)]


def main(argv : list[str] | None = None) -> int:
    """
    Run the simulation from the command line.

    :param argv: arguments without the program name; None reads sys.argv
    :return: exit status
    """
    ap = ArgumentParser(prog='sim_ccyc_ukr_fills.py',
                        formatter_class=RawDescriptionHelpFormatter,
                        description=(__doc__ or '').split('Usage')[0].strip())
    ap.add_argument('--shape', nargs=2, type=int, metavar=('M_T', 'N_T'),
                    help="logical tile in elements (32 64); ignored with --sweep-shapes")
    ap.add_argument('--m-za', type=int, default=16, help="rows of one ZA tile")
    ap.add_argument('--n-za', type=int, default=32, help="columns of one ZA tile")
    ap.add_argument('--esize', type=int, default=8, help="element size in bytes")
    ap.add_argument('--k-mu', default='32', metavar='N|A,B|A:B:S',
                    help="k-steps per ZA pass, or a list/sweep")
    ap.add_argument('--line', type=int, default=64, help="cache line in bytes")
    ap.add_argument('--size', default='32K', metavar='BYTES',
                    help="cache capacity, e.g. 32K, 48K, 32768")
    ap.add_argument('--assoc', type=int, default=0,
                    help="ways per set; 0 = fully associative")
    ap.add_argument('--touch', choices=('both', 'start', 'end', 'none'), default='both',
                    help="when the accumulator is touched per pass (default both: "
                         "load at the start, store at the end)")
    ap.add_argument('--ab-order', choices=('k', 'strip'), default='k',
                    help="interleave A and B per k-step, or strip by strip")
    ap.add_argument('--pass-order', choices=('row', 'col'), default='row',
                    help="visit the ZA tiles tile-major in m or in n")
    ap.add_argument('--kc', type=int, default=0,
                    help="k-steps in the packed A/B block (default: every chunk)")
    ap.add_argument('--layout', choices=('packed', 'row'), default='packed',
                    help="packed micro-panels (physical) or row-major/column-major")
    ap.add_argument('--packdim-a', type=int, default=0,
                    help="stored rows of a packed A panel, elements (default: the "
                         "logical tile, since MR follows the micro-kernel; pass "
                         "m_za to model a non-cycling kernel)")
    ap.add_argument('--packdim-b', type=int, default=0,
                    help="stored columns of a packed B panel (default: the logical tile)")
    ap.add_argument('--panel-pad', type=int, default=0,
                    help="extra bytes per packed panel stride: breaks panel "
                         "aliasing without changing the k stride")
    ap.add_argument('--lda', type=int, default=0, help="A row stride in bytes (--layout row)")
    ap.add_argument('--ldb', type=int, default=0, help="B column stride in bytes")
    ap.add_argument('--chunks', type=int, default=60, help="chunks simulated")
    ap.add_argument('--warmup', type=int, default=10, help="chunks discarded")
    ap.add_argument('--dc-zva', action='store_true',
                    help="skip the accumulator load on the first chunk of a "
                         "packed block: the first accumulation starts at zero "
                         "(and a dc zva avoids a read-for-ownership on it)")
    ap.add_argument('--epilogue', action='store_true',
                    help="add the end-of-kc-block epilogue: AB read (L1) + C read "
                         "and write (miss stream), per ZA tile, once per kc block")
    ap.add_argument('--count-wb', action='store_true',
                    help="also count dirty evictions (write-backs)")
    ap.add_argument('--peak', type=float, default=128.0, help="FLOP/cycle in the k-loop")
    ap.add_argument('--l1-bw', type=float, default=128.0,
                    help="L1 bytes/cycle for loads, and for loads+stores combined")
    ap.add_argument('--l1-store-bw', type=float, default=64.0,
                    help="L1 bytes/cycle for stores (the shared port alone)")
    ap.add_argument('--mshr', type=int, default=16, help="L1 fill-buffer entries")
    ap.add_argument('--l2-latency', default='11,16,21,25',
                    help="L2 latencies in cycles for the in-flight columns")
    ap.add_argument('--sweep-shapes', action='store_true',
                    help="run 16x32, 16x64, 32x32, 32x64")
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    size = a.size.strip().upper()
    mult = {'K': 1024, 'M': 1 << 20}.get(size[-1:], 1)
    size = int(size[:-1]) * mult if mult > 1 else int(size)
    lat = tuple(int(x) for x in a.l2_latency.split(','))
    base = dict(m_za=a.m_za, n_za=a.n_za, esize=a.esize, line=a.line, size=size,
                assoc=a.assoc, touch=a.touch, ab_order=a.ab_order,
                pass_order=a.pass_order, kc=a.kc, layout=a.layout,
                packdim_a=a.packdim_a, packdim_b=a.packdim_b, panel_pad=a.panel_pad,
                lda_a=a.lda, ldb_b=a.ldb,
                chunks=a.chunks, warmup=a.warmup, dc_zva=a.dc_zva,
                epilogue=a.epilogue,
                count_wb=a.count_wb,
                peak=a.peak, l1_bw=a.l1_bw, l1_st_bw=a.l1_store_bw,
                mshr=a.mshr, latency=lat)
    shapes = [(16, 32), (16, 64), (32, 32), (32, 64)] if a.sweep_shapes else [tuple(a.shape)]
    if any(s is None for s in shapes[0]):
        ap.error("give --shape M_T N_T, or --sweep-shapes")

    print(header(cfg(**base)))
    for (m_t, n_t) in shapes:
        for k_mu in parse_kmu(str(a.k_mu)):
            rows, cols = m_t // a.m_za, n_t // a.n_za
            if rows * a.m_za != m_t or cols * a.n_za != n_t:
                ap.error(f"--shape {m_t} {n_t} is not a multiple of the ZA tile "
                         f"{a.m_za}x{a.n_za}")
            c = cfg(rows=rows, cols=cols, k_mu=k_mu, **base)
            print(fmt(c, run(c)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
