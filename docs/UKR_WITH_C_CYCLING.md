SME micro-kernel with C cycling
===============================

Design notes for a BLIS micro-kernel on an SME core (SVL = 512 bits, fp64) in
which the ZA array is cycled through several tiles inside one micro-kernel
call, with C staged in memory and a k-chunk depth as the tuning knob.  Nothing
here is implemented: the numbers are traffic estimates, a residency
simulation, and predictions from `model.py` where it can express the
structure at all.

Status: `model.py` on this branch CANNOT express the design.  `--nest`
requires the innermost loops to be the register loops:

    model.py: error: --nest: the nest must end in the register loops,
             m:16 and n:32 in either order (m:mr n:nr)

so there is exactly one C invariant in the model (the `pc` rung, sized
`M x n_c`) and no way to put a k rung below the register loops whose
invariant is a ZA-sized C tile.  See "Model gaps" at the end.

Contents
--------
  1. Hardware facts the estimates rest on
  2. Where the time goes in the current kernel
  3. The proposal
  4. Requirements at 1 FMOPA/cycle: L1 bandwidth, fills, MLP, L2, DRAM
  5. Residency: what actually stays in L1 (simulation)
  6. Recommended operating point
  7. Closed forms for the standard counters (used above, validated)
  8. Caveats
  9. Model gaps and the knobs that would close them
 10. Open measurements

Numbers are tagged: [meas] measured on the target, [model] `model.py`
output, [calc] hand arithmetic from the two, [sim] the LRU simulation in
section 5.


1. Hardware facts the estimates rest on
---------------------------------------

  [meas] 1 FMOPA/cycle, fp64, SVL=512  ->  128 FLOP/cy
         an FMOPA at fp64 outer-products SVL/64 = 8 x 8 doubles = 64 MACs
  [meas] epilogue 2 fmla/cycle          ->   32 FLOP/cy
  [meas] 8 cycles per k-step in the inner loop
  [meas] 300-500 cycles per micro-kernel call
  [meas] 48 B/cycle of L1D read bandwidth in the inner loop
         L1D believed to deliver ~128 B/cycle (unverified)

One ZA array at SVL=512 holds 4 KiB of accumulators = 512 doubles, i.e.
16 x 32 fp64.  That is simultaneously the largest tile the array can hold and
the tile in today's kernel: **there is no larger ZA-resident tile to move to**.
Any bigger logical tile has to cycle ZA contents through memory.

Why 48 B/cy is the requirement and not headroom [calc]:

    one k-step of a 16x32 tile = 512 MACs = 8 FMOPAs = 1024 FLOP
    at 1 FMOPA/cy that is 8 cycles
    operands: 16 A doubles (128 B) + 32 B doubles (256 B) = 384 B
    384 B / 8 cycles = 48 B/cycle

and 8 FMOPAs + 6 vector loads = 14 instructions per 8 cycles = 1.75 IPC, with
the FMOPA port fully occupied.  So the inner loop is at peak FLOP rate, at
the bandwidth peak requires, and at 87 % of a 2-IPC issue budget.  The spare
~80 B/cy cannot be converted into FLOPs; its only use is hiding work that is
not in the k-loop.


2. Where the time goes in the current kernel
--------------------------------------------

At N=1024, kc=208, mc=192, nc=2048, 16x32 tile [calc]:

    calls              = (1024/16)(1024/32)(1024/208) = 10 240
    k_mean per call    = 1024/5 = 204.8 steps, 8 cy each = 1638 cy
    k-loop             = 10 240 x 1638 cy             = 16.8 Mcy
    ideal (2.2 GFLOP / 128 FLOP/cy)                   = 17.2 Mcy
    per-call overhead 300-500 cy                      =  3.1-5.1 Mcy (18-30 %)

The model independently charges the same thing [model]:

    C epilogue: 64.0 lines/call from beyond L1, 4 wave(s) x 111 cy
                - lookahead 0 = 444 cy exposed per call

444 cy/call against a 300-500 cy measurement means the model's epilogue term
is calibrated; it is not an artefact.  What the model cannot say is that ZA
accumulates across kc blocks: it charges one epilogue per packed k-block,
whereas an SME kernel can keep the accumulator live for the whole k extent.

The model's OTHER limiter at this blocking is wrong for this part [model]:
with the spec's 16 L1 MSHRs / 5 cy L1 / 21 cy L2 it needs ~17 outstanding
fills to sustain peak, so it predicts 61 % of peak limited by "L1 fill
buffers".  The measured 8 cy/k says fills are being covered.  Raising the
MSHR count alone moves the prediction to 78 % and the limiter to "compute +
exposed C fetch".  Treat every "the panels do not fit L1" conclusion in this
document as conditional on the real MSHR/latency numbers.


3. The proposal
---------------

One micro-kernel call computes a logical tile of t ZA tiles (e.g. 2 x 2 =
32x64).  Inside the call, a k-chunk loop interleaves the tiles so that A and
B strips are shared across ZA passes:

    for k-chunk of depth k_mu in 0..k_c:
      for each ZA tile (r, c) of the logical tile:
        ldr za  <- C buffer[(r,c)]        # 4 KiB
        for k in chunk:
          fmopa into ZA using A strip(r), B strip(c)
        str za  -> C buffer[(r,c)]

  t        number of ZA tiles in the logical tile (1, 2, 4, ...)
  k_mu     k-steps per ZA pass: the knob that trades C traffic against the
           L1 footprint of the A/B strips
  tile     how the t tiles are arranged.  Splitting in m matters:
           32x32 is better than 16x64, see section 5.

What it buys: A is reused across column groups and B across row groups
without either being resident at full panel depth, so the strips come from
L1 instead of L2.

What it costs: every ZA pass moves its 4 KiB accumulator out and back, i.e.
2 x 4 KiB per tile per chunk, and the C buffer (t x 4 KiB) competes with the
strips for L1.


4. Requirements at 1 FMOPA/cycle
--------------------------------

Per cycle, at 1 FMOPA/cy = 64 MAC/cy, with one k-step of a ZA tile = 8 cycles
and 384 B of A/B [calc]:

    A/B loads      = 48 B/cy                     fixed by the ZA tile
    C loads+stores = 1024/k_mu B/cy              one 4 KiB accumulator
                                                 out and back, per tile per
                                                 chunk, independent of t
    L1 total       = 48 + 1024/k_mu B/cy

A/B fills, while the strips stay resident [calc]:

    (M_t + N_t)/(64 t) lines/cy
      32x32 (t=2) -> 0.500          16x64 (t=2) -> 0.625
      32x64 (t=4) -> 0.375          16x32 (t=1) -> 0.750

    i.e. the loads that are first touch are (M_t+N_t)/(t*(m_za+n_za)) of all
    loads; a square-ish logical tile shares better, a tall one wastes A.

C fills, when the accumulator does not fit L1 [calc]:

    8/k_mu lines/cy

MLP requirement = fills/cy x L2 latency; at 25 cy [calc].  All per-cycle
figures in this document use the peak-rate denominator: 8 cy/k, i.e. the
time the k-loop would take at 1 FMOPA/cy.  The model's own achieved-rate
denominator is different because it predicts 61 %, not peak.

    k_mu=16, accumulator resident   0.500 fills/cy -> 13 in flight
    k_mu=32, accumulator resident   0.500 fills/cy -> 13    (32x32, 24 KiB)
    k_mu=64, accumulator spilled    0.625 fills/cy -> 16    (32x32, 40 KiB)
    today (16x32, kc=208)           0.805 fills/cy -> 20   [model]

L2 bandwidth [calc]:

    read  = 64 x fills/cy                    B/cy
    write = 512/k_mu B/cy if the accumulator lives in L2, ~0 if in L1

Measured L2 capability for comparison [meas, inferred]: today's kernel pulls
~48-50 B/cy of L2 read to sustain 8 cy/k, so 32-40 B/cy is not the binding
constraint anywhere in the feasible region.

DRAM (whole GEMM), in passes of one matrix, with
    jc = ceil(N/n_c), ic = ceil(M/m_c), pc = ceil(K/k_c),
    tau = 1 if the C accumulator spans the kc blocks, else tau = pc [calc]:

    reads  = jc (A source) + 1 (B source) + ic (B per m-block) + tau (C)
    writes = tau (C) + 1 (B packed write-back) + [jc if A_c evicted]

At N=1024 (jc=1, ic=6, pc=5), one pass = N^2 doubles = 8.4 MB:

    today (tau=pc=5):  reads 13 passes = 109 MB   writes 6 = 50 MB
    design (tau=1):    reads  9 passes =  75 MB   writes 2 = 17 MB

So the micro-blocking does NOT change DRAM traffic by itself; only the
accumulator's lifetime does.  Nothing new appears in DRAM either: the C
buffer is not extra traffic, it is where C lives.

Requirement summary at 1 FMOPA/cy [calc]:

    k_mu   L1 B/cy   fills/cy   L2 rd B/cy   live set (32x32)   comment
      16      112       0.500         32           16 KiB        L1 88 %
      24       91       0.500         32           20 KiB
      32       80       0.500         32           24 KiB        recommended
      48       69       0.583         37           32 KiB        at the edge
      64       64       0.625         40           40 KiB        spills


5. Residency: what actually stays in L1 (simulation)
----------------------------------------------------

Estimates above assume the A/B strips *and* the C accumulator can be
resident.  Whether they can is a residency question, not a bandwidth one, so
it was simulated: 32 KiB L1 = 512 lines, fully associative, true LRU, the
pass order (1,2,3,4) within a k-chunk, A/B strips as new data each chunk
(the k octet advances), C accumulator lines re-touched once per chunk.
Counted over 40 chunks, steady state.  Fully associative is a best case; see
caveats.

    tile    t  C buf  k_mu  live set  fills/cy  L2 rd B/cy  L1 B/cy
    16x32   1    4 K  any   10-28 K     0.750       48        64-112
    32x64   4   16 K   16      28 K     0.750       48         112
    32x64   4   16 K   32      40 K !   0.625       40          80
    32x64   4   16 K   64      64 K !   0.750       48          64
    32x32   2    8 K   16      16 K     0.500       32         112
    32x32   2    8 K   32      24 K     0.500       32          80
    32x32   2    8 K   64      40 K !   0.625       40          64
    16x64   2    8 K   16-32   18-28 K   0.625       40        80-112

The `16x32` row is this same simulation with t=1, i.e. no cycling: 0.750.
The model's number for today's kernel is 0.805 (it also charges the C
epilogue and the packing reads); the difference between 0.750 and 0.805 is
those terms, not the tile.

Readings:

  * Four ZA tiles do not work with plain LRU.  The C accumulator is touched
    once per pass, so a full A/B chunk (192 lines at k_mu=16) flows between
    two touches of the same tile; a 16 KiB accumulator plus that stream makes
    L1 87 % live and LRU evicts the accumulator every chunk.  The C fills
    that come back exactly cancel the 2x saving on A/B: 0.750, i.e. today.

  * Two tiles arranged 32x32 (split in m) is the sweet spot: 8 KiB of C plus
    128 lines of strips = half of L1, so LRU keeps the accumulator and fills
    sit at the A/B compulsory rate, 0.500, for every k_mu up to 32.

  * Splitting the same two tiles in n (16x64) gives 0.625: the sharing ratio
    is (M_t+N_t)/(t*(m_za+n_za)) = 5/6 rather than 2/3, and the asymmetry
    wastes A loads.

Design rule that falls out:

    fills/cy = (M_t + N_t)/(64 t)     while (M_t+N_t) k_mu 8 + t*4 KiB < L1/2
             = 0.75 (no reuse)        once the live set passes about half L1

not a cliff: the transition is gradual as the accumulator loses more of itself
per chunk, which is where the 0.583 and 0.625 rows of the table above come
from (24% - 25% worse than the compulsory rate, not 50%).

Keeping the live set to half L1 is what buys LRU the slack to retain the
accumulator; without it the design degenerates to the current fill rate while
adding C traffic.


6. Recommended operating point
------------------------------

32x32 logical tile (t=2, split in m), k_mu = 32 [calc/sim]:

    L1 bandwidth      80 B/cy   (62 % of ~128)
    fills             0.500/cy  vs 0.805 today   (1.6x fewer)
    L2 read           32 B/cy   vs 48-50 today
    in-flight misses  13 at 25 cy latency, vs 20 today
    live set          24 KiB of 32 KiB
    C traffic         32 passes over C at k_mu=32 (L1/L2-local, not DRAM)
    DRAM              unchanged unless the accumulator spans kc blocks
                      (tau=1: reads 13 -> 9 passes, writes 6 -> 2)

k_mu=16 trades 112 B/cy of L1 (88 %) for the same 0.5 fills; k_mu=48/64 lose
the accumulator and land at 0.583/0.625.  If the core can make A/B loads
non-allocating (streaming hints) or the accumulator can be way-locked, the
4-tile/16 KiB variant reaches 0.375 and then t=4 becomes the better choice;
without that policy support, take t=2.

If the per-call cost is not solved independently, none of the above shows up
in wall clock: 300-500 cy/call at 10 240 calls is 18-30 %, and the tile
cycling *adds* C events.  Hiding the C epilogue (prefetch one tile ahead) is
worth +12 % in the model and needs only 64 lines = 4 KiB of the 133 KiB of
spare L1 bandwidth per call.


7. Closed forms for the standard counters
-----------------------------------------

Used above, and validated by exact-integer agreement with `model.py` over
square and rectangular problems, jc = 1 and jc = 2, and four residency
regimes (1 703 936 / 5 636 096 / 917 504 / 393 216 lines at N=1024).
Notation: s bytes/element, L line size, v doubles per uop,
ic = ceil(M/m_c), pc = ceil(K/k_c), jc = ceil(N/n_c).

L1D_CACHE (memory uops; residency-independent: these count accesses, not
misses):

    ukr    [ M N K (1/m_r + 1/n_r) + 2 M N pc ] / v
    packm  2 (jc M K + N K) / v

L1D_CACHE_REFILL (lines; nothing survives this 32 KiB L1, so every read that
enters the core is a fill):

    [ M N K (1/m_r + 1/n_r) + M N pc + jc M K + N K ] s / L

L2D_CACHE_REFILL (lines), split by issuer and by whether the operand's block
is resident in L2:

    operand  ukr, block EVICTED      ukr, RESIDENT   packm (always)
    A        M N K / n_r             0                jc M K
    B        ic N K                  0                N K
    C        pc M N                  M N              --

L2D_CACHE_WB (lines), same splits, writes only:

    C        ukr:  pc M N (evicted) or M N (resident)
    A        packm: jc M K if A_c evicted, else 0
    B        packm: N K if B_c evicted, else 0

Notes that cost time to work out, worth keeping:

  * A's evicted term is a *no-reuse fallback*: A's only reuse rung is jr, so
    losing it costs the full core traffic, M N K / n_r (N/n_r passes over A,
    32 here).  B's evicted term is not analogous: it keeps the surviving ir
    rung (11x), giving ic N K (6 passes).  C has one rung (pc) and falls back
    to pc M N.
  * The residency switches are rung tests of the form "invariant + one
    iteration's streamers", NOT block-size comparisons.  At kc=416 A_c is
    624 KB, under the 768 KB L2, and still evicted, because jr:A reports
    "needs 776K resident > 768K".  B_c's test at N=1024/kc=208 is
    1.66 + 0.31 + 1.55 MB = 3.5 MB, which is why kc=104 with a 2 MB L2 still
    counts as evicted.
  * DO NOT scale C's DRAM term by jc.  jc partitions the n dimension, so each
    jc iteration touches a disjoint M x n_c slice of C: total M N tau.  Only
    A carries a jc factor, because the whole of A is re-swept per n-panel.
    (Checked against the model at jc=2: C = M N pc, not jc M N pc.)
  * The difference from a model that has no packing term at all is exactly
    one term: the packer's read of B's source, N K s/L.  A's compulsory pass
    appears in both, in one issued by the ukr and in the other by the packer.
    (Checked term by term against an independent analytic model: identical
    up to that term, with ceil() on both block counts and A left unpadded,
    M K rather than ceil(K/k_c) M k_c.)
  * The same pack traffic appears in L1D_CACHE as 2(jc M K + N K)/v (source
    read + packed write), and in L2D_CACHE_WB only when the packed block is
    evicted -- a packed block that stays resident is repacked in place and
    never written back (the model prints exactly that verdict for A_c).
  * s/L == 1/v numerically (8/64 = 1/8) only because vl = L = 64 B here.
    Keep them symbolic: --vl (bytes per vector uop) and each level's line
    size are independent knobs, and only the uop counter follows --vl.
  * kc=416 as a way to halve the call count is a trap: A_c goes to 624 KB,
    leaves L2, and the prediction falls to 41.9 FLOP/cy (33 %) as DRAM reads
    go 1.70 M -> 5.64 M lines.  Fewer epilogues must come from accumulating
    across k-blocks, not from deepening the packed blocks.


8. Caveats
----------

  * Everything about L1 capacity is a model/simulation statement, not a
    measurement.  The 128 B/cy L1 bandwidth, the MSHR count and the L1/L2
    latencies in the spec above were guesses; the measured 8 cy/k says the
    MSHR guess is too tight.  Re-derive before trusting any "does not fit"
    conclusion.
  * Section 5 uses a fully associative LRU.  A real 512-set 8-way L1 with
    power-of-two panel strides will alias and do worse than 0.5.  Check the
    strips' set-index behaviour before believing the recommended point.
  * Section 5 also assumes the pass order 1,2,3,4 inside each chunk.  A
    tile-major order (all chunks of tile 1, then tile 2) removes the C
    cycling entirely, reverting to today's 0.75 fills with no C cost.
  * The design needs C staged somewhere; `str za` can write back into C
    itself (in-place read-modify-write, 64 B aligned), so no separate buffer
    is required for C.  The packed panels are the buffer problem: A_c 312 KB,
    B_c 1.66 MB at kc=208, which cannot come from a stack buffer --
    BLIS_STACK_BUF_MAX_SIZE is registers x register width (a few KiB,
    frame/include/bli_kernel_macro_defs.h:192), with
    BLIS_INSUFFICIENT_STACK_BUF_SIZE as the guard.
  * Whether the 300-500 cy per-call cost is per packed k-block call (pc
    outside, 5 calls per tile at N=1024) or per (ic,jr) tile (one call over
    all k) changes the whole picture: in the latter case the epilogue is 5x
    rarer, you are already near the 87 % ceiling, and only the lookahead
    gain remains.  Establish which before building anything.
  * Estimates assume the peak is really 1 FMOPA/cy sustained and that the
    epilogue rate (2 fmla/cy) is what bounds the ZA moves.  It is probably
    not: `ldr za`/`str za` move a slice per instruction, so a 4 KiB tile is a
    handful of instructions limited by L1 bandwidth (~64 cycles), not by fmla
    throughput (256 cycles at 0.094 cy/element).  Hand-pricing the epilogue
    with epi_peak over-states it ~6x; if the term is added to the model, give
    it its own rate.


9. Model gaps and the knobs that would close them
-------------------------------------------------

`model.py` cannot express any of this today:

  * `parse_nest` rejects a nest that does not end in `m:mr n:nr`, so no rung
    can sit below the register loops; the micro-kernel's k-loop is the one
    non-rung case.
  * There is one C invariant, the `pc` rung, sized M x n_c.
  * `kc` conflates two things an SME kernel separates: the packed A/B block
    depth (panel footprint, packing traffic) and C's lifetime (epilogue
    frequency, C DRAM traffic).

The smallest useful change is `--k-accum N` (or a trailing `k:k_mu` accepted
by `parse_nest`): charge the C traffic and the per-call epilogue per
ceil(K/k_accum) instead of per ceil(K/kc), leaving the panel footprint and
the packing tied to kc.  With the default k_accum = kc the existing numbers
must be reproduced exactly, so `scripts/eqcheck.sh` should show additions
only.  Then tau becomes an input rather than a hand-set quantity, and the
C-fill term 8/k_mu becomes decomposable from the rung structure.

If the chunk rung is added properly, two pitfalls already known from the
MOMMS work apply: beta belongs at the first k-step of every chunk, not at
pc == 0 (same class of bug as B3A2C0 applying beta twice), and no packer can
run on a register rung, so packing placement stays at ic/pc.

If the epilogue cost is to be modelled rather than guessed, it needs its own
rate and its own level (the accumulator's home), not `--epi-peak`.


10. Open measurements
---------------------

  1. Real L1D MSHR/fill-buffer count and L1/L2 latency: everything about
     fills rests on them, and the spec in the runs above was invented.
  2. Whether the core can make A/B loads non-allocating (or L1D ways can be
     locked): that is the difference between the 0.500 and 0.375 fill rates,
     and between t=2 and t=4 being the right shape.
  3. Decomposition of the 300-500 cy/call: ZA load, ZA store, alpha/beta
     scaling, loop setup.  Without it the prefetch/lookahead case cannot be
     sized.
  4. Whether ZA is preserved across kc blocks today, i.e. where the
     accumulator is actually flushed.
  5. Per-call and per-k cost with the tile at 32x32 to confirm the strip
     sharing: 8 cy/k should hold if the strips hit L1, and the per-call cost
     should not grow by the C cycling.
