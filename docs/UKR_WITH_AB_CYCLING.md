SME micro-kernel with AB cycling
================================

Design notes for a BLIS micro-kernel on an SME core (SVL = 512 bits, fp64) in
which the ZA array is cycled through several tiles inside one micro-kernel
call: the partial product (AB) is staged in L1 between passes, C is touched
only in the epilogue, and the k-chunk depth is the tuning knob.  Nothing here
is implemented: the numbers are traffic estimates, a residency simulation, and
predictions from `model.py` where it can express the structure at all.

Vocabulary, since the two are easy to confuse: AB is the accumulated product
of this call -- what ZA holds and what gets staged in L1 -- and C is the
caller's matrix, read and written once per packed block in the epilogue.
"Accumulator" below always means the AB tile, never C.

Status: `model.py` on this branch CANNOT express the design.  `--nest`
requires the innermost loops to be the register loops:

    model.py: error: --nest: the nest must end in the register loops,
             m:16 and n:32 in either order (m:mr n:nr)

so there is exactly one C invariant in the model (the `pc` rung, sized
`M x n_c`) and no way to put a k rung below the register loops whose
invariant is a ZA-sized *accumulator* tile (AB here).  The model also has no
AB at all: it charges C's traffic per `pc` pass, which is the same shape as
AB staging but a different operand.  See "Model gaps" at the end.

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
         L1D has 2 ports - one ld, one ld/st for 128 B/cycle read
         or (64B read + 64B write)/cycle

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

The model's OTHER limiter at this blocking [model]:
with the spec's 16 L1 MSHRs / 5 cy L1 / 16 cy L2 it needs ~12.7 outstanding
fills to sustain peak, so it predicts 71-72 % of peak limited by 
"L1 fill buffers", and 78 % with a 32-entry MSHR file, rising to the 
87 % compute ceiling once the C epilogue is also hidden
(section 4).  Treat every "the panels do not fit L1" conclusion in this
document as conditional on the real MSHR/latency numbers.


3. The proposal
---------------

One micro-kernel call computes a logical tile of t ZA tiles (example: t=4, 
2 x 2 = 32x64).  Inside the call, a k-chunk loop interleaves the tiles so
that A and B strips are shared across ZA passes:

    for k-chunk of depth k_mu in 0..k_c:
      for each ZA tile (r, c) of the logical tile:
        ldr za  <- AB buffer[(r,c)]        # 4 KiB
        for k in chunk:
          fmopa into ZA using A strip(r), B strip(c)
        str za  -> AB buffer[(r,c)]

  t        number of ZA tiles in the logical tile (1, 2, 4, ...)

Three shapes, and the names matter.  One ZA pass over one ZA array of AB for
k_mu k-steps is the *nano-kernel*: m_za x n_za doubles = 16x32 fp64 at
SVL=512 (8 fp64 subtiles za0-za7 of 8x8 each, which is what one FMOPA
accumulates into).  BLIS calls its register-resident tile MR x NR; on SME the
register file for accumulators IS ZA, so MR x NR = the nano shape whenever the
micro-kernel is exactly one ZA array.  A micro-kernel whose tile is larger
than ZA -- the design here -- cannot hold MR x NR in registers at all: that is
what forces the cycling, the logical tile (rows x cols nano-kernels,
`--shape`) plays the role of MR/NR, and the packed panel width follows the
logical tile, not the nano shape.

  k_mu     k-steps per ZA pass: the knob that trades AB traffic against the
           L1 footprint of the A/B strips
  tile     how the t tiles are arranged.  Splitting in m matters:
           32x32 is better than 16x64, see section 5.

What it buys: A is reused across column groups and B across row groups
without either being resident at full panel depth, so the strips come from
L1 instead of L2.

What it costs: every ZA pass moves its 4 KiB AB tile out and back, i.e.
2 x 4 KiB per tile per chunk, and the AB buffer (t x 4 KiB) competes with the
strips for L1.  C itself is not part of that: it is read and written once per
packed block, in the epilogue.


4. Requirements at 1 FMOPA/cycle
--------------------------------

Per cycle, at 1 FMOPA/cy = 64 MAC/cy, with one k-step of a ZA tile = 8 cycles
and 384 B of A/B [calc]:

    A/B loads (L1 side) = 48 B/cy                loads issued: fixed by the
                                                 ZA tile, because an FMOPA
                                                 takes z-register operands and
                                                 each ZA pass can only reuse
                                                 the vectors it holds
    AB loads+stores      = 1024/k_mu B/cy         one 4 KiB AB tile out and
                                                 back, per tile per chunk,
                                                 independent of t
    L1 total            = 48 + 1024/k_mu B/cy

The two L1 ports make that split matter.  The core has one load-only port and
one port that does either a load or a store, 64 B/cy each, so:

    loads  <= 128 B/cy          (both ports)
    stores <=  64 B/cy          (the shared port alone)
    loads + stores <= 128 B/cy  (a store cycle leaves only one port for loads)

which turns the single "L1 total" above into [calc, at 1 FMOPA/cy]:

    k_mu   loads B/cy   stores B/cy   combined   verdict
      8       112           64           176      over on both counts
     16        80           32           112      62 % / 50 % of their ports
     24        69           21            90
     32        64           16            80

so the design needs k_mu >= 13 for the combined cap and k_mu >= 8 for the
store cap, i.e. the same bound the single-number model gave -- but the margin
is now explicit, and it is the STORE port that is half busy at k_mu=16 rather
than the total.  The scheduling consequence: during a 4 KiB store (64 cycles
of the store port, half of a 128-cycle pass at k_mu=16) the k-loop still gets
64 B/cy from the load-only port and needs 48, so a store burst cannot starve
it -- but only 16 B/cy is left for prefetches in those cycles.  Hence
interleaving the str za / ldr za slices is a scheduling requirement, not a
bandwidth one: spread them across the pass so the drain overlaps the k-loop,
and keep any prefetch out of the store cycles.

A/B fills (L2 side), while the strips stay resident [calc]: the unique data,
which is what a bigger logical tile actually reduces:

    (M_t + N_t)/(64 t) lines/cy          in bytes/cy at 1 FMOPA/cy
      32x32 (t=2) -> 0.500  = 32 B/cy        16x64 (t=2) -> 0.625 = 40
      32x64 (t=4) -> 0.375  = 24 B/cy        16x32 (t=1) -> 0.750 = 48

    i.e. the loads that are first touch are (M_t+N_t)/(t*(m_za+n_za)) of all
    loads; a square-ish logical tile shares better, a tall one wastes A.

    The two rows must not be confused: a 32x64 tile runs 4v x 8v of sub-tiles
    = 32 FMOPAs = 32 cycles per k-step, so its unique operand data is
    (32+64)*8 = 768 B per 32 cy = 24 B/cy, while the loads ISSUED are 4 passes
    x (2v + 4v) = 1536 B per 32 cy = 48 B/cy.  The saving is in what the loads
    hit, not in how many are issued.  24 B/cy on both sides needs the tile
    resident in the accumulator: at fp64 that is 16 KiB of ZA, i.e. SVL=1024.
    At SVL=512 a 32x64 tile must cycle, which is what costs the second
    24 B/cy.  If a wider SVL is on the roadmap it beats every cycled variant
    here outright -- no AB reload traffic at all.  Note that 0.375 is a BOUND
    for A/B fills: reaching it also needs the AB buffer to survive each pass,
    which section 5 shows the 4-tile layout cannot do with packdim = MR/NR.

AB fills, when the accumulator does not fit L1 [calc]:

    8/k_mu lines/cy

Epilogue: AB is not C
---------------------
The 2 x 4 KiB per pass above is AB staging (the partial product), not C.  The
real C is touched once per kc block per logical tile, in the epilogue:

    per tile:   read AB   4 KiB   L1 hit (except the LAST tile, whose AB is
                                  still in ZA -- a mova to Z costs no memory)
                read C    4 KiB   miss stream: L2 if prefetched, else RAM
                write C   4 KiB

so per logical tile per block: (t-1) AB reads, t C reads, t C writes, i.e. at
t=2 12 KiB read + 8 KiB write, against the 12 chunks x 8 KiB = 96 KiB of AB
staging that the k-loop did in the same block [calc, kc=192]:

     k_mu   epilogue     L1 reads   L1 stores   combined (of 128 / 64 / 128)
      16    +6.7 B/cy       84          35          119  (93 %)  tight
      24    +4.5            73          24           97  (76 %)
      32    +3.4            68          19           87  (68 %)

So the epilogue is cheap in deadline terms and not free in bandwidth: it costs
about a fifth of the AB staging traffic, and at k_mu=16 that is the difference
between 88 % and 93 % of L1 -- an argument for k_mu = 24-32 rather than 16.

What it BUYS is the latency.  The model charges 444 cycles of "exposed C
fetch" per call because in a classic epilogue the C load has nothing to hide
behind.  Here it does: (t-1) AB tiles are sitting in L1 and the epilogue has
12 KiB of L1 moves per tile to run while the C loads for the later tiles are
outstanding, which is 100-190 cycles at the port rates -- plenty to cover one
16-21 cycle C latency, and the whole exposure disappears if the C loads are
issued one tile ahead.  That is the +12-15 % the model's --lookahead knob
predicted (section 4); in this structure it is inherent rather than an
optimisation.

It is also why a beta==1 / alpha==1 fast path that preLOADs C before the
k-loop is the wrong shape here.  Holding C in registers across the k-loop is
what makes that path slow in an ordinary micro-kernel (live range, register
pressure, no reuse to show for it).  This design never does that: it stages AB
in L1 and touches C only in the epilogue, so the fast path should skip the
multiply, not move the load.

`--epilogue` in the simulation adds exactly the term above (and counts the C
read as a miss stream), so the bandwidth tables can be read with or without
it.

First accumulation and dc zva
-----------------------------
The first chunk of a packed block starts from AB = 0, so its accumulator load
is unnecessary: skipping it (or dc zva-ing the lines) saves one 4 KiB read per
tile per block, which at k_c=192 / k_mu=16 is one chunk in twelve [sim,
t=2, with the epilogue]:

     L1 reads   84 -> 81 B/cy      (93 % -> 91 % of the load port's 128)
     L1 stores  35 -> 38 B/cy if the zeroing is itself a line write
     fills      unchanged (0.500/cy), combined unchanged

i.e. a like-for-like move from the load port to the store port, worth having
but not decisive.  What IS decisive is the failure mode it insures against: the
whole budget above assumes a full-line staging store does not fetch its line.
If the core still reads for ownership, that is another t x 4 KiB per chunk --
32 B/cy at k_mu=16, taking L1 from 119 to 151 of 128, i.e. infeasible (and
still 103 of 128 at k_mu=32).  dc zva on the first store of each tile removes
that risk; without it the design is one microarchitectural detail away from not
fitting at all.  --dc-zva models the read saving.

Latency: ADDITIVE, and the figure matters a lot.  The model's `--cache`
latency field is what a level ADDS to a miss from the level inside it, not
load-to-use (module docstring: total hits of 7/14/46 cycles are written
7, 7, 32).  The target's L2 is reported as ~16 cycles, which may be
load-to-use, i.e. it would include the L1 hit path:

    if 16 is load-to-use  ->  additive L2 = 16 - (L1 hit, 4-5)  ~ 11-12
    if 16 is additive     ->  use 16
    everything above used 21, which only holds if the L1 hit costs ~0

[model] 16x32, kc=208, N=1024, L1 latency 5:

    L1 lat  L2 lat  mshr   FLOP/cy  %peak  exp/call  limiter
      5      21      16     78.4     61      444     ukr: L1 fill buffers
      5      16      16     91.2     71      424     ukr: compute + exposed C fetch
      5      11      16     92.0     72      404     ukr: compute + exposed C fetch
      5      21      32    100.0     78      222     ukr: compute + exposed C fetch
      5      16      32    100.4     78      212     ukr: compute + exposed C fetch
      5      11      32    100.9     79      202     ukr: compute + exposed C fetch

    and with the C epilogue hidden (--lookahead 300-600):
      5      16      16     92.3     72        0     ukr: L1 fill buffers
      5      11      16    111.7     87        0     ukr compute
      5      16      32    111.7     87        0     ukr compute

The 21-cy guess was pessimistic, but latency alone does not explain the gap:
at 16 MSHRs the model moves only 61 % -> 71-72 % and stays fill-queue bound,
because shortening the latency does not relieve a queue that is already
saturated.  The 87 % ceiling (111.7 FLOP/cy = compute) needs one of two
things -- a bigger MSHR file (32 entries: 78 % before hiding C, 87 % after) or
a lower latency together with hiding the C epilogue (11 cy and --lookahead
600: 87 %).  The measured 81-85 % end to end is consistent with a partly
hidden epilogue on either point, which is as far as these two numbers can
pin it down.

In-flight requirement, fills/cy x latency, with the share of a 16-entry MSHR
file it occupies [calc].  "Resident" means the AB accumulator stays in L1, which
is what determines the fill rate (section 5):

    point (fills/cy)          11 cy        16 cy        21 cy        25 cy
    today, 16x32     0.805   8.9  55 %   12.9  81 %   16.9 106 %   20.1 126 %
    32x32 k_mu<=32   0.500   5.5  34 %    8.0  50 %   10.5  66 %   12.5  78 %
    32x64 k_mu=16    0.625   6.9  43 %   10.0  63 %   13.1  82 %   15.6  98 %
    32x32 k_mu=64    0.625   6.9  43 %   10.0  63 %   13.1  82 %   15.6  98 %
    32x64 k_mu=32    0.750   8.3  52 %   12.0  75 %   15.8  98 %   18.8 117 %

    max sustainable fill rate = mshr / latency:  1.45/cy   1.00/cy   0.76/cy   0.64/cy

The RATIO the design buys is latency-independent (it is a fill-rate ratio):
1.6x fewer fills wherever the accumulator is resident (0.805 -> 0.500), only
1.3x where it spills (0.625), and 1.07x at the 0.750 points.  What is NOT
latency-independent is whether that matters:

  * at 21-25 cy the current kernel needs 106-126 % of a 16-entry file, i.e. it
    cannot sustain peak at all, and AB cycling brings it to 66-78 %: the
    difference between infeasible and feasible.
  * at 16 cy it needs 81 %, which fits.  AB cycling takes it to 50 % -- real
    headroom, but not a fix for an existing failure.
  * at 11 cy it needs 55 %, so there is nothing here to fix and the fill-buffer
    argument for the design evaporates.

[model] proxy runs make the same point from the other end.  Using mr/nr as the
logical tile (so the fill rate 1/mr + 1/nr matches, C charged once per kc),
16 MSHRs, kc=208:

    tile    L2=21        L2=16        L2=11
    16x32   78.4  61 %   91.2  71 %   92.0  72 %   ukr: fill buffers -> compute
    32x32   90.5  71 %   91.2  71 %   92.0  72 %   same at 16 and 11
    32x64   90.5  71 %   91.2  71 %   92.0  72 %

i.e. the bigger tile buys 61 % -> 71 % at 21 cy and NOTHING at 16 or 11 cy,
because the limiter has moved off the fill queue and the (proxy's) epilogue
takes over.  The proxy cannot represent the cycled AB, so its epilogue term
over-states the design's cost -- but the fill-buffer conclusion does not
depend on that: at 16 cy there is no fill-buffer shortfall for AB cycling to
remove.

Three consequences:

  * A 16-cycle L2 makes today's kernel feasible with ~13 of 16 MSHRs rather
    than over-subscribed, so the measured 8 cy/k is consistent with an
    ordinary 16-entry MSHR file.  Section 2's "needs ~12.7 outstanding fills"
    already reflects the 16-cy figure; the 17 came from the 21-cy guess.  (The model's fill-buffer test is not
    exactly this product -- it also pools C's burst waves -- so treat the
    table as first order.)
  * The design's MLP motivation shrinks: 20 -> 13 at 21 cy becomes 13 -> 8 at
    16 cy, and 13 already leaves 24 % headroom against a 1.0/cy sustainable
    fill rate.  Fewer fills per MAC stays the right target, but it is no
    longer the difference between working and not.
  * The two fixes are SEQUENTIAL, not additive.  Hiding the C epilogue is
    worth +15 % at 11 cy additive (92.0 -> 105.9 -> 111.7 as lookahead goes
    0 -> 300 -> 600) but only +1 % at 16 cy with 16 MSHRs (91.2 -> 92.3)
    and nothing at all at 21 cy, because the fill queue is then the binding
    constraint.  So: relieve the fills first, and only then does prefetching
    C ahead pay.

L2 bandwidth [calc]:

    read  = 64 x fills/cy                    B/cy
    write = 512/k_mu B/cy if the accumulator lives in L2, ~0 if in L1

Measured L2 capability for comparison [meas, inferred]: today's kernel pulls
~48-50 B/cy of L2 read to sustain 8 cy/k, so 32-40 B/cy is not the binding
constraint anywhere in the feasible region.

DRAM (whole GEMM), in passes of one matrix, with
    jc = ceil(N/n_c), ic = ceil(M/m_c), pc = ceil(K/k_c),
    tau = 1 if AB accumulates across the packed k-blocks (this design, since
    the AB buffer outlives them), else tau = pc [calc]:

    reads  = jc (A source) + 1 (B source) + ic (B per m-block) + tau (C)
    writes = tau (C) + 1 (B packed write-back) + [jc if A_c evicted]

At N=1024 (jc=1, ic=6, pc=5), one pass = N^2 doubles = 8.4 MB:

    today (tau=pc=5):  reads 13 passes = 109 MB   writes 6 = 50 MB
    design (tau=1):    reads  9 passes =  75 MB   writes 2 = 17 MB

The C term is the real C here, and only it moves: today the C tile is read
and written once per packed k-block (pc times per tile), in the design once
per (ic,jr) tile, in the epilogue, because AB keeps accumulating across the
k-blocks.  That is a property of AB's lifetime, not of the cycling -- the same
saving is available by accumulating across k-blocks in any structure.

AB itself adds NOTHING to DRAM, and should not be counted there: the staging
buffer is t x 4 KiB (8-16 KiB), which is nothing against the 768 KiB L2, so
its read-modify-writes are L1/L2-local at worst.  Only if the buffer were
larger than L2 would AB appear in this table at all.


5. Residency: what actually stays in L1 (simulation)
----------------------------------------------------

The estimates above assume the A/B strips *and* the AB accumulator can be
resident.  Whether they can is a capacity question, not a bandwidth one, so it
is simulated rather than assumed.  The simulation is
`tools/sim_ccyc_ukr_fills.py`; every figure in this section reproduces from it
and `--selftest` pins the numbers quoted below.

How the simulation works:

  * 32 KiB L1 = 512 lines of 64 B, FULLY ASSOCIATIVE, true LRU: a hit moves
    the line to the MRU end, an insert evicts the LRU end.
  * One line holds 8 consecutive k of one row of A (or one column of B, or 8
    doubles of AB), so a line is touched once per octet of k, not once per
    k-step.  This is what turns "m_za k_mu elements" into "m_za k_mu/8 lines".
  * The access sequence is generated from the loop structure: for each ZA
    pass (tile r,c) of a k-chunk, the accumulator's 64 lines are touched
    (load), then for each k in the chunk the A strip's line for that octet
    followed by the B strip's, then the accumulator's 64 lines again (store).
  * Chunks are simulated in sequence with k advancing, so the A/B lines are
    new data each chunk while the AB lines are the same lines re-touched; the
    reported figure is the steady-state mean after a warmup.
  * A "fill" is any touch of a non-resident line.  Stores count like loads
    (the accumulator's line is the unit of transfer); dirty state and
    write-backs are not modelled, so this counts fills only.

Does the ORDER matter?  One choice dominates: WHEN the accumulator is touched.
The real sequence is load AB, k-loop, store AB, i.e. twice per pass [sim]:

    point               AB at start  AB at end   AB at start AND end
    32x64, k_mu=16        0.750       0.750          0.375
    32x64, k_mu=32        0.625       0.625          0.844
    32x32, k_mu=32        0.500       0.500          0.500

Touching AB only once per pass lets the accumulator fall out of LRU between two
touches of the same tile even when the whole working set would fit; touching it
at both ends keeps it MRU-hot and recovers the compulsory rate.  At k_mu=32 the
second touch does HARM instead, because by then the live set exceeds L1 and
refreshing AB only evicts A/B sooner.  An earlier version of this table used the
one-touch pattern and therefore overstated the 4-tile case (0.750 instead of
0.375 at k_mu=16).

Order choices that do NOT change the result here: A and B interleaved per
k-step versus whole-A-strip-then-whole-B-strip (the strips are small, both fit);
load-then-store versus store-then-load within a pass (same lines).  Order
choices that DO change it: the pass order inside a chunk (tile-major in m
versus in n: 0.375 -> 0.625 at 32x64/k_mu=16), and tile-major ACROSS chunks
(all k for tile 1, then tile 2), which removes the AB cycling entirely and
reverts to the 16x32 row.

Fully associative baseline [sim], load+store accumulator pattern, and (for
the 32x64 row only) packdim = the nano shape -- see the MR/NR table below for
what a real build can have:

    tile    t  k_mu     live set    fills/cy    L2 rd B/cy
    16x32   1  8-48      7-22 K       0.750         48
    16x64   2  8-32     13-28 K       0.625         40
    32x32   2  8-48     12-32 K       0.500         32
    32x64   4  8-16     22-28 K       0.375         24   (nano packdim only)
    32x64   4  20-48    31-52 K    0.775-0.844    50-54

Readings:

  * The compulsory rates of section 4 are ACHIEVABLE, but only where the whole
    live set -- A/B strips plus the AB buffer -- fits: 0.500 at 32x32 up to
    k_mu=48, and 0.375 at 32x64 for k_mu <= 16.  The 32x64 point reaches the
    24 B/cy that a resident 4-tile accumulator would, with AB living in L1
    instead of in ZA -- but see the layout section: it needs packdim = the
    nano shape, which a cycling kernel cannot have.
  * 32x32 is the robust point -- flat from k_mu=8 to 48 at 32 B/cy -- and is
    the one section 6 recommends.  32x64 is a knife edge (window k_mu = 14-18,
    live set 26-30 K of 32 K) and collapses to 0.775 at k_mu=20.
  * The two knobs are not independent: the L1 bandwidth requirement
    48 + 1024/k_mu pulls k_mu UP while the capacity requirement
    (M_t+N_t) k_mu 8 + t 4 KiB < L1 pulls it DOWN, so t=2 wants k_mu=16-32
    and t=4 only k_mu=14-18.
  * A/B eviction (a strip fetched twice because the first copy was pushed out)
    appears only where the live set is far past L1: at 32x64/k_mu>=20 the
    shortfall against the compulsory formula is A/B re-fetch, not AB, and it is
    what makes those rows worse than today's kernel.

Design rule that falls out:

    fills/cy = (M_t + N_t)/(64 t)   while the live set fits L1 AND the
                                    accumulator is re-touched every pass
             = 0.5 - 0.85 otherwise (loses the accumulator, then the strips)

Associativity, and why the LAYOUT comes first
----------------------------------------------

Everything above is a FULLY ASSOCIATIVE LRU, i.e. a best case.  Modelling a
real cache is easy once the simulation has addresses -- give each line one,
index sets by (address / line) % nsets, keep one LRU list per set.  Twenty
lines of code and `--assoc W` does it.  The work is not the code, it is
getting the ADDRESSES right, and there are two layouts to distinguish:

  * packed micro-panels (what a BLIS/SME kernel actually reads): A is stored
    as m_za-wide panels of kc x packdim_a doubles with m_za contiguous, so the
    stride between consecutive k is packdim_a*esize (128 B for m_za=16, fp64)
    and the stride between panels is kc*packdim_a*esize.  B likewise with
    n_za and packdim_b.
  * row-major A / column-major B with k contiguous -- the shape of the first
    version of this simulation, kept as `--layout row`.  Its strides are kc
    long, so a "leading dimension" of a convenient round number aliases badly.

`--layout packed` is the default; `--packdim-a/-b` are the padding knobs, and
they are the same idea as BLIS's `BLIS_PACKDIM_MR` / `_NR` (which exist
precisely because MR/NR alone make the packed stride a power of two).

With packed panels at 8 ways (32 KiB, 64 sets x 8 ways), packdim = the nano
shape (i.e. the layout you get with no cycling at all) [sim]:

     shape  k_mu  live set  fills/cy fully assoc   8-way packed
     16x32    16     10 K        0.750              0.750
     16x32    32     16 K        0.750              0.750
     16x64    16     18 K        0.625              0.625
     16x64    32     28 K        0.625              0.625
     32x32    16     16 K        0.500              0.500
     32x32    32     24 K        0.500              0.500
     32x64    16     28 K        0.375              0.375
     (rows past L1 capacity are in the MR/NR table below, where they matter)

i.e. with no cycling at all the compulsory rates survive a real 8-way L1.
Take this table as a reference point only: it needs packdim = the nano shape,
and the realistic layout follows below.

Padding the STORED panel width (packdim, i.e. what BLIS_PACKDIM_MR/_NR does) is
the wrong knob here, and the simulation now says why.  packdim sets the stride
between consecutive k inside a panel; the AB buffer is contiguous -- 8 KiB at
t=2, 16 KiB at t=4 -- which is 2 and 4 lines respectively in every one of the
64 sets, so it leaves 6 and 4 ways of streaming per set.  Any packdim whose k stride in LINES is a multiple of 2
concentrates the strips onto fewer sets and starves AB [sim, 32x64/k_mu=16,
kc=192, 8-way; ABf is AB lines re-fetched per chunk, 512 cycles]:

     packdim a/b  k stride   sets the strips use      ABf   fills/cy
     16 / 32      2 / 4 L    64 / 64                    0      0.375
     32 / 32      4 / 4 L    32 / 64                   64      0.500
     24 / 40      3 / 5 L    64 / 55                   51      0.498
     32 / 64      4 / 8 L    32 / 32                   96      0.625
     16 / 64      2 / 8 L    64 / 32                   96      0.625

and padding the PANEL stride instead (--panel-pad, which leaves the k stride
alone) moves 0.375 to 0.379-0.387 only, so panel aliasing is not the limiter:
the k stride is.

With packdim = MR/NR = the LOGICAL tile -- the only layout an AB-cycling kernel
can have, since the accumulator is one ZA array at a time -- the same table
becomes [sim, 8-way, kc = 192 or 208]:

     shape  k_mu   packdim   live   fills/cy   AB lines re-fetched/chunk
     16x32    16    16/32     10 K    0.750       0
     16x64    16    16/64     18 K    0.625       0
     32x32    16    32/32     16 K    0.500       0
     32x64    16    32/64     28 K    0.625      96
     32x64    32    32/64     40 K    0.750     128  (over L1 capacity)

So the two-tile tile keeps its compulsory rate in a real build and the four-
tile one does not: its wider packed panel is what costs it the accumulator.

Past k_mu ~ 32 BOTH streams lose lines, and the tool reports them separately
(A/Bf is strip fills per chunk, uniq the unique strip lines, ABf the
accumulator's) [sim, 8-way, kc=192]:

     shape  k_mu   A/Bf / uniq     ABf       fills/cy
     32x32    16      128 / 128         0     0.500     exact
     32x32    32      256 / 256         0     0.500     exact
     32x32    48      384 / 384         0     0.500     exact
     32x32    64      640 / 512       128     0.750     both streams over
     32x64    16      224 / 192        96     0.625
     32x64    32      640 / 384       128     0.750
     32x64    48     1056 / 576       256     0.854
     32x64    64     1408 / 768       320     0.844     1.8x A/B, as bad as today

i.e. the capacity rule applies to the strips as much as to the accumulator:
once the live set passes L1, A/B is re-fetched 1.8x and the bigger tile buys
nothing at all.

That is why section 6 recommends the two-tile version: not because it gives a
quarter of the fills more, but because with packdim = --shape it is the only
one that cannot avoid the padding.

So a single 8-way number is not meaningful: sweep kc and packdim.  The 4-tile
point's 0.375 needs packdim = the nano shape and is not reachable in a real
build, while the 32x32 point reads 0.500 in every configuration tried.

For contrast, the row-major layout with a 1024-byte leading dimension (1024 B
= 16 lines = a quarter of the 64 set count) gives SIX fills per cycle -- every
touch misses.  That is what a bad stride does, and it is why the layout, not
the associativity, is the thing to model first.


6. Recommended operating point
------------------------------

Two points survive, the second only on paper [calc/sim]:

  A. 32x32 (t=2, split in m), k_mu = 24..32 -- the one to build
       fills             0.500/cy  vs 0.805 today   (1.6x fewer)
                         AB never re-fetched: 0 accumulator lines per chunk
       L2 read           32 B/cy   vs 48-50 today
       L1 bandwidth     97 B/cy at k_mu=24 (76 % of 128), 84 at k_mu=32
                        (67 %); k_mu=16 is the same fills at 119 B/cy (93 %)
       in-flight         8 at 16 cy, 13 at 25 cy   (13 at 16 cy today)
       live set          20 KiB at k_mu=24 (62 %), 24 at k_mu=32 (75 %)
       holds at kc = 192/208/256, at 8 ways as well as fully associative

  B. 32x64 (t=4, split 2x2), k_mu = 16 -- WORSE than A in a real build
       fills             0.625/cy  (1.26x fewer than today)
                         the wider packed panel makes the strips skip sets, so
                         96 accumulator lines are re-fetched per chunk
       L2 read           40 B/cy;  live set 28 KiB (87 %), no slack
       the 0.375 / 24 B/cy figure needs packdim = 16/32, i.e. MR/NR = the
       nano shape -- which is exactly what an AB-cycling micro-kernel does not
       have (section 5)

Both need the AB buffer re-touched at both ends of every ZA pass (load AB,
store AB); with a single touch per pass the 4-tile point degrades to 0.750 and
the exercise is pointless (section 5).  AB traffic is 64 passes over AB at
k_mu=16, all L1/L2-local.  The DRAM saving (tau=1: reads 13 -> 9 passes,
writes 6 -> 2) comes from C being touched once per (ic,jr) tile in the
epilogue rather than once per packed k-block -- not from the cycling, and not
from any AB traffic, which never leaves L1/L2.

A wins, and the reason is a constraint rather than a tuning choice: the packer
writes panels MR x kc / NR x kc, and in an AB-cycling micro-kernel MR/NR IS the
logical tile (the accumulator is only one ZA array at a time, so nothing
smaller is "resident").  packdim therefore equals --shape, not the nano shape,
and the k stride inside a nano-pass is the padded width: 4 lines for A and 8
for B at 32x64, against slabs of 2 and 4 lines.  The strips then skip sets and
the 28 KiB accumulator cannot hold its 4 ways in every set.

A sidesteps it because its NR x esize = 256 B is the same 4 lines as the nano
B slab, so nothing changes for B, and its A panel at 32 rows only doubles a
strided footprint that was already half the sets; the accumulator stays whole
(0 lines re-fetched).

So the honest summary: 1.6x fewer fills and 1.6x less L2 read, at 76 % of L1
bandwidth and 62 % of its capacity, from the two-tile tile at k_mu=24 (all
inclusive of the epilogue).  The
24 B/cy resident-tile number is not reachable with packdim = MR/NR = the
logical tile; it would need either no cycling at all (MR/NR = one ZA array,
i.e. today's kernel) or an SVL wide enough that the logical tile fits ZA
(32x64 fp64 = 16 KiB, SVL = 1024).

No recommendation here survives an unknown latency, so read this point as
conditional (section 4): the fill-buffer argument for AB cycling is worth
everything at 21-25 cy, worth real but smaller headroom at 16, and worth
nothing at 11.  What survives at every latency is the L2 read bandwidth
(32 vs 48-50 B/cy) and the option value of 50 % instead of 81 % MSHR
occupancy when the L2 latency varies with DRAM contention.

Non-temporal strip loads (ldnt1d / ldnt1w)
------------------------------------------
A strip is dead after the last pass that uses it inside a chunk -- A(r) after
the last column group, B(c) after the last row group -- so those loads can be
non-temporal, while the earlier uses must allocate because a later pass in the
same chunk reuses them.  That is the right discipline, and the simulation says
where it pays [sim, 8-way, kc=192, 32x64 with packdim = MR/NR]:

     k_mu   none: fills / L2 rd        last-use ldnt: fills / L2 rd
      16      0.625     40.0 B/cy          0.594    38.0 B/cy
      24      0.792     50.7                0.635    40.7
      32      0.750     48.0                0.641    41.0

and on 32x32 it is EXACTLY neutral (0.500 and 32 B/cy either way), because
there the last-use load is already a hit and bypassing a hit changes nothing.

So the predictor is whether that load was a miss: with residency holding there
is nothing to gain; in the aliasing regime of the 4-tile layout it stops a
miss from evicting the AB buffer, which recovers most of the k_mu sensitivity
(0.792 -> 0.635 at k_mu=24) without reaching the 2-tile's floor.

Marking ALL strip loads non-temporal is a large loss instead (0.500 -> 0.750 on
32x32): the first use of every strip must stay cached, or the later pass
re-fetches it.  `--ldnt last` and `--ldnt all` in the tool; `--ldnt-pol
bypass|lru` chooses between not allocating and allocating as the first victim,
and the two give identical numbers here, so the conclusion does not rest on the
implementation's choice -- but a core that implements ldnt as "allocate
normally, only mark it" would show no gain at all, which is worth a
microbenchmark before relying on it.

fp32 at SVL=512
---------------
The nano tile doubles to 32x32 floats (still 4 KiB of ZA) and an FMOPA covers
16x16 floats at one per TWO cycles, i.e. 256 FLOP/cy -- the tool derives that
from `--fmopa-cycles 2` rather than a hand-computed `--peak`.  The traffic per
cycle is unchanged from fp64 (half the bytes per MAC, twice the MACs), but the
fill rate in lines/cy halves, so the same shapes come out better [sim, 8-way,
kc=256, epilogue, 1 FMOPA/2 cy]:

     shape   t  k_mu   live   fills/cy   L2 rd B/cy   L1 rd/wr   ABf
     32x32   1    16     8 K    0.500       32.0        66/34      0
     64x32   2    16    14 K    0.375       24.0        67/34      0
     32x64   2    16    14 K    0.375       24.0        67/34      0
     64x64   4    16    24 K    0.375       24.0        68/34     64

i.e. in fp32 the TWO-tile version (64x32, split in m) reaches the 0.375 /
24 B/cy that fp64's four-tile layout cannot, with no AB re-fetch and 14 KiB
live.  The four-tile version reports the same fill rate but is already losing
one AB tile per chunk (ABf 64) at 24 KiB live, i.e. it is on the edge of the
same capacity cliff: at k_mu=32 it costs 0.500 with both streams
re-fetching (section 5's rule).

If the per-call cost is not solved independently, none of the above shows up
in wall clock: 300-500 cy/call at 10 240 calls is 18-30 %, and the tile
cycling *adds* AB events.  Hiding the C epilogue (prefetch one tile ahead) is
worth +12 to +15 % once the fill queue is relieved, and needs only 64 lines
= 4 KiB of the 133 KiB of spare L1 bandwidth per call.


7. Closed forms for the standard counters
-----------------------------------------

In this section C is the REAL C of `model.py`, which has no AB staging: do
not rename these terms to AB.  The forms are used above to price the design's
epilogue and to compare against the current kernel.

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
    measurement.  The port split (128 B/cy of loads, or 64 + 64) is measured;
    the MSHR count and the L1 latency are still guesses, and the L2 latency is
    only reported as "~16 cycles" without
    saying whether that is additive or load-to-use -- which is worth 15
    percentage points of predicted peak (section 4).  Re-derive both before
    trusting any "does not fit" conclusion.
  * Associativity is now simulated (--assoc) and it is the LAYOUT that
    decides, not the associativity: with packdim = MR/NR = the logical tile an
    8-way L1 keeps the 2-tile compulsory rate exactly (0.500, 0 AB fills) but
    costs the 4-tile one a fifth of its accumulator (0.625, 96 lines per
    chunk); a row-major model with a 1024 B leading dimension thrashes
    outright (6 fills/cy).  Still unmodelled: the real
    packed strides (use --packdim-a/-b to match BLIS_PACKDIM_MR/_NR) and any
    prefetcher, which in hardware fetches A/B ahead and displaces the AB
    buffer in exactly the way the accumulator touch pattern is fighting.
  * Section 5 also assumes the pass order 1,2,3,4 inside each chunk.  A
    tile-major order (all chunks of tile 1, then tile 2) removes the AB
    cycling entirely, reverting to today's 0.75 fills with no AB cost.
  * AB needs a buffer of its own: t x 4 KiB (8 KiB at t=2, 16 KiB at t=4),
    which cannot be C itself, because C still holds the previous accumulation
    until the epilogue consumes it.  So the design pays for a staging buffer
    that a register-accumulator kernel does not have -- 8-16 KiB is more than
    BLIS_STACK_BUF_MAX_SIZE (registers x register width, a few KiB,
    frame/include/bli_kernel_macro_defs.h:192, with
    BLIS_INSUFFICIENT_STACK_BUF_SIZE as the guard), so it comes from the pool
    or an explicit stack allocation.  The packed panels are the *other*
    buffer problem: A_c 312 KB, B_c 1.66 MB at kc=208.
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
  * There is one C invariant, the `pc` rung, sized M x n_c -- and it is the
    real C; there is no AB in the model at all.
  * `kc` conflates two things an SME kernel separates: the packed A/B block
    depth (panel footprint, packing traffic) and C's lifetime (epilogue
    frequency, C DRAM traffic).

The smallest useful change is `--k-accum N` (or a trailing `k:k_mu` accepted
by `parse_nest`): charge the C traffic and the per-call epilogue per
ceil(K/k_accum) instead of per ceil(K/kc), leaving the panel footprint and
the packing tied to kc.  With the default k_accum = kc the existing numbers
must be reproduced exactly, so `scripts/eqcheck.sh` should show additions
only.  Then tau becomes an input rather than a hand-set quantity, and the
AB-fill term 8/k_mu becomes decomposable from the rung structure.

If the chunk rung is added properly, two pitfalls already known from the
MOMMS work apply: beta belongs at the first k-step of every chunk, not at
pc == 0 (same class of bug as B3A2C0 applying beta twice), and no packer can
run on a register rung, so packing placement stays at ic/pc.

If the epilogue cost is to be modelled rather than guessed, it needs its own
rate and its own level (the accumulator's home), not `--epi-peak`.


10. Open measurements
---------------------

  1. Real L1D MSHR/fill-buffer count, and whether the ~16-cycle L2 figure is
     additive or load-to-use.  Those two numbers decide whether the kernel is
     fill-queue-limited at 72 % (16 MSHRs, 16 cy additive) or compute-limited
     at 87 % (32 MSHRs, or 11 cy additive), and therefore whether AB cycling
     has a job to do at all.
  2. Whether the core can make A/B loads non-allocating (or L1D ways can be
     locked).  That is what would let the 4-tile version survive packdim = the
     logical tile: with the strips not allocating at all, the AB buffer would
     keep its 4 ways per set and 32x64 would reach its compulsory 0.375,
     against 0.500 for the 2-tile version built today.
  3. Decomposition of the 300-500 cy/call: ZA load, ZA store, alpha/beta
     scaling, loop setup.  Without it the prefetch/lookahead case cannot be
     sized.
  4. Whether ZA is preserved across kc blocks today, i.e. where the
     accumulator is actually flushed.
  5. Per-call and per-k cost with the tile at 32x32 to confirm the strip
     sharing: 8 cy/k should hold if the strips hit L1, and the per-call cost
     should not grow by the AB cycling.
