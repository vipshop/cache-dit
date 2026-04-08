<!-- PTX ISA 9.1 -->
# Hopper (sm_90) PTX Features

## sm_90 vs sm_90a

| Target | Features |
|--------|----------|
| `sm_90` | Clusters, `barrier.cluster`, DSMEM (`mapa`/`getctarank`), `cp.async.bulk.tensor` (TMA), cluster special registers, `mbarrier.try_wait`, `elect.sync` |
| `sm_90a` | `wgmma.*`, `setmaxnreg`, optimized `.multicast::cluster` on TMA. NOT forward-compatible (Blackwell uses `tcgen05.mma`) |

---

## Cluster Dimension Directives

### .reqnctapercluster
### Syntax
```ptx
.reqnctapercluster nx
.reqnctapercluster nx, ny
.reqnctapercluster nx, ny, nz
```
### Constraints
- Kernel entry only. If cluster dims specified at launch, must match exactly or launch fails.
- Cannot combine with `.maxclusterrank`.

### .explicitcluster
### Syntax
```ptx
.explicitcluster
```
### Constraints
- Kernel must be launched with cluster dims (either at launch or via `.reqnctapercluster`), else runtime error.

### .maxclusterrank
### Syntax
```ptx
.maxclusterrank n
```
### Constraints
- Product of cluster dims at launch must be <= `n`.
- Cannot combine with `.reqnctapercluster`.

### Example
```ptx
.entry foo .reqnctapercluster 2 { ... }
.entry bar .explicitcluster .maxclusterrank 8 { ... }
```

---

## Cluster Special Registers

| Register | Type | Description |
|----------|------|-------------|
| `%cluster_ctaid.{x,y,z}` | `.v4.u32` | CTA position within cluster |
| `%cluster_nctaid.{x,y,z}` | `.v4.u32` | Cluster shape (CTAs per dim) |
| `%cluster_ctarank` | `.u32` | Flat linear rank of CTA in cluster, `[0, %cluster_nctarank)` |
| `%cluster_nctarank` | `.u32` | Total CTAs in cluster |
| `%clusterid.{x,y,z}` | `.v4.u32` | Cluster position within grid |
| `%nclusterid.{x,y,z}` | `.v4.u32` | Number of clusters per grid dim |
| `%is_explicit_cluster` | `.pred` | True if cluster launch was explicit |

All require `sm_90`. Introduced PTX ISA 7.8.

---

## barrier.cluster

See also `ptx-isa-barriers.md` section 3.

### Syntax
```ptx
barrier.cluster.arrive{.sem}{.aligned};
barrier.cluster.wait{.acquire}{.aligned};

.sem = { .release, .relaxed }   // default: .release
```
### Constraints
- All non-exited cluster threads must arrive before wait completes.
- Auto-reinitializes on completion. Each thread arrives exactly once per phase.
- `.relaxed` on arrive removes memory ordering; use explicit `fence.cluster.acq_rel` if needed.
- `.aligned` -- all threads in warp must execute the instruction.

### Example
```ptx
ld.shared::cluster.u32 r0, [addr];
barrier.cluster.arrive.aligned;
// ... independent work ...
barrier.cluster.wait.aligned;
st.shared::cluster.u32 [addr], r1;
```

---

## Distributed Shared Memory (DSMEM)

CTAs within a cluster can access each other's shared memory via `.shared::cluster` state space.

### mapa -- Map Address to Peer CTA Shared Memory
### Syntax
```ptx
mapa.shared::cluster.size  dest, src_addr, target_ctarank;

.size = { .u32, .u64 }
```
### Constraints
- `src_addr` -- a `.shared` address (generic or explicit) in the current CTA.
- `target_ctarank` -- `%cluster_ctarank` of the target CTA (`.u32`).
- Returns `.shared::cluster` address at the same offset in the target CTA's shared memory.
- Requires `sm_90`. PTX ISA 7.8.

### getctarank -- Get CTA Rank from Shared Address
### Syntax
```ptx
getctarank.shared::cluster.u32  dest, src_addr;
```
### Constraints
- `src_addr` -- a `.shared::cluster` generic address.
- Returns the `%cluster_ctarank` of the CTA that owns that shared memory location.
- Requires `sm_90`. PTX ISA 7.8.

### Example
```ptx
cvta.shared.u64 addr, shMem;
mapa.shared::cluster.u64 remAddr, addr, 0;    // CTA0's shMem
getctarank.shared::cluster.u32 rank, remAddr;  // returns 0
```

---

## elect.sync -- Elect Leader Thread

### Syntax
```ptx
elect.sync  d|p, membermask;
```
### Constraints
- `membermask` (`.u32`) -- bit mask of participating lanes.
- `d` (`.u32`) -- laneid of elected leader (can use sink `_`).
- `p` (`.pred`) -- True for leader, False for others.
- Deterministic: same `membermask` always elects same leader.
- `.sync` -- all threads in `membermask` must execute before any resume.
- Requires `sm_90`. PTX ISA 8.0.

### Example
```ptx
elect.sync _|%p0, 0xffffffff;
@%p0 mbarrier.expect_tx.shared.b64 [mbar], 2048;
```

---

## cp.async.bulk.tensor (TMA)

See `ptx-isa-async-copy.md` for full syntax, load modes, and completion mechanisms.
Hopper-specific notes here.

### Multicast (sm_90a optimized)
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
    [dstMem], [tensorMap, {c0, c1}], [mbar], ctaMask;
```

### Constraints
- `ctaMask` -- 16-bit, each bit = `%cluster_ctarank` of a destination CTA.
- Data is copied to same CTA-relative offset in each destination CTA's shared memory.
- Mbarrier signal is also multicast to each destination CTA.
- `.multicast::cluster` is optimized on `sm_90a`; substantially reduced perf on plain `sm_90`.

### Load Modes (sm_90)

| Mode | Description |
|------|-------------|
| `.tile` | Preserves multi-dimensional tensor layout |
| `.im2col` | Unrolls spatial dims for convolution (3D+ tensors) |

---

## wgmma (Warpgroup MMA)

See `ptx-isa-tensor-cores.md` sections 3-4 for full shape/type tables, descriptor format, and lifecycle.

### Syntax
```ptx
wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
    d, {a-desc|a-regs}, b-desc, scale-d, imm-scale-a, imm-scale-b{, imm-trans-a, imm-trans-b};
```

### Lifecycle
```ptx
wgmma.fence.sync.aligned;                     // 1. Fence before first MMA / after reg writes
wgmma.mma_async.sync.aligned.m64n128k16...;   // 2. Issue MMA(s)
wgmma.commit_group.sync.aligned;              // 3. Commit into wgmma-group
wgmma.wait_group.sync.aligned N;              // 4. Wait (N=0 waits all)
```

### Constraints
- All 128 threads in the warpgroup must execute each instruction (`.sync.aligned`).
- Accessing accumulator registers before `wait_group` returns is undefined behavior.
- `wgmma.fence` required before first MMA and whenever registers are modified between MMAs.
- Requires `sm_90a`. PTX ISA 8.0.

---

## setmaxnreg -- Dynamic Register Reallocation

### Syntax
```ptx
setmaxnreg.action.sync.aligned.u32  imm-reg-count;

.action = { .inc, .dec }
```

### Constraints
- `imm-reg-count`: range **[24, 256]**, must be **multiple of 8**.
- `.inc` -- blocks until enough regs available in per-CTA pool. New regs have undefined contents.
- `.dec` -- releases regs. Current count must be >= `imm-reg-count`.
- All warps in the **warpgroup** must execute the same `setmaxnreg`.
- Must synchronize all warpgroup warps before issuing another `setmaxnreg`.
- Register changes happen at tail end of register file.
- Requires `sm_90a`. PTX ISA 8.0.

### Example
```ptx
// Producer warp: release registers
setmaxnreg.dec.sync.aligned.u32 40;

// Consumer warp: claim registers for large accumulator
setmaxnreg.inc.sync.aligned.u32 232;
```

---

## mbarrier Cluster-Scope Features (sm_90)

See `ptx-isa-barriers.md` sections 4-6 for full mbarrier reference.
Hopper additions:

### mbarrier.try_wait (sm_90)
```ptx
mbarrier.try_wait{.sem.scope}{.shared{::cta}}.b64  waitComplete, [addr], state{, suspendTimeHint};
mbarrier.try_wait.parity{.sem.scope}{.shared{::cta}}.b64  waitComplete, [addr], phaseParity{, suspendTimeHint};

.sem   = { .acquire, .relaxed }
.scope = { .cta, .cluster }
```
- Potentially blocking: thread may suspend until phase completes or timeout.
- `.relaxed` and `.cluster` scope require `sm_90`.

### mbarrier.arrive with .cluster scope
```ptx
mbarrier.arrive{.release}.cluster{.shared::cluster}.b64  _, [remAddr]{, count};
mbarrier.arrive.expect_tx{.release}.cluster{.shared::cluster}.b64  _, [remAddr], txCount;
```
- Remote arrive on mbarrier in another CTA's shared memory (via `mapa` address).
- Cannot return state when targeting `.shared::cluster` (use sink `_`).

### Example (cross-CTA synchronization)
```ptx
cvta.shared.u64 addr, shMem;
mapa.shared::cluster.u64 remAddr, addr, 0;                  // CTA0's mbarrier
@p0 mbarrier.init.shared::cta.b64 [shMem], N;              // CTA0 inits

barrier.cluster.arrive;
barrier.cluster.wait;

mbarrier.arrive.release.cluster.b64 _, [remAddr];           // all CTAs arrive

// CTA0 waits
waitLoop:
mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 complete, [shMem], 0;
@!complete bra waitLoop;
```

---

## Summary: sm_90 vs sm_90a Requirements

| Feature | Target |
|---------|--------|
| Clusters, `barrier.cluster`, DSMEM | `sm_90` |
| `cp.async.bulk.tensor` (TMA) base | `sm_90` |
| TMA `.multicast::cluster` (optimized) | `sm_90a` |
| `wgmma.*` (mma_async, fence, commit, wait) | `sm_90a` |
| `setmaxnreg` | `sm_90a` |
| `elect.sync` | `sm_90` |
| `mbarrier.try_wait` | `sm_90` |
| Cluster special registers | `sm_90` |
