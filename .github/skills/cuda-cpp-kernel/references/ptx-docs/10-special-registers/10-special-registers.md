# 10. Special Registers


PTX includes a number of predefined, read-only variables, which are visible as special registers and accessed through `mov` or `cvt` instructions.


The special registers are:


  * `%tid`

  * `%ntid`

  * `%laneid`

  * `%warpid`

  * `%nwarpid`

  * `%ctaid`

  * `%nctaid`

  * `%smid`

  * `%nsmid`

  * `%gridid`

  * `%is_explicit_cluster`

  * `%clusterid`

  * `%nclusterid`

  * `%cluster_ctaid`

  * `%cluster_nctaid`

  * `%cluster_ctarank`

  * `%cluster_nctarank`

  * `%lanemask_eq`, `%lanemask_le`, `%lanemask_lt`, `%lanemask_ge`, `%lanemask_gt`

  * `%clock`, `%clock_hi`, `%clock64`

  * `%pm0, ..., %pm7`

  * `%pm0_64, ..., %pm7_64`

  * `%envreg0, ..., %envreg31`

  * `%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`

  * `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset<2>`

  * `%total_smem_size`

  * `%aggr_smem_size`

  * `%dynamic_smem_size`

  * `%current_graph_exec`


##  10.1. [Special Registers: `%tid`](#special-registers-tid)

`%tid`

Thread identifier within a CTA.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %tid;                  // thread id vector
    .sreg .u32 %tid.x, %tid.y, %tid.z;    // thread id components
    

Description

A predefined, read-only, per-thread special register initialized with the thread identifier within the CTA. The `%tid` special register contains a 1D, 2D, or 3D vector to match the CTA shape; the `%tid` value in unused dimensions is `0`. The fourth element is unused and always returns zero. The number of threads in each dimension are specified by the predefined special register `%ntid`.

Every thread in the CTA has a unique `%tid`.

`%tid` component values range from `0` through `%ntid-1` in each CTA dimension.

`%tid.y == %tid.z == 0` in 1D CTAs. `%tid.z == 0` in 2D CTAs.

It is guaranteed that:
    
    
    0  <=  %tid.x <  %ntid.x
    0  <=  %tid.y <  %ntid.y
    0  <=  %tid.z <  %ntid.z
    

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit `mov` and `cvt` instructions may be used to read the lower 16-bits of each component of `%tid`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32      %r1,%tid.x;  // move tid.x to %rh
    
    // legacy code accessing 16-bit components of %tid
    mov.u16      %rh,%tid.x;
    cvt.u32.u16  %r2,%tid.z;  // zero-extend tid.z to %r2
    


##  10.2. [Special Registers: `%ntid`](#special-registers-ntid)

`%ntid`

Number of thread IDs per CTA.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %ntid;                   // CTA shape vector
    .sreg .u32 %ntid.x, %ntid.y, %ntid.z;   // CTA dimensions
    

Description

A predefined, read-only special register initialized with the number of thread ids in each CTA dimension. The `%ntid` special register contains a 3D CTA shape vector that holds the CTA dimensions. CTA dimensions are non-zero; the fourth element is unused and always returns zero. The total number of threads in a CTA is `(%ntid.x * %ntid.y * %ntid.z)`.
    
    
    %ntid.y == %ntid.z == 1 in 1D CTAs.
    %ntid.z ==1 in 2D CTAs.
    

Maximum values of %ntid.{x,y,z} are as follows:

.target architecture | %ntid.x | %ntid.y | %ntid.z  
---|---|---|---  
`sm_1x` | 512 | 512 | 64  
`sm_20`, `sm_3x`, `sm_5x`, `sm_6x`, `sm_7x`, `sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 1024 | 1024 | 64  
  
PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit `mov` and `cvt` instructions may be used to read the lower 16-bits of each component of `%ntid`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    // compute unified thread id for 2D CTA
    mov.u32  %r0,%tid.x;
    mov.u32  %h1,%tid.y;
    mov.u32  %h2,%ntid.x;
    mad.u32  %r0,%h1,%h2,%r0;
    
    mov.u16  %rh,%ntid.x;      // legacy code
    


##  10.3. [Special Registers: `%laneid`](#special-registers-laneid)

`%laneid`

Lane Identifier.

Syntax (predefined)
    
    
    .sreg .u32 %laneid;
    

Description

A predefined, read-only special register that returns the thread’s lane within the warp. The lane identifier ranges from zero to `WARP_SZ-1`.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32  %r, %laneid;
    


##  10.4. [Special Registers: `%warpid`](#special-registers-warpid)

`%warpid`

Warp identifier.

Syntax (predefined)
    
    
    .sreg .u32 %warpid;
    

Description

A predefined, read-only special register that returns the thread’s warp identifier. The warp identifier provides a unique warp number within a CTA but not across CTAs within a grid. The warp identifier will be the same for all threads within a single warp.

Note that `%warpid` returns the location of a thread at the moment when read, but its value may change during execution, e.g., due to rescheduling of threads following preemption. For this reason, `%ctaid` and `%tid` should be used to compute a virtual warp index if such a value is needed in kernel code; `%warpid` is intended mainly to enable profiling and diagnostic code to sample and log information such as work place mapping and load distribution.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32  %r, %warpid;
    


##  10.5. [Special Registers: `%nwarpid`](#special-registers-nwarpid)

`%nwarpid`

Number of warp identifiers.

Syntax (predefined)
    
    
    .sreg .u32 %nwarpid;
    

Description

A predefined, read-only special register that returns the maximum number of warp identifiers.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%nwarpid` requires `sm_20` or higher.

Examples
    
    
    mov.u32  %r, %nwarpid;
    


##  10.6. [Special Registers: `%ctaid`](#special-registers-ctaid)

`%ctaid`

CTA identifier within a grid.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %ctaid;                      // CTA id vector
    .sreg .u32 %ctaid.x, %ctaid.y, %ctaid.z;    // CTA id components
    

Description

A predefined, read-only special register initialized with the CTA identifier within the CTA grid. The `%ctaid` special register contains a 1D, 2D, or 3D vector, depending on the shape and rank of the CTA grid. The fourth element is unused and always returns zero.

It is guaranteed that:
    
    
    0  <=  %ctaid.x <  %nctaid.x
    0  <=  %ctaid.y <  %nctaid.y
    0  <=  %ctaid.z <  %nctaid.z
    

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit `mov` and `cvt` instructions may be used to read the lower 16-bits of each component of `%ctaid`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32  %r0,%ctaid.x;
    mov.u16  %rh,%ctaid.y;   // legacy code
    


##  10.7. [Special Registers: `%nctaid`](#special-registers-nctaid)

`%nctaid`

Number of CTA ids per grid.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %nctaid                      // Grid shape vector
    .sreg .u32 %nctaid.x,%nctaid.y,%nctaid.z;   // Grid dimensions
    

Description

A predefined, read-only special register initialized with the number of CTAs in each grid dimension. The `%nctaid` special register contains a 3D grid shape vector, with each element having a value of at least `1`. The fourth element is unused and always returns zero.

Maximum values of %nctaid.{x,y,z} are as follows:

.target architecture | %nctaid.x | %nctaid.y | %nctaid.z  
---|---|---|---  
`sm_1x`, `sm_20` | 65535 | 65535 | 65535  
`sm_3x`, `sm_5x`, `sm_6x`, `sm_7x`, `sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 231 -1 | 65535 | 65535  
  
PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit `mov` and `cvt` instructions may be used to read the lower 16-bits of each component of `%nctaid`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32  %r0,%nctaid.x;
    mov.u16  %rh,%nctaid.x;     // legacy code
    


##  10.8. [Special Registers: `%smid`](#special-registers-smid)

`%smid`

SM identifier.

Syntax (predefined)
    
    
    .sreg .u32 %smid;
    

Description

A predefined, read-only special register that returns the processor (SM) identifier on which a particular thread is executing. The SM identifier ranges from `0` to `%nsmid-1`. The SM identifier numbering is not guaranteed to be contiguous.

Notes

Note that `%smid` returns the location of a thread at the moment when read, but its value may change during execution, e.g. due to rescheduling of threads following preemption. `%smid` is intended mainly to enable profiling and diagnostic code to sample and log information such as work place mapping and load distribution.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u32  %r, %smid;
    


##  10.9. [Special Registers: `%nsmid`](#special-registers-nsmid)

`%nsmid`

Number of SM identifiers.

Syntax (predefined)
    
    
    .sreg .u32 %nsmid;
    

Description

A predefined, read-only special register that returns the maximum number of SM identifiers. The SM identifier numbering is not guaranteed to be contiguous, so `%nsmid` may be larger than the physical number of SMs in the device.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%nsmid` requires `sm_20` or higher.

Examples
    
    
    mov.u32  %r, %nsmid;
    


##  10.10. [Special Registers: `%gridid`](#special-registers-gridid)

`%gridid`

Grid identifier.

Syntax (predefined)
    
    
    .sreg .u64 %gridid;
    

Description

A predefined, read-only special register initialized with the per-grid temporal grid identifier. The `%gridid` is used by debuggers to distinguish CTAs and clusters within concurrent (small) grids.

During execution, repeated launches of programs may occur, where each launch starts a grid-of-CTAs. This variable provides the temporal grid launch number for this context.

For `sm_1x` targets, `%gridid` is limited to the range [0..216-1]. For `sm_20`, `%gridid` is limited to the range [0..232-1]. `sm_30` supports the entire 64-bit range.

PTX ISA Notes

Introduced in PTX ISA version 1.0 as type `.u16`.

Redefined as type `.u32` in PTX ISA version 1.3.

Redefined as type `.u64` in PTX ISA version 3.0.

For compatibility with legacy PTX code, 16-bit and 32-bit `mov` and `cvt` instructions may be used to read the lower 16-bits or 32-bits of each component of `%gridid`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.u64  %s, %gridid;  // 64-bit read of %gridid
    mov.u32  %r, %gridid;  // legacy code with 32-bit %gridid
    


##  10.11. [Special Registers: `%is_explicit_cluster`](#special-registers-is-explicit-cluster)

`%is_explicit_cluster`

Checks if user has explicitly specified cluster launch.

Syntax (predefined)
    
    
    .sreg .pred %is_explicit_cluster;
    

Description

A predefined, read-only special register initialized with the predicate value of whether the cluster launch is explicitly specified by user.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .pred p;
    
    mov.pred  p, %is_explicit_cluster;
    


##  10.12. [Special Registers: `%clusterid`](#special-registers-clusterid)

`%clusterid`

Cluster identifier within a grid.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %clusterid;
    .sreg .u32 %clusterid.x, %clusterid.y, %clusterid.z;
    

Description

A predefined, read-only special register initialized with the cluster identifier in a grid in each dimension. Each cluster in a grid has a unique identifier.

The `%clusterid` special register contains a 1D, 2D, or 3D vector, depending upon the shape and rank of the cluster. The fourth element is unused and always returns zero.

It is guaranteed that:
    
    
    0  <=  %clusterid.x <  %nclusterid.x
    0  <=  %clusterid.y <  %nclusterid.y
    0  <=  %clusterid.z <  %nclusterid.z
    

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r<2>;
    .reg .v4 .b32 %rx;
    
    mov.u32     %r0, %clusterid.x;
    mov.u32     %r1, %clusterid.z;
    mov.v4.u32  %rx, %clusterid;
    


##  10.13. [Special Registers: `%nclusterid`](#special-registers-nclusterid)

`%nclusterid`

Number of cluster identifiers per grid.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %nclusterid;
    .sreg .u32 %nclusterid.x, %nclusterid.y, %nclusterid.z;
    

Description

A predefined, read-only special register initialized with the number of clusters in each grid dimension.

The `%nclusterid` special register contains a 3D grid shape vector that holds the grid dimensions in terms of clusters. The fourth element is unused and always returns zero.

Refer to the _Cuda Programming Guide_ for details on the maximum values of `%nclusterid.{x,y,z}`.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r<2>;
    .reg .v4 .b32 %rx;
    
    mov.u32     %r0, %nclusterid.x;
    mov.u32     %r1, %nclusterid.z;
    mov.v4.u32  %rx, %nclusterid;
    


##  10.14. [Special Registers: `%cluster_ctaid`](#special-registers-cluster-ctaid)

`%cluster_ctaid`

CTA identifier within a cluster.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %cluster_ctaid;
    .sreg .u32 %cluster_ctaid.x, %cluster_ctaid.y, %cluster_ctaid.z;
    

Description

A predefined, read-only special register initialized with the CTA identifier in a cluster in each dimension. Each CTA in a cluster has a unique CTA identifier.

The `%cluster_ctaid` special register contains a 1D, 2D, or 3D vector, depending upon the shape of the cluster. The fourth element is unused and always returns zero.

It is guaranteed that:
    
    
    0  <=  %cluster_ctaid.x <  %cluster_nctaid.x
    0  <=  %cluster_ctaid.y <  %cluster_nctaid.y
    0  <=  %cluster_ctaid.z <  %cluster_nctaid.z
    

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r<2>;
    .reg .v4 .b32 %rx;
    
    mov.u32     %r0, %cluster_ctaid.x;
    mov.u32     %r1, %cluster_ctaid.z;
    mov.v4.u32  %rx, %cluster_ctaid;
    


##  10.15. [Special Registers: `%cluster_nctaid`](#special-registers-cluster-nctaid)

`%cluster_nctaid`

Number of CTA identifiers per cluster.

Syntax (predefined)
    
    
    .sreg .v4 .u32 %cluster_nctaid;
    .sreg .u32 %cluster_nctaid.x, %cluster_nctaid.y, %cluster_nctaid.z;
    

Description

A predefined, read-only special register initialized with the number of CTAs in a cluster in each dimension.

The `%cluster_nctaid` special register contains a 3D grid shape vector that holds the cluster dimensions in terms of CTAs. The fourth element is unused and always returns zero.

Refer to the _Cuda Programming Guide_ for details on the maximum values of `%cluster_nctaid.{x,y,z}`.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r<2>;
    .reg .v4 .b32 %rx;
    
    mov.u32     %r0, %cluster_nctaid.x;
    mov.u32     %r1, %cluster_nctaid.z;
    mov.v4.u32  %rx, %cluster_nctaid;
    


##  10.16. [Special Registers: `%cluster_ctarank`](#special-registers-cluster-ctarank)

`%cluster_ctarank`

CTA identifier in a cluster across all dimensions.

Syntax (predefined)
    
    
    .sreg .u32 %cluster_ctarank;
    

Description

A predefined, read-only special register initialized with the CTA rank within a cluster across all dimensions.

It is guaranteed that:
    
    
    0  <=  %cluster_ctarank <  %cluster_nctarank
    

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r;
    
    mov.u32  %r, %cluster_ctarank;
    


##  10.17. [Special Registers: `%cluster_nctarank`](#special-registers-cluster-nctarank)

`%cluster_nctarank`

Number of CTA identifiers in a cluster across all dimensions.

Syntax (predefined)
    
    
    .sreg .u32 %cluster_nctarank;
    

Description

A predefined, read-only special register initialized with the nunber of CTAs within a cluster across all dimensions.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .reg .b32 %r;
    
    mov.u32  %r, %cluster_nctarank;
    


##  10.18. [Special Registers: `%lanemask_eq`](#special-registers-lanemask-eq)

`%lanemask_eq`

32-bit mask with bit set in position equal to the thread’s lane number in the warp.

Syntax (predefined)
    
    
    .sreg .u32 %lanemask_eq;
    

Description

A predefined, read-only special register initialized with a 32-bit mask with a bit set in the position equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_eq` requires `sm_20` or higher.

Examples
    
    
    mov.u32     %r, %lanemask_eq;
    


##  10.19. [Special Registers: `%lanemask_le`](#special-registers-lanemask-le)

`%lanemask_le`

32-bit mask with bits set in positions less than or equal to the thread’s lane number in the warp.

Syntax (predefined)
    
    
    .sreg .u32 %lanemask_le;
    

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions less than or equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_le` requires `sm_20` or higher.

Examples
    
    
    mov.u32     %r, %lanemask_le
    


##  10.20. [Special Registers: `%lanemask_lt`](#special-registers-lanemask-lt)

`%lanemask_lt`

32-bit mask with bits set in positions less than the thread’s lane number in the warp.

Syntax (predefined)
    
    
    .sreg .u32 %lanemask_lt;
    

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions less than the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_lt` requires `sm_20` or higher.

Examples
    
    
    mov.u32     %r, %lanemask_lt;
    


##  10.21. [Special Registers: `%lanemask_ge`](#special-registers-lanemask-ge)

`%lanemask_ge`

32-bit mask with bits set in positions greater than or equal to the thread’s lane number in the warp.

Syntax (predefined)
    
    
    .sreg .u32 %lanemask_ge;
    

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions greater than or equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_ge` requires `sm_20` or higher.

Examples
    
    
    mov.u32     %r, %lanemask_ge;
    


##  10.22. [Special Registers: `%lanemask_gt`](#special-registers-lanemask-gt)

`%lanemask_gt`

32-bit mask with bits set in positions greater than the thread’s lane number in the warp.

Syntax (predefined)
    
    
    .sreg .u32 %lanemask_gt;
    

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions greater than the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_gt` requires `sm_20` or higher.

Examples
    
    
    mov.u32     %r, %lanemask_gt;
    


##  10.23. [Special Registers: `%clock`, `%clock_hi`](#special-registers-clock)

`%clock`, `%clock_hi`

`%clock`
    

A predefined, read-only 32-bit unsigned cycle counter.

`%clock_hi`
    

The upper 32-bits of `%clock64` special register.

Syntax (predefined)
    
    
    .sreg .u32 %clock;
    .sreg .u32 %clock_hi;
    

Description

Special register `%clock` and `%clock_hi` are unsigned 32-bit read-only cycle counters that wrap silently.

PTX ISA Notes

`%clock` introduced in PTX ISA version 1.0.

`%clock_hi` introduced in PTX ISA version 5.0.

Target ISA Notes

`%clock` supported on all target architectures.

`%clock_hi` requires `sm_20` or higher.

Examples
    
    
    mov.u32 r1,%clock;
    mov.u32 r2, %clock_hi;
    


##  10.24. [Special Registers: `%clock64`](#special-registers-clock64)

`%clock64`

A predefined, read-only 64-bit unsigned cycle counter.

Syntax (predefined)
    
    
    .sreg .u64 %clock64;
    

Description

Special register `%clock64` is an unsigned 64-bit read-only cycle counter that wraps silently.

Notes

The lower 32-bits of `%clock64` are identical to `%clock`.

The upper 32-bits of `%clock64` are identical to `%clock_hi`.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%clock64` requires `sm_20` or higher.

Examples
    
    
    mov.u64  r1,%clock64;
    


##  10.25. [Special Registers: `%pm0` … `%pm7`](#special-registers-pm0-pm7)

`%pm0` … `%pm7`

Performance monitoring counters.

Syntax (predefined)
    
    
    .sreg .u32 %pm<8>;
    

Description

Special registers `%pm0` … `%pm7` are unsigned 32-bit read-only performance monitor counters. Their behavior is currently undefined.

PTX ISA Notes

`%pm0` … `%pm3` introduced in PTX ISA version 1.3.

`%pm4` … `%pm7` introduced in PTX ISA version 3.0.

Target ISA Notes

`%pm0` … `%pm3` supported on all target architectures.

`%pm4` … `%pm7` require `sm_20` or higher.

Examples
    
    
    mov.u32  r1,%pm0;
    mov.u32  r1,%pm7;
    


##  10.26. [Special Registers: `%pm0_64` … `%pm7_64`](#special-registers-pm0-64-pm7-64)

`%pm0_64` … `%pm7_64`

64 bit Performance monitoring counters.

Syntax (predefined)
    
    
    .sreg .u64 %pm0_64;
    .sreg .u64 %pm1_64;
    .sreg .u64 %pm2_64;
    .sreg .u64 %pm3_64;
    .sreg .u64 %pm4_64;
    .sreg .u64 %pm5_64;
    .sreg .u64 %pm6_64;
    .sreg .u64 %pm7_64;
    

Description

Special registers `%pm0_64` … `%pm7_64` are unsigned 64-bit read-only performance monitor counters. Their behavior is currently undefined.

Notes

The lower 32bits of `%pm0_64` … `%pm7_64` are identical to `%pm0` … `%pm7`.

PTX ISA Notes

`%pm0_64` … `%pm7_64` introduced in PTX ISA version 4.0.

Target ISA Notes

`%pm0_64` … `%pm7_64` require `sm_50` or higher.

Examples
    
    
    mov.u32  r1,%pm0_64;
    mov.u32  r1,%pm7_64;
    


##  10.27. [Special Registers: `%envreg<32>`](#special-registers-envreg-32)

`%envreg<32>`

Driver-defined read-only registers.

Syntax (predefined)
    
    
    .sreg .b32 %envreg<32>;
    

Description

A set of 32 pre-defined read-only registers used to capture execution environment of PTX program outside of PTX virtual machine. These registers are initialized by the driver prior to kernel launch and can contain cta-wide or grid-wide values.

Precise semantics of these registers is defined in the driver documentation.

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    mov.b32      %r1,%envreg0;  // move envreg0 to %r1
    


##  10.28. [Special Registers: `%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`](#special-registers-globaltimer)

`%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`

`%globaltimer`
    

A predefined, 64-bit global nanosecond timer.

`%globaltimer_lo`
    

The lower 32-bits of %globaltimer.

`%globaltimer_hi`
    

The upper 32-bits of %globaltimer.

Syntax (predefined)
    
    
    .sreg .u64 %globaltimer;
    .sreg .u32 %globaltimer_lo, %globaltimer_hi;
    

Description

Special registers intended for use by NVIDIA tools. The behavior is target-specific and may change or be removed in future GPUs. When JIT-compiled to other targets, the value of these registers is unspecified.

PTX ISA Notes

Introduced in PTX ISA version 3.1.

Target ISA Notes

Requires target `sm_30` or higher.

Examples
    
    
    mov.u64  r1,%globaltimer;
    


##  10.29. [Special Registers: `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset_<2>`](#special-registers-reserved-smem)

`%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset_<2>`

`%reserved_smem_offset_begin`
    

Start of the reserved shared memory region.

`%reserved_smem_offset_end`
    

End of the reserved shared memory region.

`%reserved_smem_offset_cap`
    

Total size of the reserved shared memory region.

`%reserved_smem_offset_<2>`
    

Offsets in the reserved shared memory region.

Syntax (predefined)
    
    
    .sreg .b32 %reserved_smem_offset_begin;
    .sreg .b32 %reserved_smem_offset_end;
    .sreg .b32 %reserved_smem_offset_cap;
    .sreg .b32 %reserved_smem_offset_<2>;
    

Description

These are predefined, read-only special registers containing information about the shared memory region which is reserved for the NVIDIA system software use. This region of shared memory is not available to users, and accessing this region from user code results in undefined behavior. Refer to _CUDA Programming Guide_ for details.

PTX ISA Notes

Introduced in PTX ISA version 7.6.

Target ISA Notes

Require `sm_80` or higher.

Examples
    
    
    .reg .b32 %reg_begin, %reg_end, %reg_cap, %reg_offset0, %reg_offset1;
    
    mov.b32 %reg_begin,   %reserved_smem_offset_begin;
    mov.b32 %reg_end,     %reserved_smem_offset_end;
    mov.b32 %reg_cap,     %reserved_smem_offset_cap;
    mov.b32 %reg_offset0, %reserved_smem_offset_0;
    mov.b32 %reg_offset1, %reserved_smem_offset_1;
    


##  10.30. [Special Registers: `%total_smem_size`](#special-registers-total-smem-size)

`%total_smem_size`

Total size of shared memory used by a CTA of a kernel.

Syntax (predefined)
    
    
    .sreg .u32 %total_smem_size;
    

Description

A predefined, read-only special register initialized with total size of shared memory allocated (statically and dynamically, excluding the shared memory reserved for the NVIDIA system software use) for the CTA of a kernel at launch time.

Size is returned in multiples of shared memory allocation unit size supported by target architecture.

Allocation unit values are as follows:

Target architecture | Shared memory allocation unit size  
---|---  
`sm_2x` | 128 bytes  
`sm_3x`, `sm_5x`, `sm_6x`, `sm_7x` | 256 bytes  
`sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 128 bytes  
  
PTX ISA Notes

Introduced in PTX ISA version 4.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples
    
    
    mov.u32  %r, %total_smem_size;
    


##  10.31. [Special Registers: `%aggr_smem_size`](#special-registers-aggr-smem-size)

`%aggr_smem_size`

Total size of shared memory used by a CTA of a kernel.

Syntax (predefined)
    
    
    .sreg .u32 %aggr_smem_size;
    

Description

A predefined, read-only special register initialized with total aggregated size of shared memory consisting of the size of user shared memory allocated (statically and dynamically) at launch time and the size of shared memory region which is reserved for the NVIDIA system software use.

PTX ISA Notes

Introduced in PTX ISA version 8.1.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    mov.u32  %r, %aggr_smem_size;
    


##  10.32. [Special Registers: `%dynamic_smem_size`](#special-registers-dynamic-smem-size)

`%dynamic_smem_size`

Size of shared memory allocated dynamically at kernel launch.

Syntax (predefined)
    
    
    .sreg .u32 %dynamic_smem_size;
    

Description

Size of shared memory allocated dynamically at kernel launch.

A predefined, read-only special register initialized with size of shared memory allocated dynamically for the CTA of a kernel at launch time.

PTX ISA Notes

Introduced in PTX ISA version 4.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples
    
    
    mov.u32  %r, %dynamic_smem_size;
    


##  10.33. [Special Registers: `%current_graph_exec`](#special-registers-current-graph-exec)

`%current_graph_exec`

An Identifier for currently executing CUDA device graph.

Syntax (predefined)
    
    
    .sreg .u64 %current_graph_exec;
    

Description

A predefined, read-only special register initialized with the identifier referring to the CUDA device graph being currently executed. This register is 0 if the executing kernel is not part of a CUDA device graph.

Refer to the _CUDA Programming Guide_ for more details on CUDA device graphs.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_50` or higher.

Examples
    
    
    mov.u64  r1, %current_graph_exec;
    
