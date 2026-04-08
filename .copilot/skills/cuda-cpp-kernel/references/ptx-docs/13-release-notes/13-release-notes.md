# 13. Release Notes


This section describes the history of change in the PTX ISA and implementation. The first section describes ISA and implementation changes in the current release of PTX ISA version 9.2, and the remaining sections provide a record of changes in previous releases of PTX ISA versions back to PTX ISA version 2.0.


[Table 59](#release-notes-ptx-release-history) shows the PTX release history.


Table 59 PTX Release History PTX ISA Version | CUDA Release | Supported Targets  
---|---|---  
PTX ISA 1.0 | CUDA 1.0 | `sm_{10,11}`  
PTX ISA 1.1 | CUDA 1.1 | `sm_{10,11}`  
PTX ISA 1.2 | CUDA 2.0 | `sm_{10,11,12,13}`  
PTX ISA 1.3 | CUDA 2.1 | `sm_{10,11,12,13}`  
PTX ISA 1.4 | CUDA 2.2 | `sm_{10,11,12,13}`  
PTX ISA 1.5 | driver r190 | `sm_{10,11,12,13}`  
PTX ISA 2.0 | CUDA 3.0, driver r195 | `sm_{10,11,12,13}`, `sm_20`  
PTX ISA 2.1 | CUDA 3.1, driver r256 | `sm_{10,11,12,13}`, `sm_20`  
PTX ISA 2.2 | CUDA 3.2, driver r260 | `sm_{10,11,12,13}`, `sm_20`  
PTX ISA 2.3 | CUDA 4.0, driver r270 | `sm_{10,11,12,13}`, `sm_20`  
PTX ISA 3.0 | CUDA 4.1, driver r285 | `sm_{10,11,12,13}`, `sm_20`  
CUDA 4.2, driver r295 | `sm_{10,11,12,13}`, `sm_20`, `sm_30`  
PTX ISA 3.1 | CUDA 5.0, driver r302 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,35}`  
PTX ISA 3.2 | CUDA 5.5, driver r319 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,35}`  
PTX ISA 4.0 | CUDA 6.0, driver r331 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35}`, `sm_50`  
PTX ISA 4.1 | CUDA 6.5, driver r340 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52}`  
PTX ISA 4.2 | CUDA 7.0, driver r346 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`  
PTX ISA 4.3 | CUDA 7.5, driver r352 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`  
PTX ISA 5.0 | CUDA 8.0, driver r361 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`  
PTX ISA 6.0 | CUDA 9.0, driver r384 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`  
PTX ISA 6.1 | CUDA 9.1, driver r387 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`, `sm_72`  
PTX ISA 6.2 | CUDA 9.2, driver r396 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`, `sm_72`  
PTX ISA 6.3 | CUDA 10.0, driver r400 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`, `sm_72`, `sm_75`  
PTX ISA 6.4 | CUDA 10.1, driver r418 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`, `sm_72`, `sm_75`  
PTX ISA 6.5 | CUDA 10.2, driver r440 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_70`, `sm_72`, `sm_75`  
PTX ISA 7.0 | CUDA 11.0, driver r445 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_80`  
PTX ISA 7.1 | CUDA 11.1, driver r455 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86}`  
PTX ISA 7.2 | CUDA 11.2, driver r460 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86}`  
PTX ISA 7.3 | CUDA 11.3, driver r465 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86}`  
PTX ISA 7.4 | CUDA 11.4, driver r470 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87}`  
PTX ISA 7.5 | CUDA 11.5, driver r495 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87}`  
PTX ISA 7.6 | CUDA 11.6, driver r510 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87}`  
PTX ISA 7.7 | CUDA 11.7, driver r515 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87}`  
PTX ISA 7.8 | CUDA 11.8, driver r520 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_90`  
PTX ISA 8.0 | CUDA 12.0, driver r525 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
PTX ISA 8.1 | CUDA 12.1, driver r530 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
PTX ISA 8.2 | CUDA 12.2, driver r535 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
PTX ISA 8.3 | CUDA 12.3, driver r545 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
PTX ISA 8.4 | CUDA 12.4, driver r550 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
PTX ISA 8.5 | CUDA 12.5, driver r555 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`  
CUDA 12.6, driver r560  
PTX ISA 8.6 | CUDA 12.7, driver r565 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`, `sm_{100,100a,101,101a}`  
PTX ISA 8.7 | CUDA 12.8, driver r570 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`, `sm_{100,100,101,101a}`, `sm_{120,120a}`  
PTX ISA 8.8 | CUDA 12.9, driver r575 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,89}`, `sm_{90,90a}`, `sm_{100,100f,100a,101,101f,101a,103,103f,103a}`, `sm_{120,120f,120a,121,121f,121a}`  
PTX ISA 9.0 | CUDA 13.0, driver r580 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,88,89}`, `sm_{90,90a}`, `sm_{100,100f,100a,103,103f,103a}`, `sm_{110,110f,110a}`, `sm_{120,120f,120a,121,121f,121a}`  
PTX ISA 9.1 | CUDA 13.1, driver r590 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,88,89}`, `sm_{90,90a}`, `sm_{100,100f,100a,103,103f,103a}`, `sm_{110,110f,110a}`, `sm_{120,120f,120a,121,121f,121a}`  
PTX ISA 9.2 | CUDA 13.2, driver r595 | `sm_{10,11,12,13}`, `sm_20`, `sm_{30,32,35,37}`, `sm_{50,52,53}`, `sm_{60,61,62}`, `sm_{70,72,75}`, `sm_{80,86,87,88,89}`, `sm_{90,90a}`, `sm_{100,100f,100a,103,103f,103a}`, `sm_{110,110f,110a}`, `sm_{120,120f,120a,121,121f,121a}`


[Table 60](#release-notes-a-spec-f-spec-ptx-feature-release-history) shows the release history of arch-specific and family-specific PTX instructions. Apart from PTX instructions, other features and constructs that are architecture-specific and family-specific are described in following sections:


  * [Restriction on Tensor Copy instructions](#data-movement-and-conversion-instructions-tensor-copy-restrictions)  
  
  * [TensorCore 5th Generation Matrix Shape Target ISA Notes](#tcgen05-matrix-shape-target-isa-note)


Table 60 Arch-specific/ Family-specific PTX Features Release History Instruction | Variant | PTX ISA Version | Supported Targets  
---|---|---|---  
`tensormap.replace` | Base variant | 8.3 | `sm_90a`  
8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`tensormap.replace.swizzle_atomicity` | 8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`.elemtype` for `.field3` with values `13`, `14`, `15` for `new_val` | 8.7 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`.swizzle_mode` for `.field3` with value `4` for `new_val` | 8.8 | `sm_103a`  
`wgmma.mma_async`, `wgmma.mma_async.sp`, `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group` | Base variant | 8.0 | `sm_90a`  
`setmaxnreg` | Base variant | 8.0 | `sm_90a`  
8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`multimem.ld_reduce`, `multimem.st`, `multimem.red` | Types `.e5m2`, `.e4m3`, `.e5m2x2`, `.e4m3x2`, `.e4m3x4`, `.e5m2x4` | 8.6 | `sm_100a`, `sm_120a`, `sm_121a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`.acc::f16` qualifier | 8.6 | `sm_100a`, `sm_120a`, `sm_121a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`cvt` | 

  * `.f32` to `.e2m1x2`/`.e2m3x2`/ `.e3m2x2`/`.ue8m0x2`
  * `.e2m1x2`/`.e2m3x2`/`.e3m2x2` to `.f16x2`
  * `.ue8m0x2` to `.bf16x2`
  * `.bf16x2` to `.ue8m0x2`

| 8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`.rs` rounding mode | 8.7 | `sm_100a`  
8.8 | `sm_103a`  
`.s2f6x2` type | 9.1 | `sm_100a`, `sm_103a`, `sm_110a`, `sm_120a`, `sm_121a`  
`.f16x2` to `.e2m1x2`/ `.e2m3x2`/`.e3m2x2` | 9.1 | `sm_100f`, `sm_110f`, `sm_120f`  
`.bf16x2` to `.e2m1x2`/ `.e2m3x2`/`.e3m2x2`/`.e4m3x2`/ `.e5m2x2` | 9.1 | `sm_100f`, `sm_110f`, `sm_120f`  
`.e2m1x2`/`.e2m3x2`/ `.e3m2x2`/`.e4m3x2`/`.e5m2x2` to `.bf16x2` | 9.2 | `sm_100f`, `sm_110f`, `sm_120f`  
`cp.async.bulk.tensor` | `.tile::gather4` and `.im2col::w` with `.shared::cluster` as destination state space | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`.tile::scatter4` and `.im2col::w::128` | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`.cta_group` | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`cp.async.bulk.prefetch.tensor` | `.tile::gather4`, `.im2col::w`, `.im2col::w::128` | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`redux.sync` | Type `.f32` and `.abs`, `.NaN` qualifiers | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
`clusterlaunchcontrol.try_cancel` | `.multicast::cluster::all` | 8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`ldmatrix` | 

  * Shapes `.m16n16`, `.m8n16`
  * Type `.b8`
  * Qualifiers `.src_fmt`, `.dst_fmt`

| 8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`stmatrix` | 

  * Shapes `.m16n8`
  * Type `.b8`

| 8.6 | `sm_100a`, `sm_120a`  
8.8 | `sm_100f`, `sm_120f`  
9.0 | `sm_110f`  
`tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`tcgen05.ld`, `tcgen05.st`, `tcgen05.wait`, `tcgen05.cp`, `tcgen05.fence`, `tcgen05.commit` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
`tcgen05.ld.red` | Base variant | 8.8 | `sm_103f`  
9.0 | `sm_110f`  
`tcgen05.shift` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_103a`  
9.0 | `sm_110a`  
`tcgen05.mma` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
Kind `.kind::i8` | 8.6 | `sm_100a`  
9.0 | `sm_110a`  
Argument `scale-input-d` | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
Qualifiers `.scale_vec::1X`, `.scale_vec::2X`, `.scale_vec::4X` | 8.6 | `sm_100a`  
Qualifiers `.block16`, `.block32` | 8.8 | `sm_100f`, `sm_110f`  
K shape value `96` | 8.8 | `sm_103a`  
`tcgen05.mma.sp` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
Kind `.kind::i8` | 8.6 | `sm_100a`  
9.0 | `sm_110a`  
Kind `.kind::mxf4nvf4` and `.kind::mxf4` | 8.6 | `sm_100a`  
8.8 | `sm_103a`  
9.0 | `sm_110a`  
Argument `scale-input-d` | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
Qualifiers `.scale_vec::1X`, `.scale_vec::2X`, `.scale_vec::4X` | 8.6 | `sm_100a`  
Qualifiers `.block16`, `.block32` | 8.8 | `sm_100f`, `sm_110f`  
`tcgen05.mma.ws`, `tcgen05.mma.ws.sp` | Base variant | 8.6 | `sm_100a`  
8.8 | `sm_100f`  
9.0 | `sm_110f`  
Kind `.kind::i8` | 8.6 | `sm_100a`  
9.0 | `sm_110a`  
`mma` | 

  * Types `.e3m2`, `.e2m3`, `.e2m1`
  * Qualifiers `.kind`, `.block_scale`, `.scale_vec_size`

| 8.7 | `sm_120a`  
8.8 | `sm_120f`  
`mma.sp` | 

  * Types `.e3m2`, `.e2m3`, `.e2m1`
  * Qualifiers `.kind`, `.block_scale`, `.scale_vec_size`

| 8.7 | `sm_120a`  
8.8 | `sm_120f`  
Kind `.kind::mxf4nvf4` and `.kind::mxf4` | 8.7 | `sm_120a`, `sm_121a`  
`add`, `sub`, `min`, `max`, `neg` | Types `.u8x4`, `.s8x4` | 9.2 | `sm_120f`  
`add` | Types `.u16x2`, `.s16x2`, `.u32` with `.sat` qualifier | 9.2 | `sm_120f`


##  13.1. [Changes in PTX ISA Version 9.2](#changes-in-ptx-isa-version-9-2)  
  
New Features

PTX ISA version 9.2 introduces the following new features:

  * Adds support for `.u8x4` and `.s8x4` instruction types for `add`, `sub`, `min`, `max`, `neg` instructions.

  * Adds support for `add.sat.{u16x2/s16x2/u32}` instruction.

  * Adds support for `.b128` type for `st.async` instruction.

  * Adds support for `.ignore_oob` qualifier for `cp.async.bulk` instruction.

  * Adds support for `.bf16x2` destination type for `cvt` instruction with `.e4m3x2`, `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` source types.


Semantic Changes and Clarifications

  * For `wgmma.mma_async` instruction with `.atype` and `.btype` as `.e4m3`/`.e5m2` and `.dtype` as `.f32`, current implementation does accumulation at higher than half precision but lower than single precision.


None.


##  13.2. [Changes in PTX ISA Version 9.1](#changes-in-ptx-isa-version-9-1)

New Features

PTX ISA version 9.1 introduces the following new features:

  * Adds support for `.volatile` qualifier with `.local` state space for `ld` and `st` instructions.

  * Adds support for `.f16x2` and `.bf16x2` source types for `cvt` instruction with destination types `.e2m1x2`, `.e2m3x2`, `.e3m2x2`, `.e4m3x2`, `.e5m2x2`.

  * Adds support for `.scale_vec::4X` with `.ue8m0` as `.stype` with `.kind::mxf4nvf4` for `mma`/`mma.sp` instructions.

  * Adds support for `.s2f6x2` instruction type for `cvt` instruction.

  * Adds support for `multimem.cp.async.bulk` and `multimem.cp.reduce.async.bulk` instructions.


Semantic Changes and Clarifications

None.


##  13.3. [Changes in PTX ISA Version 9.0](#changes-in-ptx-isa-version-9-0)

New Features

PTX ISA version 9.0 introduces the following new features:

  * Adds support for `sm_88` target architecture.

  * Adds support for `sm_110` target architecture.

  * Adds support for target `sm_110f` that supports family-specific features.

  * Adds support for target `sm_110a` that supports architecture-specific features.

  * Adds support for pragma `enable_smem_spilling` that is used to enable shared memory spilling for a function.

  * Adds support for pragma `frequency` that is used to specify the execution frequency of a basic block.

  * Adds support for directive `.blocksareclusters` that is used to specify that CUDA thread blocks are mapped to clusters.

  * Extends `size` operand of `st.bulk` instruction to support 32-bit length.

  * Adds support for performance-tuning directives `.abi_preserve` and `.abi_preserve_control` that are used to specify the number of data and control registers that should be preserved by the callers of a function.


Notes

  * Targets `sm_{101,101f,101a}` are renamed to targets `sm_{110,110f,110a}` from PTX ISA version 9.0.


Semantic Changes and Clarifications

  * All `tcgen05` instructions(`tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.cp`, `tcgen05.shift`, `tcgen05.mma`, `tcgen05.mma.sp`, `tcgen05.mma.ws, tcgen05.mma.ws.sp`, `tcgen05.commit`) within a kernel must specify the same value for the `.cta_group` qualifier.


##  13.4. [Changes in PTX ISA Version 8.8](#changes-in-ptx-isa-version-8-8)

New Features

PTX ISA version 8.8 introduces the following new features:

  * Adds support for `sm_103` target architecture.

  * Adds support for target `sm_103a` that supports architecture-specific features.

  * Adds support for `sm_121` target architecture.

  * Adds support for target `sm_121a` that supports architecture-specific features.

  * Introduces family-specific target architectures that are represented with “f” suffix. PTX for family-specific targets is compatible with all subsequent targets in same family. Adds support for `sm_100f`, `sm_101f`, `sm_103f`, `sm_120f`, `sm_121f`.

  * Extends `min` and `max` instructions to support three input arguments.

  * Extends `tcgen05.mma` instruction to add support for new `scale_vectorsize` qualifiers `.block16` and `.block32` and K dimension 96.

  * Extends `.field3` of `tensormap.replace` instruction to support 96B swizzle mode.

  * Adds support for `tcgen05.ld.red` instruction.

  * Extends `ld`, `ld.global.nc` and `st` instructions to support 256b load/store operations.

  * [Table 61](#changes-in-ptx-isa-8-8-family-specific-features) shows the list of features that are supported on family-specific targets:

Table 61 List of features promoted to family-specific architecture Feature | Supported targets  
---|---  
`.m16n8`, `.m16n16`, `.m8n16` shapes and `.b8` type for `ldmatrix`/`stmatrix` | `sm_100f`, `sm_101f`, `sm_120f`  
Shapes for `tcgen05` `.16x64b` `.16x128b`, `.16x256b`, `.16x32bx2`, `.32x32b`, `.4x256b`, `.32x128b`, `.64x128b`, `.128x256b`, `.128x128b`, `.31x256b` | `sm_100f`, `sm_101f`  
`setmaxnreg` | `sm_100f`, `sm_101f`, `sm_120f`  
`.cta_group` modifier | `sm_100f`, `sm_101f`  
`cvt` with `.e2m1x2`, `.e3m2x2`, `.e2m3x2`, `.ue8m0x2` | `sm_100f`, `sm_101f`, `sm_120f`  
`multimem` with `.acc::f16` and `.e5m2`, `.e5m2x2`, `.e5m2x4`, `.e4m3`, `.e4m3x2`, `.e4m3x4` types | `sm_100f`, `sm_101f`  
`tensormap.replace` | `sm_100f`, `sm_101f`, `sm_120f`  
`tcgen05.ld.red` | `sm_101f`, `sm_103f`  
`tcgen05.ld`/`st`/`fence`/ `wait`/`commit`/`cp`/ `alloc`/`dealloc`/ `relinquish_alloc_permit` | `sm_100f`, `sm_101f`  
`tcgen05.mma{.ws}{.sp}` (except `kind::mxf4`/ `kind::mxf4nvf4` for `.sp`) | `sm_100f`, `sm_101f`  
`tcgen05` `.kind::mxf4nvf4`, `.kind::mxf4`, `.kind::mxf8f6f4`, `.kind::f16`, `.kind::tf32`, `.kind::f8f6f4` | `sm_100f`, `sm_101f`  
`.ashift`, `.collector_usage` modifiers for `tcgen05` | `sm_100f`, `sm_101f`  
Modifiers `.b8x16`, `.b6x16_p32`, `.b4x16_p64` | `sm_100f`, `sm_101f`, `sm_120f`  
`.block_scale` modifier | `sm_100f`, `sm_101f`, `sm_120f`  
`mma{.sp}` with `.e3m2`, `.e2m3`, `.e2m1` types and `.kind`, `.block_scale`, `.scale_vec_size` modifiers (except `.sp` with `mxf4`/ `mxf4nvf4`) | `sm_120f`  
`.scale_vec::1X`/`2X`/`4X` modifiers | `sm_120f`  
`.block16`/`.block32` modifiers (alias to `scale_vec`) | `sm_100f`, `sm_101f`  
`.warpx2::02_13`, `.warpx2::01_23`, `.warpx4`, `.pack::16b`, `.unpack::16b` modifiers for `tcgen05` | `sm_100f`, `sm_101f`  
`clusterlaunchcontrol.try_cancel` `multicast::cluster::all` | `sm_100f`, `sm_101f`, `sm_120f`  
`.tile::scatter4`, `.tile::gather4`, `.im2col::w`, `.im2col::w::128` | `sm_100f`, `sm_101f`  
`redux.f32` | `sm_100f`  
`scale-input-d` for `tcgen05` | `sm_100f`  


Semantic Changes and Clarifications

  * Clarified the behavior of float-to-integer conversions for `NaN` input.


##  13.5. [Changes in PTX ISA Version 8.7](#changes-in-ptx-isa-version-8-7)

New Features

PTX ISA version 8.7 introduces the following new features:

  * Adds support for `sm_120` target architecture.

  * Adds support for target `sm_120a` that supports architecture-specific features.

  * Extends `tcgen05.mma` instruction to add support for `.kind::mxf4nvf4` and `.scale_vec::4X` qualifiers.

  * Extends `mma` instructions to support `.f16` type accumulator and shape `.m16n8k16` with FP8 types `.e4m3` and `.e5m2`.

  * Extends `cvt` instruction to add support for `.rs` rounding mode and destination types `.e2m1x4`, `.e4m3x4`, `.e5m2x4`, `.e3m2x4`, `.e2m3x4`.

  * Extends support for `st.async` and `red.async` instructions to add support for `.mmio`, `.release`, `.global` and `.scope` qualifiers.

  * Extends `tensormap.replace` instruction to add support for values `13` to `15` for `.elemtype` qualifier.

  * Extends `mma` and `mma.sp::ordered_metadata` instructions to add support for types `.e3m2`/`.e2m3`/ `.e2m1` and qualifiers `.kind`, `.block_scale`, `.scale_vec_size`.


Semantic Changes and Clarifications

  * Clarified that in `.tile::gather4`, `.tile::scatter4` modes, tensor coordinates need to be specified as {col_idx, row_idx0, row_idx1, row_idx2, row_idx3} i.e. {x, y0, y1, y2, y3} instead of {x0, x1, x2, x3, y}.

  * Updated [Instruction descriptor](#tcgen05-instruction-descriptor) of `tcgen05.mma` instruction to clarify the bits that are reserved for future use.


##  13.6. [Changes in PTX ISA Version 8.6](#changes-in-ptx-isa-version-8-6)

New Features

PTX ISA version 8.6 introduces the following new features:

  * Adds support for `sm_100` target architecture.

  * Adds support for target `sm_100a` that supports architecture-specific features.

  * Adds support for `sm_101` target architecture.

  * Adds support for target `sm_101a` that supports architecture-specific features.

  * Extends `cp.async.bulk` and `cp.async.bulk.tensor` instructions to add `.shared::cta` as destination state space.

  * Extends `fence` instruction to add support for `.acquire` and `.release` qualifiers.

  * Extends `fence` and `fence.proxy` instructions to add support for `.sync_restrict` qualifier.

  * Extends `ldmatrix` instruction to support `.m16n16`, `.m8n16` shapes and `.b8` type.

  * Extends `ldmatrix` instruction to support `.src_fmt`, `.dst_fmt` qualifiers.

  * Extends `stmatrix` instruction to support `.m16n8` shape and `.b8` type.

  * Adds support for `clusterlaunchcontrol` instruction.

  * Extends `add`, `sub` and `fma` instructions to support mixed precision floating point operations with `.f32` as destaination operand type and `.f16`/`.bf16` as source operand types.

  * Extends `add`, `sub`, `mul` and `fma` instructions to support `.f32x2` type.

  * Extends `cvt` instruction with `.tf32` type to support `.satfinite` qualifier for `.rn`/`.rz` rounding modes.

  * Extends `cp.async.bulk` instruction to support `.cp_mask` qualifier and `byteMask` operand.

  * Extends `multimem.ld_reduce` and `multimem.st` instructions to support `.e5m2`, `.e5m2x2`, `.e5m2x4`, `.e4m3`, `.e4m3x2` and `.e4m3x4` types.

  * Extends `cvt` instruction to support conversions to/from `.e2m1x2`, `.e3m2x2`, `.e2m3x2` and `.ue8m0x2` types.

  * Extends `cp.async.bulk.tensor` and `cp.async.bulk.prefetch.tensor` instructions to support new load_mode qualifiers `.tile::scatter4` and `.tile::gather4`.

  * Extends `tensormap.replace` instruction to add support for new qualifier `.swizzle_atomicity` for supporting new swizzle modes.

  * Extends `mbarrier.arrive`, `mbarrier.arrive_drop`, `.mbarrier.test_wait` and `.mbarrier.try_wait` instructions to support `.relaxed` qualifier.

  * Extends `cp.async.bulk.tensor` and `cp.async.bulk.prefetch.tensor` instructions to support new load_mode qualifiers `.im2col::w` and `.im2col::w::128`.

  * Extends `cp.async.bulk.tensor` instruction to support new qualifier `.cta_group`.

  * Add support for `st.bulk` instruction.

  * Adds support for tcgen05 features and related instructions: `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.ld`, `tcgen05.st`, `tcgen05.wait`, `tcgen05.cp`, `tcgen05.shift`, `tcgen05.mma`, `tcgen05.mma.sp`, `tcgen05.mma.ws`, `tcgen05.mma.ws.sp`, `tcgen05.fence` and `tcgen05.commit`.

  * Extends `redux.sync` instruction to add support for `.f32` type with qualifiers `.abs` and `.NaN`.


Semantic Changes and Clarifications

None.


##  13.7. [Changes in PTX ISA Version 8.5](#changes-in-ptx-isa-version-8-5)

New Features

PTX ISA version 8.5 introduces the following new features:

  * Adds support for `mma.sp::ordered_metadata` instruction.


Semantic Changes and Clarifications

  * Values `0b0000`, `0b0101`, `0b1010`, `0b1111` for sparsity metadata (operand `e`) of instruction `mma.sp` are invalid and their usage results in undefined behavior.


##  13.8. [Changes in PTX ISA Version 8.4](#changes-in-ptx-isa-version-8-4)

New Features

PTX ISA version 8.4 introduces the following new features:

  * Extends `ld`, `st` and `atom` instructions with `.b128` type to support `.sys` scope.

  * Extends integer `wgmma.mma_async` instruction to support `.u8.s8` and `.s8.u8` as `.atype` and `.btype` respectively.

  * Extends `mma`, `mma.sp` instructions to support FP8 types `.e4m3` and `.e5m2`.


Semantic Changes and Clarifications

None.


##  13.9. [Changes in PTX ISA Version 8.3](#changes-in-ptx-isa-version-8-3)

New Features

PTX ISA version 8.3 introduces the following new features:

  * Adds support for pragma `used_bytes_mask` that is used to specify mask for used bytes for a load operation.

  * Extends `isspacep`, `cvta.to`, `ld` and `st` instructions to accept `::entry` and `::func` sub-qualifiers with `.param` state space qualifier.

  * Adds support for `.b128` type on instructions `ld`, `ld.global.nc`, `ldu`, `st`, `mov` and `atom`.

  * Add support for instructions `tensormap.replace`, `tensormap.cp_fenceproxy` and support for qualifier `.to_proxykind::from_proxykind` on instruction `fence.proxy` to support modifying `tensor-map`.


Semantic Changes and Clarifications

None.


##  13.10. [Changes in PTX ISA Version 8.2](#changes-in-ptx-isa-version-8-2)

New Features

PTX ISA version 8.2 introduces the following new features:

  * Adds support for `.mmio` qualifier on `ld` and `st` instructions.

  * Extends `lop3` instruction to allow predicate destination.

  * Extends `multimem.ld_reduce` instruction to support `.acc::f32` qualifer to allow `.f32` precision of the intermediate accumulation.

  * Extends the asynchronous warpgroup-level matrix multiply-and-accumulate operation `wgmma.mma_async` to support `.sp` modifier that allows matrix multiply-accumulate operation when input matrix A is sparse.


Semantic Changes and Clarifications

The `.multicast::cluster` qualifier on `cp.async.bulk` and `cp.async.bulk.tensor` instructions is optimized for target architecture `sm_90a` and may have substantially reduced performance on other targets and hence `.multicast::cluster` is advised to be used with `sm_90a`.


##  13.11. [Changes in PTX ISA Version 8.1](#changes-in-ptx-isa-version-8-1)

New Features

PTX ISA version 8.1 introduces the following new features:

  * Adds support for `st.async` and `red.async` instructions for asynchronous store and asynchronous reduction operations respectively on shared memory.

  * Adds support for `.oob` modifier on half-precision `fma` instruction.

  * Adds support for `.satfinite` saturation modifer on `cvt` instruction for `.f16`, `.bf16` and `.tf32` formats.

  * Extends support for `cvt` with `.e4m3`/`.e5m2` to `sm_89`.

  * Extends `atom` and `red` instructions to support vector types.

  * Adds support for special register `%aggr_smem_size`.

  * Extends `sured` instruction with 64-bit `min`/`max` operations.

  * Adds support for increased kernel parameter size of 32764 bytes.

  * Adds support for multimem addresses in memory consistency model.

  * Adds support for `multimem.ld_reduce`, `multimem.st` and `multimem.red` instructions to perform memory operations on multimem addresses.


Semantic Changes and Clarifications

None.


##  13.12. [Changes in PTX ISA Version 8.0](#changes-in-ptx-isa-version-8-0)

New Features

PTX ISA version 8.0 introduces the following new features:

  * Adds support for target `sm_90a` that supports architecture-specific features.

  * Adds support for asynchronous warpgroup-level matrix multiply-and-accumulate operation `wgmma`.

  * Extends the asynchronous copy operations with bulk operations that operate on large data, including tensor data.

  * Introduces packed integer types `.u16x2` and `.s16x2`.

  * Extends integer arithmetic instruction `add` to allow packed integer types `.u16x2` and `.s16x2`.

  * Extends integer arithmetic instructions `min` and `max` to allow packed integer types `.u16x2` and `.s16x2`, as well as saturation modifier `.relu` on `.s16x2` and `.s32` types.

  * Adds support for special register `%current_graph_exec` that identifies the currently executing CUDA device graph.

  * Adds support for `elect.sync` instruction.

  * Adds support for `.unified` attribute on functions and variables.

  * Adds support for `setmaxnreg` instruction.

  * Adds support for `.sem` qualifier on `barrier.cluster` instruction.

  * Extends the `fence` instruction to allow opcode-specific synchronizaion using `op_restrict` qualifier.

  * Adds support for `.cluster` scope on `mbarrier.arrive`, `mbarrier.arrive_drop`, `mbarrier.test_wait` and `mbarrier.try_wait` operations.

  * Adds support for transaction count operations on `mbarrier` objects, specified with `.expect_tx` and `.complete_tx` qualifiers.


Semantic Changes and Clarifications

None.


##  13.13. [Changes in PTX ISA Version 7.8](#changes-in-ptx-isa-version-7-8)

New Features

PTX ISA version 7.8 introduces the following new features:

  * Adds support for `sm_89` target architecture.

  * Adds support for `sm_90` target architecture.

  * Extends `bar` and `barrier` instructions to accept optional scope qualifier `.cta`.

  * Extends `.shared` state space qualifier with optional sub-qualifier `::cta`.

  * Adds support for `movmatrix` instruction which transposes a matrix in registers across a warp.

  * Adds support for `stmatrix` instruction which stores one or more matrices to shared memory.

  * Extends the `.f64` floating point type `mma` operation with shapes `.m16n8k4`, `.m16n8k8`, and `.m16n8k16`.

  * Extends `add`, `sub`, `mul`, `set`, `setp`, `cvt`, `tanh`, `ex2`, `atom` and `red` instructions with `bf16` alternate floating point data format.

  * Adds support for new alternate floating-point data formats `.e4m3` and `.e5m2`.

  * Extends `cvt` instruction to convert `.e4m3` and `.e5m2` alternate floating point data formats.

  * Adds support for `griddepcontrol` instruction as a communication mechanism to control the execution of dependent grids.

  * Extends `mbarrier` instruction to allow a new phase completion check operation _try_wait_.

  * Adds support for new thread scope `.cluster` which is a set of Cooperative Thread Arrays (CTAs).

  * Extends `fence`/`membar`, `ld`, `st`, `atom`, and `red` instructions to accept `.cluster` scope.

  * Adds support for extended visibility of shared state space to all threads within a cluster.

  * Extends `.shared` state space qualifier with `::cluster` sub-qualifier for cluster-level visibility of shared memory.

  * Extends `isspacep`, `cvta`, `ld`, `st`, `atom`, and `red` instructions to accept `::cluster` sub-qualifier with `.shared` state space qualifier.

  * Adds support for `mapa` instruction to map a shared memory address to the corresponding address in a different CTA within the cluster.

  * Adds support for `getctarank` instruction to query the rank of the CTA that contains a given address.

  * Adds support for new barrier synchronization instruction `barrier.cluster`.

  * Extends the memory consistency model to include the new cluster scope.

  * Adds support for special registers related to cluster information: `%is_explicit_cluster`, `%clusterid`, `%nclusterid`, `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`, `%cluster_nctarank`.

  * Adds support for cluster dimension directives `.reqnctapercluster`, `.explicitcluster`, and `.maxclusterrank`.


Semantic Changes and Clarifications

None.


##  13.14. [Changes in PTX ISA Version 7.7](#changes-in-ptx-isa-version-7-7)

New Features

PTX ISA version 7.7 introduces the following new features:

  * Extends `isspacep` and `cvta` instructions to include the `.param` state space for kernel function parameters.


Semantic Changes and Clarifications

None.


##  13.15. [Changes in PTX ISA Version 7.6](#changes-in-ptx-isa-version-7-6)

New Features

PTX ISA version 7.6 introduces the following new features:

  * Support for `szext` instruction which performs sign-extension or zero-extension on a specified value.

  * Support for `bmsk` instruction which creates a bitmask of the specified width starting at the specified bit position.

  * Support for special registers `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset<2>`.


Semantic Changes and Clarifications

None.


##  13.16. [Changes in PTX ISA Version 7.5](#changes-in-ptx-isa-version-7-5)

New Features

PTX ISA version 7.5 introduces the following new features:

  * Debug information enhancements to support label difference and negative values in the `.section` debugging directive.

  * Support for `ignore-src` operand on `cp.async` instruction.

  * Extensions to the memory consistency model to introduce the following new concepts:

>     * A _memory proxy_ as an abstract label for different methods of memory access.
> 
>     * Virtual aliases as distinct memory addresses accessing the same physical memory location.

  * Support for new `fence.proxy` and `membar.proxy` instructions to allow synchronization of memory accesses performed via virtual aliases.


Semantic Changes and Clarifications

None.


##  13.17. [Changes in PTX ISA Version 7.4](#changes-in-ptx-isa-version-7-4)

New Features

PTX ISA version 7.4 introduces the following new features:

  * Support for `sm_87` target architecture.

  * Support for `.level::eviction_priority` qualifier which allows specifying cache eviction priority hints on `ld`, `ld.global.nc`, `st`, and `prefetch` instructions.

  * Support for `.level::prefetch_size` qualifier which allows specifying data prefetch hints on `ld` and `cp.async` instructions.

  * Support for `createpolicy` instruction which allows construction of different types of cache eviction policies.

  * Support for `.level::cache_hint` qualifier which allows the use of cache eviction policies with `ld`, `ld.global.nc`, `st`, `atom`, `red` and `cp.async` instructions.

  * Support for `applypriority` and `discard` operations on cached data.


Semantic Changes and Clarifications

None.


##  13.18. [Changes in PTX ISA Version 7.3](#changes-in-ptx-isa-version-7-3)

New Features

PTX ISA version 7.3 introduces the following new features:

  * Extends `mask()` operator used in initializers to also support integer constant expression.

  * Adds support for stack manpulation instructions that allow manipulating stack using `stacksave` and `stackrestore` instructions and allocation of per-thread stack using `alloca` instruction.


Semantic Changes and Clarifications

The unimplemented version of `alloca` from the older PTX ISA specification has been replaced with new stack manipulation instructions in PTX ISA version 7.3.


##  13.19. [Changes in PTX ISA Version 7.2](#changes-in-ptx-isa-version-7-2)

New Features

PTX ISA version 7.2 introduces the following new features:

  * Enhances `.loc` directive to represent inline function information.

  * Adds support to define labels inside the debug sections.

  * Extends `min` and `max` instructions to support `.xorsign` and `.abs` modifiers.


Semantic Changes and Clarifications

None.


##  13.20. [Changes in PTX ISA Version 7.1](#changes-in-ptx-isa-version-7-1)

New Features

PTX ISA version 7.1 introduces the following new features:

  * Support for `sm_86` target architecture.

  * Adds a new operator, `mask()`, to extract a specific byte from variable’s address used in initializers.

  * Extends `tex` and `tld4` instructions to return an optional predicate that indicates if data at specified coordinates is resident in memory.

  * Extends single-bit `wmma` and `mma` instructions to support `.and` operation.

  * Extends `mma` instruction to support `.sp` modifier that allows matrix multiply-accumulate operation when input matrix A is sparse.

  * Extends `mbarrier.test_wait` instruction to test the completion of specific phase parity.


Semantic Changes and Clarifications

None.


##  13.21. [Changes in PTX ISA Version 7.0](#changes-in-ptx-isa-version-7-0)

New Features

PTX ISA version 7.0 introduces the following new features:

  * Support for `sm_80` target architecture.

  * Adds support for asynchronous copy instructions that allow copying of data asynchronously from one state space to another.

  * Adds support for `mbarrier` instructions that allow creation of _mbarrier objects_ in memory and use of these objects to synchronize threads and asynchronous copy operations initiated by threads.

  * Adds support for `redux.sync` instruction which allows reduction operation across threads in a warp.

  * Adds support for new alternate floating-point data formats `.bf16` and `.tf32`.

  * Extends `wmma` instruction to support `.f64` type with shape `.m8n8k4`.

  * Extends `wmma` instruction to support `.bf16` data format.

  * Extends `wmma` instruction to support `.tf32` data format with shape `.m16n16k8`.

  * Extends `mma` instruction to support `.f64` type with shape `.m8n8k4`.

  * Extends `mma` instruction to support `.bf16` and `.tf32` data formats with shape `.m16n8k8`.

  * Extends `mma` instruction to support new shapes `.m8n8k128`, `.m16n8k4`, `.m16n8k16`, `.m16n8k32`, `.m16n8k64`, `.m16n8k128` and `.m16n8k256`.

  * Extends `abs` and `neg` instructions to support `.bf16` and `.bf16x2` data formats.

  * Extends `min` and `max` instructions to support `.NaN` modifier and `.f16`, `.f16x2`, `.bf16` and `.bf16x2` data formats.

  * Extends `fma` instruction to support `.relu` saturation mode and `.bf16` and `.bf16x2` data formats.

  * Extends `cvt` instruction to support `.relu` saturation mode and `.f16`, `.f16x2`, `.bf16`, `.bf16x2` and `.tf32` destination formats.

  * Adds support for `tanh` instruction that computes hyperbolic-tangent.

  * Extends `ex2` instruction to support `.f16` and `.f16x2` types.


Semantic Changes and Clarifications

None.


##  13.22. [Changes in PTX ISA Version 6.5](#changes-in-ptx-isa-version-6-5)

New Features

PTX ISA version 6.5 introduces the following new features:

  * Adds support for integer destination types for half precision comparison instruction `set`.

  * Extends `abs` instruction to support `.f16` and `.f16x2` types.

  * Adds support for `cvt.pack` instruction which allows converting two integer values and packing the results together.

  * Adds new shapes `.m16n8k8`, `.m8n8k16` and `.m8n8k32` on the `mma` instruction.

  * Adds support for `ldmatrix` instruction which loads one or more matrices from shared memory for `mma` instruction.


Removed Features

PTX ISA version 6.5 removes the following features:

  * Support for `.satfinite` qualifier on floating point `wmma.mma` instruction has been removed. This support was deprecated since PTX ISA version 6.4.


Semantic Changes and Clarifications

None.


##  13.23. [Changes in PTX ISA Version 6.4](#changes-in-ptx-isa-version-6-4)

New Features

PTX ISA version 6.4 introduces the following new features:

  * Adds support for `.noreturn` directive which can be used to indicate a function does not return to it’s caller function.

  * Adds support for `mma` instruction which allows performing matrix multiply-and-accumulate operation.


Deprecated Features

PTX ISA version 6.4 deprecates the following features:

  * Support for `.satfinite` qualifier on floating point `wmma.mma` instruction.


Removed Features

PTX ISA version 6.4 removes the following features:

  * Support for `shfl` and `vote` instructions without the `.sync` qualifier has been removed for `.target``sm_70` and higher. This support was deprecated since PTX ISA version 6.0 as documented in PTX ISA version 6.2.


Semantic Changes and Clarifications

  * Clarified that resolving references of a `.weak` symbol considers only `.weak` or `.visible` symbols with the same name and does not consider local symbols with the same name.

  * Clarified that in `cvt` instruction, modifier `.ftz` can only be specified when either `.atype` or `.dtype` is `.f32`.


##  13.24. [Changes in PTX ISA Version 6.3](#changes-in-ptx-isa-version-6-3)

New Features

PTX ISA version 6.3 introduces the following new features:

  * Support for `sm_75` target architecture.

  * Adds support for a new instruction `nanosleep` that suspends a thread for a specified duration.

  * Adds support for `.alias` directive which allows definining alias to function symbol.

  * Extends `atom` instruction to perform `.f16` addition operation and `.cas.b16` operation.

  * Extends `red` instruction to perform `.f16` addition operation.

  * The `wmma` instructions are extended to support multiplicand matrices of type `.s8`, `.u8`, `.s4`, `.u4`, `.b1` and accumulator matrices of type `.s32`.


Semantic Changes and Clarifications

  * Introduced the mandatory `.aligned` qualifier for all `wmma` instructions.

  * Specified the alignment required for the base address and stride parameters passed to `wmma.load` and `wmma.store`.

  * Clarified that layout of fragment returned by `wmma` operation is architecture dependent and passing `wmma` fragments around functions compiled for different link compatible SM architectures may not work as expected.

  * Clarified that atomicity for `{atom/red}.f16x2}` operations is guranteed separately for each of the two `.f16` elements but not guranteed to be atomic as single 32-bit access.


##  13.25. [Changes in PTX ISA Version 6.2](#changes-in-ptx-isa-version-6-2)

New Features

PTX ISA version 6.2 introduces the following new features:

  * A new instruction `activemask` for querying active threads in a warp.

  * Extends atomic and reduction instructions to perform `.f16x2` addition operation with mandatory `.noftz` qualifier.


Deprecated Features

PTX ISA version 6.2 deprecates the following features:

  * The use of `shfl` and `vote` instructions without the `.sync` is deprecated retrospectively from PTX ISA version 6.0, which introduced the `sm_70` architecture that implements [Independent Thread Scheduling](#independent-thread-scheduling).


Semantic Changes and Clarifications

  * Clarified that `wmma` instructions can be used in conditionally executed code only if it is known that all threads in the warp evaluate the condition identically, otherwise behavior is undefined.

  * In the memory consistency model, the definition of _morally strong operations_ was updated to exclude fences from the requirement of _complete overlap_ since fences do not access memory.


##  13.26. [Changes in PTX ISA Version 6.1](#changes-in-ptx-isa-version-6-1)

New Features

PTX ISA version 6.1 introduces the following new features:

  * Support for `sm_72` target architecture.

  * Support for new matrix shapes `32x8x16` and `8x32x16` in `wmma` instruction.


Semantic Changes and Clarifications

None.


##  13.27. [Changes in PTX ISA Version 6.0](#changes-in-ptx-isa-version-6-0)

New Features

PTX ISA version 6.0 introduces the following new features:

  * Support for `sm_70` target architecture.

  * Specifies the memory consistency model for programs running on `sm_70` and later architectures.

  * Various extensions to memory instructions to specify memory synchronization semantics and scopes at which such synchronization can be observed.

  * New instruction `wmma` for matrix operations which allows loading matrices from memory, performing multiply-and-accumulate on them and storing result in memory.

  * Support for new `barrier` instruction.

  * Extends `neg` instruction to support `.f16` and `.f16x2` types.

  * A new instruction `fns` which allows finding n-th set bit in integer.

  * A new instruction `bar.warp.sync` which allows synchronizing threads in warp.

  * Extends `vote` and `shfl` instructions with `.sync` modifier which waits for specified threads before executing the `vote` and `shfl` operation respectively.

  * A new instruction `match.sync` which allows broadcasting and comparing a value across threads in warp.

  * A new instruction `brx.idx` which allows branching to a label indexed from list of potential targets.

  * Support for unsized array parameter for `.func` which can be used to implement variadic functions.

  * Support for `.b16` integer type in dwarf-lines.

  * Support for taking address of device function return parameters using `mov` instruction.


Semantic Changes and Clarifications

  * Semantics of `bar` instruction were updated to indicate that executing thread waits for other non-exited threads from it’s warp.

  * Support for indirect branch introduced in PTX 2.1 which was unimplemented has been removed from the spec.

  * Support for taking address of labels, using labels in initializers which was unimplemented has been removed from the spec.

  * Support for variadic functions which was unimplemented has been removed from the spec.


##  13.28. [Changes in PTX ISA Version 5.0](#changes-in-ptx-isa-version-5-0)

New Features

PTX ISA version 5.0 introduces the following new features:

  * Support for `sm_60`, `sm_61`, `sm_62` target architecture.

  * Extends atomic and reduction instructions to perform double-precision add operation.

  * Extends atomic and reduction instructions to specify `scope` modifier.

  * A new `.common` directive to permit linking multiple object files containing declarations of the same symbol with different size.

  * A new `dp4a` instruction which allows 4-way dot product with accumulate operation.

  * A new `dp2a` instruction which allows 2-way dot product with accumulate operation.

  * Support for special register `%clock_hi`.


Semantic Changes and Clarifications

Semantics of cache modifiers on `ld` and `st` instructions were clarified to reflect cache operations are treated as performance hint only and do not change memory consistency behavior of the program.

Semantics of `volatile` operations on `ld` and `st` instructions were clarified to reflect how `volatile` operations are handled by optimizing compiler.


##  13.29. [Changes in PTX ISA Version 4.3](#changes-in-ptx-isa-version-4-3)

New Features

PTX ISA version 4.3 introduces the following new features:

  * A new `lop3` instruction which allows arbitrary logical operation on 3 inputs.

  * Adds support for 64-bit computations in extended precision arithmetic instructions.

  * Extends `tex.grad` instruction to support `cube` and `acube` geometries.

  * Extends `tld4` instruction to support `a2d`, `cube` and `acube` geometries.

  * Extends `tex` and `tld4` instructions to support optional operands for offset vector and depth compare.

  * Extends `txq` instruction to support querying texture fields from specific LOD.


Semantic Changes and Clarifications

None.


##  13.30. [Changes in PTX ISA Version 4.2](#changes-in-ptx-isa-version-4-2)

New Features

PTX ISA version 4.2 introduces the following new features:

  * Support for `sm_53` target architecture.

  * Support for arithmetic, comparsion and texture instructions for `.f16` and `.f16x2` types.

  * Support for `memory_layout` field for surfaces and `suq` instruction support for querying this field.


Semantic Changes and Clarifications

Semantics for parameter passing under ABI were updated to indicate `ld.param` and `st.param` instructions used for argument passing cannot be predicated.

Semantics of `{atom/red}.add.f32` were updated to indicate subnormal inputs and results are flushed to sign-preserving zero for atomic operations on global memory; whereas atomic operations on shared memory preserve subnormal inputs and results and don’t flush them to zero.


##  13.31. [Changes in PTX ISA Version 4.1](#changes-in-ptx-isa-version-4-1)

New Features

PTX ISA version 4.1 introduces the following new features:

  * Support for `sm_37` and `sm_52` target architectures.

  * Support for new fields `array_size`, `num_mipmap_levels` and `num_samples` for Textures, and the `txq` instruction support for querying these fields.

  * Support for new field `array_size` for Surfaces, and the `suq` instruction support for querying this field.

  * Support for special registers `%total_smem_size` and `%dynamic_smem_size`.


Semantic Changes and Clarifications

None.


##  13.32. [Changes in PTX ISA Version 4.0](#changes-in-ptx-isa-version-4-0)

New Features

PTX ISA version 4.0 introduces the following new features:

  * Support for `sm_32` and `sm_50` target architectures.

  * Support for 64bit performance counter special registers `%pm0_64,..,%pm7_64`.

  * A new `istypep` instruction.

  * A new instruction, `rsqrt.approx.ftz.f64` has been added to compute a fast approximation of the square root reciprocal of a value.

  * Support for a new directive `.attribute` for specifying special attributes of a variable.

  * Support for `.managed` variable attribute.


Semantic Changes and Clarifications

The `vote` instruction semantics were updated to clearly indicate that an inactive thread in a warp contributes a 0 for its entry when participating in `vote.ballot.b32`.


##  13.33. [Changes in PTX ISA Version 3.2](#changes-in-ptx-isa-version-3-2)

New Features

PTX ISA version 3.2 introduces the following new features:

  * The texture instruction supports reads from multi-sample and multisample array textures.

  * Extends `.section` debugging directive to include label + immediate expressions.

  * Extends `.file` directive to include timestamp and file size information.


Semantic Changes and Clarifications

The `vavrg2` and `vavrg4` instruction semantics were updated to indicate that instruction adds 1 only if Va[i] + Vb[i] is non-negative, and that the addition result is shifted by 1 (rather than being divided by 2).


##  13.34. [Changes in PTX ISA Version 3.1](#changes-in-ptx-isa-version-3-1)

New Features

PTX ISA version 3.1 introduces the following new features:

  * Support for `sm_35` target architecture.

  * Support for CUDA Dynamic Parallelism, which enables a kernel to create and synchronize new work.

  * `ld.global.nc` for loading read-only global data though the non-coherent texture cache.

  * A new funnel shift instruction, `shf`.

  * Extends atomic and reduction instructions to perform 64-bit `{and, or, xor}` operations, and 64-bit integer `{min, max}` operations.

  * Adds support for `mipmaps`.

  * Adds support for indirect access to textures and surfaces.

  * Extends support for generic addressing to include the `.const` state space, and adds a new operator, `generic()`, to form a generic address for `.global` or `.const` variables used in initializers.

  * A new `.weak` directive to permit linking multiple object files containing declarations of the same symbol.


Semantic Changes and Clarifications

PTX 3.1 redefines the default addressing for global variables in initializers, from generic addresses to offsets in the global state space. Legacy PTX code is treated as having an implicit `generic()` operator for each global variable used in an initializer. PTX 3.1 code should either include explicit `generic()` operators in initializers, use `cvta.global` to form generic addresses at runtime, or load from the non-generic address using `ld.global`.

Instruction `mad.f32` requires a rounding modifier for `sm_20` and higher targets. However for PTX ISA version 3.0 and earlier, ptxas does not enforce this requirement and `mad.f32` silently defaults to `mad.rn.f32`. For PTX ISA version 3.1, ptxas generates a warning and defaults to `mad.rn.f32`, and in subsequent releases ptxas will enforce the requirement for PTX ISA version 3.2 and later.


##  13.35. [Changes in PTX ISA Version 3.0](#changes-in-ptx-isa-version-3-0)

New Features

PTX ISA version 3.0 introduces the following new features:

  * Support for `sm_30` target architectures.

  * SIMD video instructions.

  * A new warp shuffle instruction.

  * Instructions `mad.cc` and `madc` for efficient, extended-precision integer multiplication.

  * Surface instructions with 3D and array geometries.

  * The texture instruction supports reads from cubemap and cubemap array textures.

  * Platform option `.target` debug to declare that a PTX module contains `DWARF` debug information.

  * `pmevent.mask`, for triggering multiple performance monitor events.

  * Performance monitor counter special registers `%pm4..%pm7`.


Semantic Changes and Clarifications

Special register `%gridid` has been extended from 32-bits to 64-bits.

PTX ISA version 3.0 deprecates module-scoped `.reg` and `.local` variables when compiling to the Application Binary Interface (ABI). When compiling without use of the ABI, module-scoped `.reg` and `.local` variables are supported as before. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.reg` or `.local` variables, the compiler silently disables use of the ABI.

The `shfl` instruction semantics were updated to clearly indicate that value of source operand `a` is unpredictable for inactive and predicated-off threads within the warp.

PTX modules no longer allow duplicate `.version` directives. This feature was unimplemented, so there is no semantic change.

Unimplemented instructions `suld.p` and `sust.p.{u32,s32,f32}` have been removed.


##  13.36. [Changes in PTX ISA Version 2.3](#changes-in-ptx-isa-version-2-3)

New Features

PTX 2.3 adds support for texture arrays. The texture array feature supports access to an array of 1D or 2D textures, where an integer indexes into the array of textures, and then one or two single-precision floating point coordinates are used to address within the selected 1D or 2D texture.

PTX 2.3 adds a new directive, `.address_size`, for specifying the size of addresses.

Variables in `.const` and `.global` state spaces are initialized to zero by default.

Semantic Changes and Clarifications

The semantics of the `.maxntid` directive have been updated to match the current implementation. Specifically, `.maxntid` only guarantees that the total number of threads in a thread block does not exceed the maximum. Previously, the semantics indicated that the maximum was enforced separately in each dimension, which is not the case.

Bit field extract and insert instructions BFE and BFI now indicate that the `len` and `pos` operands are restricted to the value range `0..255`.

Unimplemented instructions `{atom,red}.{min,max}.f32` have been removed.


##  13.37. [Changes in PTX ISA Version 2.2](#changes-in-ptx-isa-version-2-2)

New Features

PTX 2.2 adds a new directive for specifying kernel parameter attributes; specifically, there is a new directives for specifying that a kernel parameter is a pointer, for specifying to which state space the parameter points, and for optionally specifying the alignment of the memory to which the parameter points.

PTX 2.2 adds a new field named `force_unnormalized_coords` to the `.samplerref` opaque type. This field is used in the independent texturing mode to override the `normalized_coords` field in the texture header. This field is needed to support languages such as OpenCL, which represent the property of normalized/unnormalized coordinates in the sampler header rather than in the texture header.

PTX 2.2 deprecates explicit constant banks and supports a large, flat address space for the `.const` state space. Legacy PTX that uses explicit constant banks is still supported.

PTX 2.2 adds a new `tld4` instruction for loading a component (`r`, `g`, `b`, or `a`) from the four texels compising the bilinear interpolation footprint of a given texture location. This instruction may be used to compute higher-precision bilerp results in software, or for performing higher-bandwidth texture loads.

Semantic Changes and Clarifications

None.


##  13.38. [Changes in PTX ISA Version 2.1](#changes-in-ptx-isa-version-2-1)

New Features

The underlying, stack-based ABI is supported in PTX ISA version 2.1 for `sm_2x` targets.

Support for indirect calls has been implemented for `sm_2x` targets.

New directives, `.branchtargets` and `.calltargets`, have been added for specifying potential targets for indirect branches and indirect function calls. A `.callprototype` directive has been added for declaring the type signatures for indirect function calls.

The names of `.global` and `.const` variables can now be specified in variable initializers to represent their addresses.

A set of thirty-two driver-specific execution environment special registers has been added. These are named `%envreg0..%envreg31`.

Textures and surfaces have new fields for channel data type and channel order, and the `txq` and `suq` instructions support queries for these fields.

Directive `.minnctapersm` has replaced the `.maxnctapersm` directive.

Directive `.reqntid` has been added to allow specification of exact CTA dimensions.

A new instruction, `rcp.approx.ftz.f64`, has been added to compute a fast, gross approximate reciprocal.

Semantic Changes and Clarifications

A warning is emitted if `.minnctapersm` is specified without also specifying `.maxntid`.


##  13.39. [Changes in PTX ISA Version 2.0](#changes-in-ptx-isa-version-2-0)

New Features

Floating Point Extensions

This section describes the floating-point changes in PTX ISA version 2.0 for `sm_20` targets. The goal is to achieve IEEE 754 compliance wherever possible, while maximizing backward compatibility with legacy PTX ISA version 1.x code and `sm_1x` targets.

The changes from PTX ISA version 1.x are as follows:

  * Single-precision instructions support subnormal numbers by default for `sm_20` targets. The `.ftz` modifier may be used to enforce backward compatibility with `sm_1x`.

  * Single-precision `add`, `sub`, and `mul` now support `.rm` and `.rp` rounding modifiers for `sm_20` targets.

  * A single-precision fused multiply-add (fma) instruction has been added, with support for IEEE 754 compliant rounding modifiers and support for subnormal numbers. The `fma.f32` instruction also supports `.ftz` and `.sat` modifiers. `fma.f32` requires `sm_20`. The `mad.f32` instruction has been extended with rounding modifiers so that it’s synonymous with `fma.f32` for `sm_20` targets. Both `fma.f32` and `mad.f32` require a rounding modifier for `sm_20` targets.

  * The `mad.f32` instruction _without rounding_ is retained so that compilers can generate code for `sm_1x` targets. When code compiled for `sm_1x` is executed on `sm_20` devices, `mad.f32` maps to `fma.rn.f32`.

  * Single- and double-precision `div`, `rcp`, and `sqrt` with IEEE 754 compliant rounding have been added. These are indicated by the use of a rounding modifier and require `sm_20`.

  * Instructions `testp` and `copysign` have been added.


New Instructions

A _load uniform_ instruction, `ldu`, has been added.

Surface instructions support additional `.clamp` modifiers, `.clamp` and `.zero`.

Instruction `sust` now supports formatted surface stores.

A _count leading zeros_ instruction, `clz`, has been added.

A _find leading non-sign bit instruction_ , `bfind`, has been added.

A _bit reversal_ instruction, `brev`, has been added.

Bit field extract and insert instructions, `bfe` and `bfi`, have been added.

A _population count_ instruction, `popc`, has been added.

A _vote ballot_ instruction, `vote.ballot.b32`, has been added.

Instructions `{atom,red}.add.f32` have been implemented.

Instructions `{atom,red}`.shared have been extended to handle 64-bit data types for `sm_20` targets.

A system-level membar instruction, `membar.sys`, has been added.

The `bar` instruction has been extended as follows:

  * A `bar.arrive` instruction has been added.

  * Instructions `bar.red.popc.u32` and `bar.red.{and,or}.pred` have been added.

  * `bar` now supports optional thread count and register operands.


Scalar video instructions (includes `prmt`) have been added.

Instruction `isspacep` for querying whether a generic address falls within a specified state space window has been added.

Instruction `cvta` for converting global, local, and shared addresses to generic address and vice-versa has been added.

Other New Features

Instructions `ld`, `ldu`, `st`, `prefetch`, `prefetchu`, `isspacep`, `cvta`, `atom`, and `red` now support generic addressing.

New special registers `%nwarpid`, `%nsmid`, `%clock64`, `%lanemask_{eq,le,lt,ge,gt}` have been added.

Cache operations have been added to instructions `ld`, `st`, `suld`, and `sust`, e.g., for `prefetching` to specified level of memory hierarchy. Instructions `prefetch` and `prefetchu` have also been added.

The `.maxnctapersm` directive was deprecated and replaced with `.minnctapersm` to better match its behavior and usage.

A new directive, `.section`, has been added to replace the `@@DWARF` syntax for passing DWARF-format debugging information through PTX.

A new directive, `.pragma nounroll`, has been added to allow users to disable loop unrolling.

Semantic Changes and Clarifications

The errata in `cvt.ftz` for PTX ISA versions 1.4 and earlier, where single-precision subnormal inputs and results were not flushed to zero if either source or destination type size was 64-bits, has been fixed. In PTX ISA version 1.5 and later, `cvt.ftz` (and `cvt` for `.target sm_1x`, where `.ftz` is implied) instructions flush single-precision subnormal inputs and results to sign-preserving zero for all combinations of floating-point instruction types. To maintain compatibility with legacy PTX code, if .version is 1.4 or earlier, single-precision subnormal inputs and results are flushed to sign-preserving zero only when neither source nor destination type size is 64-bits.

Components of special registers `%tid`, `%ntid`, `%ctaid`, and `%nctaid` have been extended from 16-bits to 32-bits. These registers now have type `.v4.u32`.

The number of samplers available in independent texturing mode was incorrectly listed as thirty-two in PTX ISA version 1.5; the correct number is sixteen.
