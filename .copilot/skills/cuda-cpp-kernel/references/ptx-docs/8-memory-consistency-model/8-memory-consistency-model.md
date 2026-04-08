# 8. Memory Consistency Model


In multi-threaded executions, the side-effects of memory operations performed by each thread become visible to other threads in a partial and non-identical order. This means that any two operations may appear to happen in no order, or in different orders, to different threads. The axioms introduced by the memory consistency model specify exactly which contradictions are forbidden between the orders observed by different threads.


In the absence of any constraint, each read operation returns the value committed by some write operation to the same memory location, including the initial write to that memory location. The memory consistency model effectively constrains the set of such candidate writes from which a read operation can return a value.


##  8.1. [Scope and applicability of the model](#scope-and-applicability)

The constraints specified under this model apply to PTX programs with any PTX ISA version number, running on `sm_70` or later architectures.

The memory consistency model does not apply to texture (including `ld.global.nc`) and surface accesses.

###  8.1.1. [Limitations on atomicity at system scope](#limitations-system-scope-atomicity)

When communicating with the host CPU, certain strong operations with system scope may not be performed atomically on some systems. For more details on atomicity guarantees to host memory, see the _CUDA Atomicity Requirements_.


##  8.2. [Memory operations](#memory-operations)

The fundamental storage unit in the PTX memory model is a byte, consisting of 8 bits. Each state space available to a PTX program is a sequence of contiguous bytes in memory. Every byte in a PTX state space has a unique address relative to all threads that have access to the same state space.

Each PTX memory instruction specifies an address operand and a data type. The address operand contains a virtual address that gets converted to a physical address during memory access. The physical address and the size of the data type together define a physical memory location, which is the range of bytes starting from the physical address and extending up to the size of the data type in bytes.

The memory consistency model specification uses the terms “address” or “memory address” to indicate a virtual address, and the term “memory location” to indicate a physical memory location.

Each PTX memory instruction also specifies the operation — either a read, a write or an atomic read-modify-write — to be performed on all the bytes in the corresponding memory location.

###  8.2.1. [Overlap](#overlap)

Two memory locations are said to overlap when the starting address of one location is within the range of bytes constituting the other location. Two memory operations are said to overlap when they specify the same virtual address and the corresponding memory locations overlap. The overlap is said to be complete when both memory locations are identical, and it is said to be partial otherwise.

###  8.2.2. [Aliases](#aliases)

Two distinct virtual addresses are said to be aliases if they map to the same memory location.

###  8.2.3. [Multimem Addresses](#multimem-addresses)

A multimem address is a virtual address which points to multiple distinct memory locations across devices.

Only _multimem._ * operations are valid on multimem addresses. That is, the behavior of accessing a multimem address in any other memory operation is undefined.

###  8.2.4. [Memory Operations on Vector Data Types](#memory-operations-on-vector-data-types)

The memory consistency model relates operations executed on memory locations with scalar data types, which have a maximum size and alignment of 64 bits. Memory operations with a vector data type are modelled as a set of equivalent memory operations with a scalar data type, executed in an unspecified order on the elements in the vector.

###  8.2.5. [Memory Operations on Packed Data Types](#memory-operations-on-packed-data-types)

A packed data type consists of two values of the same scalar data type, as described in [Packed Data Types](#packed-data-types). These values are accessed in adjacent memory locations. A memory operation on a packed data type is modelled as a pair of equivalent memory operations on the scalar data type, executed in an unspecified order on each element of the packed data.

###  8.2.6. [Initialization](#initialization)

Each byte in memory is initialized by a hypothetical write _W0_ executed before starting any thread in the program. If the byte is included in a program variable, and that variable has an initial value, then _W0_ writes the corresponding initial value for that byte; else _W0_ is assumed to have written an unknown but constant value to the byte.


##  8.3. [State spaces](#memory-consistency-state-spaces)

The relations defined in the memory consistency model are independent of state spaces. In particular, causality order closes over all memory operations across all the state spaces. But the side-effect of a memory operation in one state space can be observed directly only by operations that also have access to the same state space. This further constrains the synchronizing effect of a memory operation in addition to scope. For example, the synchronizing effect of the PTX instruction `ld.relaxed.shared.sys` is identical to that of `ld.relaxed.shared.cluster`, since no thread outside the same cluster can execute an operation that accesses the same memory location.


##  8.4. [Operation types](#operation-types)

For simplicity, the rest of the document refers to the following operation types, instead of mentioning specific instructions that give rise to them.

Table 20 Operation Types Operation Type | Instruction/Operation  
---|---  
atomic operation | `atom` or `red` instruction.  
read operation | All variants of `ld` instruction and `atom` instruction (but not `red` instruction).  
write operation | All variants of `st` instruction, and _atomic_ operations if they result in a write.  
memory operation | A _read_ or _write_ operation.  
volatile operation | An instruction with `.volatile` qualifier.  
acquire operation | A _memory_ operation with `.acquire` or `.acq_rel` qualifier.  
release operation | A _memory_ operation with `.release` or `.acq_rel` qualifier.  
mmio operation | An `ld` or `st` instruction with `.mmio` qualifier.  
memory fence operation | A `membar`, `fence.sc` or `fence.acq_rel` instruction.  
proxy fence operation | A `fence.proxy` or a `membar.proxy` instruction.  
strong operation | A _memory fence_ operation, or a _memory_ operation with a `.relaxed`, `.acquire`, `.release`, `.acq_rel`, `.volatile`, or `.mmio` qualifier.  
weak operation | An `ld` or `st` instruction with a `.weak` qualifier.  
synchronizing operation | A `barrier` instruction, _fence_ operation, _release_ operation or _acquire_ operation.  
  
###  8.4.1. [mmio Operation](#mmio-operation)

An _mmio_ operation is a memory operation with `.mmio` qualifier specified. It is usually performed on a memory location which is mapped to the control registers of peer I/O devices. It can also be used for communication between threads but has poor performance relative to non-_mmio_ operations.

The semantic meaning of _mmio_ operations cannot be defined precisely as it is defined by the underlying I/O device. For formal specification of semantics of _mmio_ operation from Memory Consistency Model perspective, it is equivalent to the semantics of a _strong_ operation. But it follows a few implementation-specific properties, if it meets the _CUDA atomicity requirements_ at the specified scope:

  * Writes are always performed and are never combined within the scope specified.

  * Reads are always performed, and are not forwarded, prefetched, combined, or allowed to hit any cache within the scope specified.

    * As an exception, in some implementations, the surrounding locations may also be loaded. In such cases the amount of data loaded is implementation specific and varies between 32 and 128 bytes in size.


###  8.4.2. [volatile Operation](#volatile-operation)

A _volatile_ operation is a memory operation with `.volatile` qualifier specified. The semantics of volatile operations are equivalent to a relaxed memory operation with system-scope but with the following extra implementation-specific constraints:

  * The number of volatile _instructions_ (not operations) executed by a program is preserved. Hardware may combine and merge volatile _operations_ issued by multiple different volatile _instructions_ , that is, the number of volatile _operations_ in the program is not preserved.

  * Volatile _instructions_ are not re-ordered around other volatile _instructions_ , but the memory _operations_ performed by those _instructions_ may be re-ordered around each other.


Note

PTX volatile operations are intended for compilers to lower volatile read and write operations from CUDA C++, and other programming languages sharing CUDA C++ volatile semantics, to PTX.

Since volatile operations are relaxed at system-scope with extra constraints, prefer using other _strong_ read or write operations (e.g. `ld.relaxed.sys` or `st.relaxed.sys`) for **Inter-Thread Synchronization** instead, which may deliver better performance.

PTX volatile operations are not suited for **Memory Mapped IO (MMIO)** because volatile operations do not preserve the number of memory operations performed, and may perform more or less operations than requested in a non-deterministic way. Use [.mmio operations](#mmio-operation) instead, which strictly preserve the number of operations performed.


##  8.5. [Scope](#scope)

Each _strong_ operation must specify a _scope_ , which is the set of threads that may interact directly with that operation and establish any of the relations described in the memory consistency model. There are four scopes:

Table 21 Scopes Scope | Description  
---|---  
`.cta` | The set of all threads executing in the same CTA as the current thread.  
`.cluster` | The set of all threads executing in the same cluster as the current thread.  
`.gpu` | The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device.  
`.sys` | The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself.  
  
Note that the warp is not a _scope_ ; the CTA is the smallest collection of threads that qualifies as a _scope_ in the memory consistency model.


##  8.6. [Proxies](#proxies)

A _memory proxy_ , or a _proxy_ is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different _proxies_.

Memory operations as defined in [Operation types](#operation-types) use _generic_ method of memory access, i.e. a _generic proxy_. Other operations such as textures and surfaces all use distinct methods of memory access, also distinct from the _generic_ method.

A _proxy fence_ is required to synchronize memory operations across different _proxies_. Although virtual aliases use the _generic_ method of memory access, since using distinct virtual addresses behaves as if using different _proxies_ , they require a _proxy fence_ to establish memory ordering.


##  8.7. [Morally strong operations](#morally-strong-operations)

Two operations are said to be _morally strong_ relative to each other if they satisfy all of the following conditions:

  1. The operations are related in _program order_ (i.e, they are both executed by the same thread), or each operation is _strong_ and specifies a _scope_ that includes the thread executing the other operation.

  2. Both operations are performed via the same _proxy_.

  3. If both are memory operations, then they overlap completely.


Most (but not all) of the axioms in the memory consistency model depend on relations between _morally strong_ operations.

###  8.7.1. [Conflict and Data-races](#conflict-and-data-races)

Two _overlapping_ memory operations are said to _conflict_ when at least one of them is a _write_.

Two _conflicting_ memory operations are said to be in a _data-race_ if they are not related in _causality order_ and they are not _morally strong_.

###  8.7.2. [Limitations on Mixed-size Data-races](#mixed-size-limitations)

A _data-race_ between operations that _overlap_ completely is called a _uniform-size data-race_ , while a _data-race_ between operations that _overlap_ partially is called a _mixed-size data-race_.

The axioms in the memory consistency model do not apply if a PTX program contains one or more _mixed-size data-races_. But these axioms are sufficient to describe the behavior of a PTX program with only _uniform-size data-races_.

Atomicity of mixed-size RMW operations

In any program with or without _mixed-size data-races_ , the following property holds for every pair of _overlapping atomic_ operations A1 and A2 such that each specifies a _scope_ that includes the other: Either the _read-modify-write_ operation specified by A1 is performed completely before A2 is initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2 overlap partially or completely.


##  8.8. [Release and Acquire Patterns](#release-acquire-patterns)

Some sequences of instructions give rise to patterns that participate in memory synchronization as described later. The _release_ pattern makes prior operations from the current thread1 visible to some operations from other threads. The _acquire_ pattern makes some operations from other threads visible to later operations from the current thread.

A _release_ pattern on a location M consists of one of the following:

  1. A _release_ operation on M

E.g.: `st.release [M];` or `atom.release [M];` or `mbarrier.arrive.release [M];`

  2. Or a _release_ or _acquire-release_ operation on M followed by a _strong_ write on M in _program order_

E.g.: `st.release [M]`; `st.relaxed [M];`

  3. Or a _release_ or _acquire-release_ _memory fence_ followed by a _strong_ write on M in _program order_

E.g.: `fence.release; st.relaxed [M];` or `fence.release; atom.relaxed [M];`


Any _memory synchronization_ established by a _release_ pattern only affects operations occurring in _program order_ before the first instruction in that pattern.

An _acquire_ pattern on a location M consists of one of the following:

  1. An _acquire_ operation on M

E.g.: `ld.acquire [M];` or `atom.acquire [M];` or `mbarrier.test_wait.acquire [M];`

  2. Or a _strong_ read on M followed by an _acquire_ operation on M in _program order_

E.g.: `ld.relaxed [M]; ld.acquire [M];`

  3. Or a _strong_ read on M followed by an acquire _memory fence_ in _program order_

E.g.: `ld.relaxed [M]; fence.acquire;` or `atom.relaxed [M]; fence.acquire;`


Any _memory synchronization_ established by an _acquire_ pattern only affects operations occurring in _program order_ after the last instruction in that pattern.

Note that while atomic reductions conceptually perform a strong read as part of its read-modify-write sequence, this strong read does not form an acquire pattern.

> E.g.: `red.add [M], 1; fence.acquire;` is not an acquire pattern.

1 For both _release_ and _acquire_ patterns, this effect is further extended to operations in other threads through the transitive nature of _causality order_.


##  8.9. [Ordering of memory operations](#ordering-memory-operations)

The sequence of operations performed by each thread is captured as _program order_ while _memory synchronization_ across threads is captured as _causality order_. The visibility of the side-effects of memory operations to other memory operations is captured as _communication order_. The memory consistency model defines contradictions that are disallowed between communication order on the one hand, and _causality order_ and _program order_ on the other.

###  8.9.1. [Program Order](#program-order)

The _program order_ relates all operations performed by a thread to the order in which a sequential processor will execute instructions in the corresponding PTX source. It is a transitive relation that forms a total order over the operations performed by the thread, but does not relate operations from different threads.

####  8.9.1.1. [Asynchronous Operations](#program-order-async-operations)

Some PTX instructions (all variants of `cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`, `wgmma.mma_async`) perform operations that are asynchronous to the thread that executed the instruction. These asynchronous operations are ordered after prior instructions in the same thread (except in the case of `wgmma.mma_async`), but they are not part of the program order for that thread. Instead, they provide weaker ordering guarantees as documented in the instruction description.

For example, the loads and stores performed as part of a `cp.async` are ordered with respect to each other, but not to those of any other `cp.async` instructions initiated by the same thread, nor any other instruction subsequently issued by the thread with the exception of `cp.async.commit_group` or `cp.async.mbarrier.arrive`. The asynchronous mbarrier [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation performed by a `cp.async.mbarrier.arrive` instruction is ordered with respect to the memory operations performed by all prior `cp.async` operations initiated by the same thread, but not to those of any other instruction issued by the thread. The implicit mbarrier [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation) operation that is part of all variants of `cp.async.bulk` and `cp.reduce.async.bulk` instructions is ordered only with respect to the memory operations performed by the same asynchronous instruction, and in particular it does not transitively establish ordering with respect to prior instructions from the issuing thread.

###  8.9.2. [Observation Order](#observation-order)

_Observation order_ relates a write W to a read R through an optional sequence of atomic read-modify-write operations.

A write W precedes a read R in _observation order_ if:

  1. R and W are _morally strong_ and R reads the value written by W, or

  2. For some atomic operation Z, W precedes Z and Z precedes R in _observation order_.


###  8.9.3. [Fence-SC Order](#fence-sc-order)

The _Fence-SC_ order is an acyclic partial order, determined at runtime, that relates every pair of _morally strong fence.sc_ operations.

###  8.9.4. [Memory synchronization](#memory-synchronization)

Synchronizing operations performed by different threads synchronize with each other at runtime as described here. The effect of such synchronization is to establish _causality order_ across threads.

  1. A `fence.sc` operation X _synchronizes_ with a `fence.sc` operation Y if X precedes Y in the _Fence-SC_ order.

  2. A `bar{.cta}.sync` or `bar{.cta}.red` or `bar{.cta}.arrive` operation _synchronizes_ with a `bar{.cta}.sync` or `bar{.cta}.red` operation executed on the same barrier.

  3. A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation.

  4. A _release_ pattern X _synchronizes_ with an _acquire_ pattern Y, if a _write_ operation in X precedes a _read_ operation in Y in _observation order_ , and the first operation in X and the last operation in Y are _morally strong_.


API synchronization

A _synchronizes_ relation can also be established by certain CUDA APIs.

  1. Completion of a task enqueued in a CUDA stream _synchronizes_ with the start of the following task in the same stream, if any.

  2. For purposes of the above, recording or waiting on a CUDA event in a stream, or causing a cross-stream barrier to be inserted due to `cudaStreamLegacy`, enqueues tasks in the associated streams even if there are no direct side effects. An event record task _synchronizes_ with matching event wait tasks, and a barrier arrival task _synchronizes_ with matching barrier wait tasks.

  3. Start of a CUDA kernel _synchronizes_ with start of all threads in the kernel. End of all threads in a kernel _synchronize_ with end of the kernel.

  4. Start of a CUDA graph _synchronizes_ with start of all source nodes in the graph. Completion of all sink nodes in a CUDA graph _synchronizes_ with completion of the graph. Completion of a graph node _synchronizes_ with start of all nodes with a direct dependency.

  5. Start of a CUDA API call to enqueue a task _synchronizes_ with start of the task.

  6. Completion of the last task queued to a stream, if any, _synchronizes_ with return from `cudaStreamSynchronize`. Completion of the most recently queued matching event record task, if any, _synchronizes_ with return from `cudaEventSynchronize`. Synchronizing a CUDA device or context behaves as if synchronizing all streams in the context, including ones that have been destroyed.

  7. Returning `cudaSuccess` from an API to query a CUDA handle, such as a stream or event, behaves the same as return from the matching synchronization API.


In addition to establishing a _synchronizes_ relation, the CUDA API synchronization mechanisms above also participate in _proxy-preserved base causality order_ except for the _tensormap-proxy_ which is not acquired from _generic-proxy_ at CUDA Kernel start and must therefore be acquired explicitly using `fence.proxy.tensormap::generic.acquire` when needed.

###  8.9.5. [Causality Order](#causality-order)

_Causality order_ captures how memory operations become visible across threads through synchronizing operations. The axiom “Causality” uses this order to constrain the set of write operations from which a read operation may read a value.

Relations in the _causality order_ primarily consist of relations in _Base causality order_ 1 , which is a transitive order, determined at runtime.

Base causality order

An operation X precedes an operation Y in _base causality order_ if:

  1. X precedes Y in _program order_ , or

  2. X _synchronizes_ with Y, or

  3. For some operation Z,

     1. X precedes Z in _program order_ and Z precedes Y in _base causality order_ , or

     2. X precedes Z in _base causality order_ and Z precedes Y in _program order_ , or

     3. X precedes Z in _base causality order_ and Z precedes Y in _base causality order_.


Proxy-preserved base causality order

A memory operation X precedes a memory operation Y in _proxy-preserved base causality order_ if X precedes Y in _base causality order_ , and:

  1. X and Y are performed to the same address, using the _generic proxy_ , or

  2. X and Y are performed to the same address, using the same _proxy_ , and by the same thread block, or

  3. X and Y are aliases and there is an alias _proxy fence_ along the base causality path from X to Y.


Causality order

_Causality order_ combines _base causality order_ with some non-transitive relations as follows:

An operation X precedes an operation Y in _causality order_ if:

  1. X precedes Y in _proxy-preserved base causality order_ , or

  2. For some operation Z, X precedes Z in observation order, and Z precedes Y in _proxy-preserved base causality order_.


1 The transitivity of _base causality order_ accounts for the “cumulativity” of synchronizing operations.

###  8.9.6. [Coherence Order](#coherence-order)

There exists a partial transitive order that relates _overlapping_ write operations, determined at runtime, called the _coherence order_ 1. Two _overlapping_ write operations are related in _coherence order_ if they are _morally strong_ or if they are related in _causality order_. Two _overlapping_ writes are unrelated in _coherence order_ if they are in a _data-race_ , which gives rise to the partial nature of _coherence order_.

1 _Coherence order_ cannot be observed directly since it consists entirely of write operations. It may be observed indirectly by its use in constraining the set of candidate writes that a read operation may read from.

###  8.9.7. [Communication Order](#communication-order)

The _communication order_ is a non-transitive order, determined at runtime, that relates write operations to other _overlapping_ memory operations.

  1. A write W precedes an _overlapping_ read R in _communication order_ if R returns the value of any byte that was written by W.

  2. A write W precedes a write W’ in _communication order_ if W precedes W’ in _coherence order_.

  3. A read R precedes an _overlapping_ write W in _communication order_ if, for any byte accessed by both R and W, R returns the value written by a write W’ that precedes W in _coherence order_.


_Communication order_ captures the visibility of memory operations — when a memory operation X1 precedes a memory operation X2 in _communication order_ , X1 is said to be visible to X2.


##  8.10. [Axioms](#axioms)

###  8.10.1. [Coherence](#coherence-axiom)

If a write W precedes an _overlapping_ write W’ in _causality order_ , then W must precede W’ in _coherence order_.

###  8.10.2. [Fence-SC](#fence-sc-axiom)

_Fence-SC_ order cannot contradict _causality order_. For a pair of _morally strong_ _fence.sc_ operations F1 and F2, if F1 precedes F2 in _causality order_ , then F1 must precede F2 in _Fence-SC_ order.

###  8.10.3. [Atomicity](#atomicity-axiom)

Single-Copy Atomicity

Conflicting _morally strong_ operations are performed with _single-copy atomicity_. When a read R and a write W are _morally strong_ , then the following two communications cannot both exist in the same execution, for the set of bytes accessed by both R and W:

  1. R reads any byte from W.

  2. R reads any byte from any write W’ which precedes W in _coherence order_.


Atomicity of read-modify-write (RMW) operations

When an _atomic_ operation A and a write W _overlap_ and are _morally strong_ , then the following two communications cannot both exist in the same execution, for the set of bytes accessed by both A and W:

  1. A reads any byte from a write W’ that precedes W in _coherence order_.

  2. A follows W in _coherence order_.


Litmus Test 1
    
    
    .global .u32 x = 0;
      
  
---  
T1 | T2  
      
    
    A1: atom.sys.inc.u32 %r0, [x];
    

| 
    
    
    A2: atom.sys.inc.u32 %r0, [x];
      
      
    
    FINAL STATE: x == 2
      
  
Atomicity is guaranteed when the operations are _morally strong_.

Litmus Test 2
    
    
    .global .u32 x = 0;
      
  
---  
T1 | T2 (In a different CTA)  
      
    
    A1: atom.cta.inc.u32 %r0, [x];
    

| 
    
    
    A2: atom.gpu.inc.u32 %r0, [x];
      
      
    
    FINAL STATE: x == 1 OR x == 2
      
  
Atomicity is not guaranteed if the operations are not _morally strong_.

###  8.10.4. [No Thin Air](#no-thin-air-axiom)

Values may not appear “out of thin air”: an execution cannot speculatively produce a value in such a way that the speculation becomes self-satisfying through chains of instruction dependencies and inter-thread communication. This matches both programmer intuition and hardware reality, but is necessary to state explicitly when performing formal analysis.

Litmus Test: Load Buffering with true dependencies
    
    
    .global .u32 x = 0;
    .global .u32 y = 0;
      
  
---  
T1 | T2  
      
    
    A1: ld.global.u32 %r0, [x];
    B1: st.global.u32 [y], %r0;
    

| 
    
    
    A2: ld.global.u32 %r1, [y];
    B2: st.global.u32 [x], %r1;
      
      
    
    FINAL STATE: x == 0 AND y == 0
      
  
The litmus test known as “LB+deps” (Load Buffering with dependencies) checks such forbidden values that may arise out of thin air. Two threads T1 and T2 each read from a first variable and copy the observed result into a second variable, with the first and second variable exchanged between the threads. If each variable is initially zero, the final result shall also be zero. If A1 reads from B2 and A2 reads from B1, then values passing through the memory operations in this example form a cycle: A1->B1->A2->B2->A1. Only the values x == 0 and y == 0 are allowed to satisfy this cycle. If any of the memory operations in this example were to speculatively associate a different value with the corresponding memory location, then such a speculation would become self-fulfilling, and hence forbidden.

Litmus Test: Load Buffering without dependencies
    
    
    .global .u32 x = 0;
    .global .u32 y = 0;
      
  
---  
T1 | T2  
      
    
    A1: ld.global.u32 %r0, [x];
    B1: st.global.u32 [y], 1;
    

| 
    
    
    A2: ld.global.u32 %r1, [y];
    B2: st.global.u32 [x], 1;
      
      
    
    FINAL STATE: x == 1 AND y == 1
      
  
This litmus test differs from the one above in that it unconditionally stores 1 to x and y. In this litmus test a final state of x == 1 and y == 1 is permitted. This execution does not contradict the requirement demonstrated by the previous litmus test. Here there is no self-fulfilling cycle – the litmus test will always and unconditionally store 1 to x and y, so here the cycle is not self-fulfilled and the speculation is valid.

Here the lack of dependencies is plain, but the implementation may perform any chain of reasoning to determine that a store is not dependent on a prior load, and thus break self-fulfilling cycles which would otherwise apparently be forbidden by the No-Thin-Air axiom.

This form of load buffering is deliberately permitted in the PTX memory consistency model.

###  8.10.5. [Sequential Consistency Per Location](#sc-per-loc-axiom)

Within any set of _overlapping_ memory operations that are pairwise _morally strong_ , _communication order_ cannot contradict _program order_ , i.e., a concatenation of _program order_ between _overlapping_ operations and _morally strong_ relations in _communication order_ cannot result in a cycle. This ensures that each program slice of _overlapping_ pairwise morally _strong operations_ is strictly _sequentially-consistent_.

Litmus Test: CoRR
    
    
    .global .u32 x = 0;
      
  
---  
T1 | T2  
      
    
    W1: st.global.relaxed.sys.u32 [x], 1;
    

| 
    
    
    R1: ld.global.relaxed.sys.u32 %r0, [x];
    R2: ld.global.relaxed.sys.u32 %r1, [x];
      
      
    
    IF %r0 == 1 THEN %r1 == 1
      
  
The litmus test “CoRR” (Coherent Read-Read), demonstrates one consequence of this guarantee. A thread T1 executes a write W1 on a location x, and a thread T2 executes two (or an infinite sequence of) reads R1 and R2 on the same location x. No other writes are executed on x, except the one modelling the initial value. The operations W1, R1 and R2 are pairwise _morally strong_. If R1 reads from W1, then the subsequent read R2 must also observe the same value. If R2 observed the initial value of x instead, then this would form a sequence of _morally-strong_ relations R2->W1->R1 in _communication order_ that contradicts the _program order_ R1->R2 in thread T2. Hence R2 cannot read the initial value of x in such an execution.

###  8.10.6. [Causality](#causality-axiom)

Relations in _communication order_ cannot contradict _causality order_. This constrains the set of candidate write operations that a read operation may read from:

  1. If a read R precedes an _overlapping_ write W in _causality order_ , then R cannot read from W.

  2. If a write W precedes an _overlapping_ read R in _causality order_ , then for any byte accessed by both R and W, R cannot read from any write W’ that precedes W in _coherence order_.


Litmus Test: Message Passing
    
    
    .global .u32 data = 0;
    .global .u32 flag = 0;
      
  
---  
T1 | T2  
      
    
    W1: st.global.u32 [data], 1;
    F1: fence.sys;
    W2: st.global.relaxed.sys.u32 [flag], 1;
    

| 
    
    
    R1: ld.global.relaxed.sys.u32 %r0, [flag];
    F2: fence.sys;
    R2: ld.global.u32 %r1, [data];
      
      
    
    IF %r0 == 1 THEN %r1 == 1
      
  
The litmus test known as “MP” (Message Passing) represents the essence of typical synchronization algorithms. A vast majority of useful programs can be reduced to sequenced applications of this pattern.

Thread T1 first writes to a data variable and then to a flag variable while a second thread T2 first reads from the flag variable and then from the data variable. The operations on the flag are _morally strong_ and the memory operations in each thread are separated by a _fence_ , and these _fences_ are _morally strong_.

If R1 observes W2, then the release pattern “F1; W2” _synchronizes_ with the _acquire pattern_ “R1; F2”. This establishes the _causality order_ W1 -> F1 -> W2 -> R1 -> F2 -> R2. Then axiom _causality_ guarantees that R2 cannot read from any write that precedes W1 in _coherence order_. In the absence of any other writes in this example, R2 must read from W1.

Litmus Test: CoWR
    
    
    // These addresses are aliases
    .global .u32 data_alias_1;
    .global .u32 data_alias_2;
      
  
---  
T1  
      
    
    W1: st.global.u32 [data_alias_1], 1;
    F1: fence.proxy.alias;
    R1: ld.global.u32 %r1, [data_alias_2];
      
      
    
    %r1 == 1
      
  
Virtual aliases require an alias _proxy fence_ along the synchronization path.

Litmus Test: Store Buffering

The litmus test known as “SB” (Store Buffering) demonstrates the _sequential consistency_ enforced by the `fence.sc`. A thread T1 writes to a first variable, and then reads the value of a second variable, while a second thread T2 writes to the second variable and then reads the value of the first variable. The memory operations in each thread are separated by `fence.`sc instructions, and these _fences_ are _morally strong_.
    
    
    .global .u32 x = 0;
    .global .u32 y = 0;
      
  
---  
T1 | T2  
      
    
    W1: st.global.u32 [x], 1;
    F1: fence.sc.sys;
    R1: ld.global.u32 %r0, [y];
    

| 
    
    
    W2: st.global.u32 [y], 1;
    F2: fence.sc.sys;
    R2: ld.global.u32 %r1, [x];
      
      
    
    %r0 == 1 OR %r1 == 1
      
  
In any execution, either F1 precedes F2 in _Fence-SC_ order, or vice versa. If F1 precedes F2 in _Fence-SC_ order, then F1 _synchronizes_ with F2. This establishes the _causality order_ in W1 -> F1 -> F2 -> R2. Axiom _causality_ ensures that R2 cannot read from any write that precedes W1 in _coherence order_. In the absence of any other write to that variable, R2 must read from W1. Similarly, in the case where F2 precedes F1 in _Fence-SC_ order, R1 must read from W2. If each `fence.sc` in this example were replaced by a `fence.acq_rel` instruction, then this outcome is not guaranteed. There may be an execution where the write from each thread remains unobserved from the other thread, i.e., an execution is possible, where both R1 and R2 return the initial value “0” for variables y and x respectively.


##  8.11. [Special Cases](#special-cases)

###  8.11.1. [Reductions do not form Acquire Patterns](#red-read)

Atomic reduction operations like `red` do not form acquire patterns with acquire fences.

**Litmus Test: Message Passing with a Red Instruction**
    
    
    .global .u32 x = 0;
    .global .u32 flag = 0;
      
  
---  
T1 | T2  
      
    
    W1: st.u32 [x], 42;
    W2: st.release.gpu.u32 [flag], 1;
    

| 
    
    
    RMW1: red.sys.global.add.u32 [flag], 1;
    F2: fence.acquire.gpu;
    R2: ld.weak.u32 %r1, [x];
      
      
    
    %r1 == 0 AND flag == 2
      
  
The litmus test known as “MP” (Message Passing) demonstrates the consequence of reductions being excluded from acquire patterns. It is possible to observe the outcome where `R2` reads the value `0` from `x` and `flag` has the final value of `2`. This outcome is possible since the release pattern in `T1` does not synchronize with any acquire pattern in `T2`. Using the `atom` instruction instead of `red` forbids this outcome.
