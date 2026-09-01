# cutlass.pipeline

### *class* cutlass.pipeline.Agent(\*values)

Bases: `Enum`

Agent indicates what is participating in the pipeline synchronization.

#### Thread *= 1*

#### ThreadBlock *= 2*

#### ThreadBlockCluster *= 3*

### *class* cutlass.pipeline.CooperativeGroup(agent: [Agent](#cutlass.pipeline.Agent), size: int = 1, alignment=None)

Bases: `object`

CooperativeGroup contains size and alignment restrictions for an Agent.

#### \_\_init_\_(agent: [Agent](#cutlass.pipeline.Agent), size: int = 1, alignment=None)

### *class* cutlass.pipeline.MbarrierArray

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

MbarrierArray implements an abstraction for an array of smem barriers.

#### \_\_init_\_() → None

#### \_abc_impl *= <_abc._abc_data object>*

#### arrive(index: int, dst: int, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) | None = None, , loc=None, ip=None) → None

Select the arrive corresponding to this MbarrierArray’s PipelineOp.

* **Parameters:**
  * **index** (*int*) – Index of the mbarrier in the array to arrive on
  * **dst** (*int* *|* *None*) – Destination parameter for selective arrival, which can be either a mask or destination cta rank.
    When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
    - For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
    in the cluster with rank = 0, 1, and 3).
    - For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
    the mbarrier with rank = 3 in the cluster).
  * **cta_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) – CTA group for `TCGen05Mma`, defaults to None for other op types

#### arrive_and_drop(, loc=None, ip=None) → None

#### arrive_and_expect_tx(index: int, tx_count: int, , loc=None, ip=None) → None

#### arrive_and_expect_tx_with_dst(index: int, tx_count: int, dst: int | None = None, , loc=None, ip=None) → None

#### arrive_and_wait(index: int, phase: int, dst: int, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) | None = None, , loc=None, ip=None) → None

#### arrive_cp_async_mbarrier(index: int, , loc=None, ip=None)

#### arrive_mbarrier(index: int, dst_rank: int | None = None, , loc=None, ip=None) → None

#### arrive_tcgen05mma(index: int, mask: int | None, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup), , loc=None, ip=None) → None

#### get_barrier(index: int, , loc=None, ip=None) → Pointer

#### max() → int

#### mbarrier_init(, loc=None, ip=None) → None

Initializes an array of mbarriers using warp 0.

#### recast_to_new_op_type(new_op_type: [PipelineOp](#cutlass.pipeline.PipelineOp)) → [MbarrierArray](#cutlass.pipeline.MbarrierArray)

Creates a copy of MbarrierArray with a different op_type without re-initializing barriers

#### try_wait(index: int, phase: int, , loc=None, ip=None) → Boolean

#### wait(index: int, phase: int, , loc=None, ip=None) → None

### *class* cutlass.pipeline.NamedBarrier(barrier_id: int, num_threads: int)

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

NamedBarrier is an abstraction for named barriers managed by hardware.
There are 16 named barriers available, with barrier_ids 0-15.

See the [PTX documentation](https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

#### \_\_init_\_(barrier_id: int, num_threads: int) → None

#### \_abc_impl *= <_abc._abc_data object>*

#### arrive(, loc=None, ip=None) → None

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

#### arrive_and_drop(, loc=None, ip=None) → None

#### arrive_and_wait(, loc=None, ip=None) → None

#### arrive_unaligned(, loc=None, ip=None) → None

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

#### barrier_id *: int*

#### get_barrier(, loc=None, ip=None) → int

#### max() → int

#### num_threads *: int*

#### sync(, loc=None, ip=None) → None

#### wait(, loc=None, ip=None) → None

NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

#### wait_unaligned(, loc=None, ip=None) → None

### *class* cutlass.pipeline.PipelineAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None)

Bases: `object`

PipelineAsync is a generic pipeline class where both the producer and consumer are
AsyncThreads. It also serves as a base class for specialized pipeline classes.

This class implements a producer-consumer pipeline pattern where both sides operate
asynchronously. The pipeline maintains synchronization state using barrier objects
to coordinate between producer and consumer threads.

The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

#### Pipeline State Transitions

| Barrier   | State   | p.acquire   | p.commit   | c.wait   | c.release   |
|-----------|---------|-------------|------------|----------|-------------|
| empty_bar | empty   | <Return>    | n/a        | n/a      |             |
| empty_bar | wait    | <Block>     | n/a        | n/a      | -> empty    |
| full_bar  | wait    | n/a         | -> full    | <Block > | n/a         |
| full_bar  | full    | n/a         |            | <Return> | n/a         |

Where:

- p: producer
- c: consumer
- <Block>: This action is blocked until transition to a state allow it to proceed by other side
  - e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

```text
Array of mbarriers as circular buffer:

     Advance Direction
   <-------------------

    Producer   Consumer
        |         ^
        V         |
   +-----------------+
 --|X|X|W|D|D|D|D|R|X|<-.
/  +-----------------+   \
|                        |
`------------------------'
```

Where:

- X: Empty buffer (initial state)
- W: Producer writing (producer is waiting for buffer to be empty)
- D: Data ready (producer has written data to buffer)
- R: Consumer reading (consumer is consuming data from buffer)

**Example:**

```python
# Create pipeline with 5 stages
pipeline = PipelineAsync.create(
    num_stages=5,                   # number of pipeline stages
    producer_group=producer_warp,
    consumer_group=consumer_warp
    barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
)

producer, consumer = pipeline.make_participants()
# Producer side
for i in range(num_iterations):
    handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
    # Write data to pipeline buffer
    handle.commit()   # Signal buffer is full

# Consumer side
for i in range(num_iterations):
    handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
    # Read data from pipeline buffer
    handle.release()  # Signal buffer is empty
```

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None) → None

#### *static* \_make_sync_object(barrier_storage: Pointer, num_stages: int, agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup)], tx_count: int = 0) → [SyncObject](#cutlass.pipeline.SyncObject)

Returns a SyncObject corresponding to an agent’s PipelineOp.

#### consumer_mask *: Int32 | None*

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

#### consumer_try_wait(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

#### consumer_wait(state: [PipelineState](#cutlass.pipeline.PipelineState), try_wait_token: Boolean | None = None, , loc=None, ip=None)

#### *static* create(, num_stages: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), barrier_storage: Pointer = None, producer_mask: Int32 = None, consumer_mask: Int32 = None, defer_sync: bool = False)

Creates and initializes a new PipelineAsync instance.

This helper function computes necessary attributes and returns an instance of PipelineAsync
with the specified configuration for producer and consumer synchronization.

* **Parameters:**
  * **barrier_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
  * **producer_mask** (*Int32* *,* *optional*) – Mask for signaling arrives for the producer agent
  * **consumer_mask** (*Int32* *,* *optional*) – Mask for signaling arrives for the consumer agent
* **Raises:**
  **ValueError** – If barrier_storage is not a cute.Pointer instance
* **Returns:**
  A new `PipelineAsync` instance
* **Return type:**
  [PipelineAsync](#cutlass.pipeline.PipelineAsync)

#### make_consumer(, loc=None, ip=None)

#### make_participants(, loc=None, ip=None)

#### make_producer(, loc=None, ip=None)

#### num_stages *: int*

#### producer_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

#### producer_commit(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

#### producer_get_barrier(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None) → Pointer

#### producer_mask *: Int32 | None*

#### producer_tail(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

Make sure the last used buffer empty signal is visible to producer.
Producer tail is usually executed by producer before exit, to avoid dangling
mbarrier arrive signals after kernel exit.

* **Parameters:**
  **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – The pipeline state that points to next useful buffer

#### producer_try_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

#### sync_object_empty *: [SyncObject](#cutlass.pipeline.SyncObject)*

#### sync_object_full *: [SyncObject](#cutlass.pipeline.SyncObject)*

### *class* cutlass.pipeline.PipelineAsyncUmma(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup))

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) → None

#### \_compute_is_leader_cta(, loc=None, ip=None)

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

#### \_compute_leading_cta_rank(, loc=None, ip=None)

Computes the leading CTA rank.

#### \_compute_peer_cta_mask(, loc=None, ip=None)

Computes a mask for signaling arrivals to multicasting threadblocks.

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

UMMA consumer release buffer empty, cta_group needs to be provided.

#### create(, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), barrier_storage: Pointer = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, defer_sync: bool = False, loc=None, ip=None)

Creates and initializes a new PipelineAsyncUmma instance.

* **Parameters:**
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the consumer agent
  * **barrier_storage** (*cute.Pointer* *,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
  * **cta_layout_vmnk** ([*cute.Layout*](cute.md#cutlass.cute.Layout) *,* *optional*) – Layout of the cluster shape
* **Raises:**
  **ValueError** – If barrier_storage is not a cute.Pointer instance
* **Returns:**
  A new PipelineAsyncUmma instance configured with the provided parameters
* **Return type:**
  [PipelineAsyncUmma](#cutlass.pipeline.PipelineAsyncUmma)

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)*

### *class* cutlass.pipeline.PipelineClcFetchAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_signalling_thread: Boolean)

Bases: `object`

PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
Control based dynamic scheduling. Both producer and consumer operate asynchronously
using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

- Producer: waits for empty buffer, signals full barrier with transection bytes
  across all CTAs in cluster, hardware autosignals each CTA’s mbarrier when
  transaction bytes are written, then the satte advance to next buffer slot.
- Consumer: waits for full barrier, then load respinse from local SMEM, then
  sigals CTA 0’s empty barrier to allow buffer reuse.

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_signalling_thread: Boolean) → None

#### *static* \_init_full_barrier_arrive_signal(cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout), tidx: Int32)

Computes producer barrier signaling parameters, returns destination CTA rank
(0 to cluster_size-1) based on thread ID, and a boolean flag indicating if
this thread participates in signaling.

* **Parameters:**
  * **cta_layout_vmnk** – Cluster layout defining CTA count
  * **tidx** – Thread ID within the CTA

#### consumer_mask *: Int32 | None*

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

#### consumer_wait(state: [PipelineState](#cutlass.pipeline.PipelineState), try_wait_token: Boolean | None = None, , loc=None, ip=None)

Consumer waits for full barrier to be signaled by hardware multicast.

* **Parameters:**
  * **state** – Pipeline state pointing to the current buffer stage
  * **try_wait_token** – Optional token to skip the full barrier wait

#### *static* create(, num_stages: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), tx_count: int, barrier_storage: Pointer = None, producer_mask: Int32 = None, consumer_mask: Int32 = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, defer_sync: bool = False)

This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
:param barrier_storage: Pointer to the shared memory address for this pipeline’s mbarriers
:type barrier_storage: cute.Pointer
:param num_stages: Number of buffer stages for this pipeline
:type num_stages: int
:param producer_group: CooperativeGroup for the producer agent
:type producer_group: CooperativeGroup
:param consumer_group: CooperativeGroup for the consumer agent
:type consumer_group: CooperativeGroup
:param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx_count: int
:param producer_mask: Mask for signaling arrives for the producer agent, defaults to `None`
:type producer_mask: Int32, optional
:param consumer_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
:type consumer_mask: Int32, optional

#### is_signalling_thread *: Boolean*

#### num_stages *: int*

#### producer_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

* **Parameters:**
  * **state** – Pipeline state pointing to the current buffer stage
  * **try_acquire_token** – Optional token to skip the empty barrier wait

#### producer_get_barrier(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None) → Pointer

#### producer_mask *: Int32 | None*

#### producer_tail(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

Ensures all in-flight buffers are released before producer exits.

* **Parameters:**
  * **state** – Pipeline state with current position in the buffer
  * **try_acquire_token** – Optional token to skip the empty barrier waits

#### sync_object_empty *: [SyncObject](#cutlass.pipeline.SyncObject)*

#### sync_object_full *: [SyncObject](#cutlass.pipeline.SyncObject)*

### *class* cutlass.pipeline.PipelineConsumer(pipeline, state: [PipelineState](#cutlass.pipeline.PipelineState), group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup))

Bases: `object`

A class representing a consumer in an asynchronous pipeline.

The Consumer class manages the consumer side of an asynchronous pipeline, handling
synchronization and state management for consuming data. It provides methods for
waiting, releasing, and advancing through pipeline stages.

* **Variables:**
  * **\_\_pipeline** – The asynchronous pipeline this consumer belongs to
  * **\_\_state** – The current state of the consumer in the pipeline
  * **\_\_group** – The cooperative group this consumer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to wait for buffer to be full
    try_wait_token = consumer.try_wait()

    # Do something else independently
    ...

    # Wait for buffer to be full & Move index to next stage
    # If try_wait_token is True, return immediately
    # If try_wait_token is False, block until buffer is full
    handle = consumer.wait_and_advance(try_wait_token)

    # Consume data
    handle.release(  )  # Signal buffer is empty

    # Alternative way to do this is:
    # handle.release()  # Signal buffer is empty
```

#### *class* ImmutableResourceHandle(\_ImmutableResourceHandle_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync), \_ImmutableResourceHandle_\_immutable_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState))

Bases: `ImmutableResourceHandle`

#### \_\_init_\_(\_ImmutableResourceHandle_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync), \_ImmutableResourceHandle_\_immutable_state: [PipelineState](#cutlass.pipeline.PipelineState)) → None

#### release(, loc=None, ip=None)

Signal that data production is complete for the current stage.
This allows consumers to start processing the data.

#### \_\_group *: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup)*

#### \_\_init_\_(pipeline, state: [PipelineState](#cutlass.pipeline.PipelineState), group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup))

Initialize a new Consumer instance.

* **Parameters:**
  * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync)) – The pipeline this consumer belongs to
  * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – Initial pipeline state
  * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – The cooperative group for synchronization

#### \_\_pipeline *: [PipelineAsync](#cutlass.pipeline.PipelineAsync)*

#### \_\_state *: [PipelineState](#cutlass.pipeline.PipelineState)*

#### advance(, loc=None, ip=None)

Advance the consumer to the next pipeline stage.

This updates the internal state to point to the next buffer in the pipeline.
Should be called after consuming data from the current buffer.

#### clone()

Create a new Consumer instance with the same state.

#### release(handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle) | None = None, , loc=None, ip=None)

Signal that data consumption is complete for the current stage.
This allows producers to start producing new data.

#### reset(, loc=None, ip=None)

Reset the count of how many handles this consumer has consumed.

#### try_wait(, loc=None, ip=None) → Boolean

Non-blocking check if data is ready in the current buffer.

This method provides a way to test if data is available without blocking.
Unlike wait(), this will return immediately regardless of buffer state.

* **Returns:**
  True if data is ready to be consumed, False if the buffer is not yet ready
* **Return type:**
  Boolean

#### wait(try_wait_token: Boolean | None = None, , loc=None, ip=None) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

Wait for data to be ready in the current buffer. This is a blocking operation
that will not return until data is available.

* **Parameters:**
  **try_wait_token** (*Optional* *[**Boolean* *]*) – Token used to attempt a non-blocking wait for the buffer.
  If provided and True, returns immediately if buffer is not ready.
* **Returns:**
  An immutable handle to the consumer that can be used to release the buffer
  once data consumption is complete
* **Return type:**
  [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

#### wait_and_advance(try_wait_token: Boolean | None = None, , loc=None, ip=None) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

Atomically wait for data and advance to next pipeline stage.

This is a convenience method that combines wait() and advance() into a single
atomic operation. It will block until data is available in the current buffer,
then automatically advance to the next stage.

* **Parameters:**
  **try_wait_token** (*Optional* *[**Boolean* *]*) – Token used to attempt a non-blocking wait for the buffer.
  If provided and True, returns immediately if buffer is not ready.
* **Returns:**
  An immutable handle to the consumer that can be used to release the buffer
  once data consumption is complete
* **Return type:**
  [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

### *class* cutlass.pipeline.PipelineCpAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None)

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineCpAsync is used for CpAsync producers and AsyncThread consumers

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None) → None

#### *static* create(barrier_storage: Pointer, num_stages: Int32, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), producer_mask: Int32 = None, consumer_mask: Int32 = None, defer_sync: bool = False)

Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

* **Parameters:**
  * **barrier_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
  * **num_stages** (*Int32*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
  * **producer_mask** (*Int32* *,* *optional*) – Mask for signaling arrives for the producer agent, defaults to None
  * **consumer_mask** (*Int32* *,* *optional*) – Mask for signaling arrives for the consumer agent, defaults to None
* **Returns:**
  A new `PipelineCpAsync` instance configured with the provided parameters
* **Return type:**
  [PipelineCpAsync](#cutlass.pipeline.PipelineCpAsync)

### *class* cutlass.pipeline.PipelineOp(\*values)

Bases: `Enum`

PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

#### AsyncLoad *= 7*

#### AsyncThread *= 1*

#### ClcLoad *= 4*

#### Composite *= 6*

#### TCGen05Mma *= 2*

#### TmaLoad *= 3*

#### TmaStore *= 5*

### *class* cutlass.pipeline.PipelineOrder(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), depth: int, length: int, group_id: int, state: [PipelineState](#cutlass.pipeline.PipelineState))

Bases: `object`

PipelineOrder is used for managing ordered pipeline execution with multiple groups.

This class implements a pipeline ordering mechanism where work is divided into groups
and stages, allowing for controlled progression through pipeline stages with proper
synchronization between different groups.

The pipeline ordering works as follows:
- The pipeline is divided into ‘length’ number of groups
- Each group has ‘depth’ number of stages
- Groups execute in a specific order with synchronization barriers
- Each group waits for the previous group to complete before proceeding

**Example:**

```python
# Create pipeline order with 3 groups, each with 2 stages
pipeline_order = PipelineOrder.create(
    barrier_storage=smem_ptr,      # shared memory pointer for barriers
    depth=2,                       # 2 stages per group
    length=3,                      # 3 groups total
    group_id=0,                    # current group ID (0, 1, or 2)
    producer_group=producer_warp   # cooperative group for producers
)

# In the pipeline loop
for stage in range(num_stages):
    pipeline_order.wait()          # Wait for previous group to complete
    # Process current stage
    pipeline_order.arrive()        # Signal completion to next group
```

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), depth: int, length: int, group_id: int, state: [PipelineState](#cutlass.pipeline.PipelineState)) → None

#### arrive(, loc=None, ip=None)

#### *static* create(barrier_storage: Pointer, depth: int, length: int, group_id: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), defer_sync: bool = False)

#### depth *: int*

#### get_barrier_for_current_stage_idx(group_id)

#### group_id *: int*

#### length *: int*

#### state *: [PipelineState](#cutlass.pipeline.PipelineState)*

#### sync_object_full *: [SyncObject](#cutlass.pipeline.SyncObject)*

#### wait(, loc=None, ip=None)

### *class* cutlass.pipeline.PipelineProducer(pipeline, state, group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup))

Bases: `object`

A class representing a producer in an asynchronous pipeline.

This class manages the producer side of an asynchronous pipeline, handling
synchronization and state management for producing data. It provides methods for
acquiring, committing, and advancing through pipeline stages.

* **Variables:**
  * **\_\_pipeline** – The asynchronous pipeline this producer belongs to
  * **\_\_state** – The current state of the producer in the pipeline
  * **\_\_group** – The cooperative group this producer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to acquire the current buffer without blocking
    try_acquire_token = producer.try_acquire()

    # Do something else independently
    ...

    # Wait for current buffer to be empty & Move index to next stage
    # If try_acquire_token is True, return immediately
    # If try_acquire_token is False, block until buffer is empty
    handle = producer.acquire_and_advance(try_acquire_token)

    # Produce data
    handle.commit()
```

#### *class* ImmutableResourceHandle(\_ImmutableResourceHandle_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync), \_ImmutableResourceHandle_\_immutable_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState))

Bases: `ImmutableResourceHandle`

#### \_\_init_\_(\_ImmutableResourceHandle_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync), \_ImmutableResourceHandle_\_immutable_state: [PipelineState](#cutlass.pipeline.PipelineState)) → None

#### *property* barrier

Get the barrier pointer for the current pipeline stage.

* **Returns:**
  Pointer to the barrier for the current stage
* **Return type:**
  cute.Pointer

#### commit(, loc=None, ip=None)

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

#### \_\_group *: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup)*

#### \_\_init_\_(pipeline, state, group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup))

Initialize a new Producer instance.

* **Parameters:**
  * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync)) – The pipeline this producer belongs to
  * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – Initial pipeline state
  * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – The cooperative group for synchronization

#### \_\_pipeline *: [PipelineAsync](#cutlass.pipeline.PipelineAsync)*

#### \_\_state *: [PipelineState](#cutlass.pipeline.PipelineState)*

#### acquire(try_acquire_token: Boolean | None = None, , loc=None, ip=None) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

Wait for the current buffer to be empty before producing data.
This is a blocking operation.

* **Parameters:**
  **try_acquire_token** (*Optional* *[**Boolean* *]*) – Optional token to try to acquire the buffer
* **Returns:**
  A handle to the producer for committing the data
* **Return type:**
  [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

#### acquire_and_advance(try_acquire_token: Boolean | None = None, , loc=None, ip=None) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

Acquire the current buffer and advance to the next pipeline stage.

This method combines the acquire() and advance() operations into a single call.
It first waits for the current buffer to be empty before producing data,
then advances the pipeline to the next stage.

* **Parameters:**
  **try_acquire_token** (*Optional* *[**Boolean* *]*) – Token indicating whether to try non-blocking acquire.
  If True, returns immediately without waiting. If False or None, blocks
  until buffer is empty.
* **Returns:**
  A handle to the producer that can be used to commit data to the
  acquired buffer stage
* **Return type:**
  [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

#### advance(, loc=None, ip=None)

Move to the next pipeline stage.

#### clone()

Create a new Producer instance with the same state.

#### commit(handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle) | None = None, , loc=None, ip=None)

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

* **Parameters:**
  **handle** (*Optional* *[*[*ImmutableResourceHandle*](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle) *]*) – Optional handle to commit, defaults to None
* **Raises:**
  **AssertionError** – If provided handle does not belong to this producer

#### reset(, loc=None, ip=None)

Reset the count of how many handles this producer has committed.

#### tail(, loc=None, ip=None)

Ensure all used buffers are properly synchronized before producer exit.

This should be called before the producer finishes to avoid dangling signals.

#### try_acquire(, loc=None, ip=None) → Boolean

Attempt to acquire the current buffer without blocking.

This method tries to acquire the current buffer stage for producing data
without waiting. It can be used to check buffer availability before
committing to a blocking acquire operation.

* **Returns:**
  A boolean token indicating whether the buffer was successfully acquired
* **Return type:**
  Boolean

### *class* cutlass.pipeline.PipelineState(stages: int, count, index, phase)

Bases: `object`

Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

#### \_\_init_\_(stages: int, count, index, phase)

#### advance(, loc=None, ip=None) → None

#### clone() → [PipelineState](#cutlass.pipeline.PipelineState)

#### *property* count *: Int32*

#### *property* index *: Int32*

#### *property* phase *: Int32*

#### reset_count(, loc=None, ip=None)

#### reverse(, loc=None, ip=None)

#### *property* stages *: int*

### *class* cutlass.pipeline.PipelineTmaAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_signalling_thread: Boolean)

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_signalling_thread: Boolean) → None

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

TMA consumer release conditionally signals the empty buffer to the producer.

#### *static* create(, num_stages: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), tx_count: int, barrier_storage: Pointer = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, tidx: Int32 | None = None, mcast_mode_mn: tuple[int, int] = (1, 1), defer_sync: bool = False)

Create a new `PipelineTmaAsync` instance.

* **Parameters:**
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
  * **tx_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
  * **barrier_storage** (*cute.Pointer* *,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers, defaults to None
  * **cta_layout_vmnk** ([*cute.Layout*](cute.md#cutlass.cute.Layout) *,* *optional*) – Layout of the cluster shape, defaults to None
  * **tidx** (*Int32* *,* *optional*) – Thread index to consumer async threads, defaults to None
  * **mcast_mode_mn** (*tuple* *[**int* *,* *int* *]* *,* *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)
* **Raises:**
  **ValueError** – If barrier_storage is not a cute.Pointer instance
* **Returns:**
  New `PipelineTmaAsync` instance
* **Return type:**
  [PipelineTmaAsync](#cutlass.pipeline.PipelineTmaAsync)

#### *static* init_empty_barrier_arrive_signal(cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout), tidx: Int32, mcast_mode_mn: tuple[int, int] = (1, 1))

Initialize the empty barrier arrive signal.

This function determines which threads should signal empty barrier arrives based on the cluster layout
and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

* **Parameters:**
  * **cta_layout_vmnk** ([*cute.Layout*](cute.md#cutlass.cute.Layout)) – Layout describing the cluster shape and CTA arrangement
  * **tidx** (*Int32*) – Thread index within the warp
  * **mcast_mode_mn** (*tuple* *[**int* *,* *int* *]*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)
* **Raises:**
  **AssertionError** – If both multicast modes are disabled (0,0)
* **Returns:**
  Tuple containing destination CTA rank and boolean indicating if current thread signals
* **Return type:**
  tuple[Int32, Boolean]

#### is_signalling_thread *: Boolean*

#### producer_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

#### producer_commit(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

### *class* cutlass.pipeline.PipelineTmaMultiConsumersAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_leader_cta: bool, sync_object_empty_umma: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty_async: [SyncObject](#cutlass.pipeline.SyncObject), cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup))

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_leader_cta: bool, sync_object_empty_umma: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty_async: [SyncObject](#cutlass.pipeline.SyncObject), cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) → None

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), op_type: [PipelineOp](#cutlass.pipeline.PipelineOp), , loc=None, ip=None)

#### *static* create(, num_stages: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group_umma: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group_async: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), tx_count: int, barrier_storage: Pointer = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, defer_sync: bool = False)

This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
:param barrier_storage: Pointer to the smem address for this pipeline’s mbarriers
:type barrier_storage: cute.Pointer
:param num_stages: Number of buffer stages for this pipeline
:type num_stages: Int32
:param producer_group: CooperativeGroup for the producer agent
:type producer_group: CooperativeGroup
:param consumer_group_umma: CooperativeGroup for the UMMA consumer agent
:type consumer_group_umma: CooperativeGroup
:param consumer_group_async: CooperativeGroup for the AsyncThread consumer agent
:type consumer_group_async: CooperativeGroup
:param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx_count: int
:param cta_layout_vmnk: Layout of the cluster shape
:type cta_layout_vmnk: cute.Layout | None

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)*

#### is_leader_cta *: bool*

#### producer_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

#### producer_commit(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

#### sync_object_empty_async *: [SyncObject](#cutlass.pipeline.SyncObject)*

#### sync_object_empty_umma *: [SyncObject](#cutlass.pipeline.SyncObject)*

### *class* cutlass.pipeline.PipelineTmaStore(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None)

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None) → None

#### consumer_release(, loc=None, ip=None)

#### consumer_wait(, loc=None, ip=None)

#### *static* create(, num_stages: int, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup))

This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

* **Parameters:**
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
* **Returns:**
  A new `PipelineTmaStore` instance
* **Return type:**
  [PipelineTmaStore](#cutlass.pipeline.PipelineTmaStore)

#### producer_acquire(, loc=None, ip=None)

#### producer_commit(, loc=None, ip=None)

#### producer_tail(, loc=None, ip=None)

Make sure the last used buffer empty signal is visible to producer.
Producer tail is usually executed by producer before exit, to avoid dangling
mbarrier arrive signals after kernel exit.

* **Parameters:**
  **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – The pipeline state that points to next useful buffer

### *class* cutlass.pipeline.PipelineTmaUmma(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_leader_cta: bool, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup))

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, is_leader_cta: bool, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) → None

#### \_compute_is_leader_cta(, loc=None, ip=None)

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

#### \_compute_mcast_arrival_mask(mcast_mode_mn: tuple[int, int], , loc=None, ip=None)

Computes a mask for signaling arrivals to multicasting threadblocks.

#### \_make_sync_object(num_stages: int, agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup)], tx_count: int = 0, , loc=None, ip=None) → [SyncObject](#cutlass.pipeline.SyncObject)

Returns a SyncObject corresponding to an agent’s PipelineOp.

#### consumer_release(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

UMMA consumer release buffer empty, cta_group needs to be provided.

#### create(, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), tx_count: int, barrier_storage: Pointer = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, mcast_mode_mn: tuple[int, int] = (1, 1), defer_sync: bool = False, loc=None, ip=None)

Creates and initializes a new PipelineTmaUmma instance.

* **Parameters:**
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the consumer agent
  * **tx_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
  * **barrier_storage** (*cute.Pointer* *,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
  * **cta_layout_vmnk** ([*cute.Layout*](cute.md#cutlass.cute.Layout) *,* *optional*) – Layout of the cluster shape
  * **mcast_mode_mn** (*tuple* *[**int* *,* *int* *]* *,* *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
* **Raises:**
  **ValueError** – If barrier_storage is not a cute.Pointer instance
* **Returns:**
  A new PipelineTmaUmma instance configured with the provided parameters
* **Return type:**
  [PipelineTmaUmma](#cutlass.pipeline.PipelineTmaUmma)

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)*

#### is_leader_cta *: bool*

#### producer_acquire(state: [PipelineState](#cutlass.pipeline.PipelineState), try_acquire_token: Boolean | None = None, , loc=None, ip=None)

TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

#### producer_commit(state: [PipelineState](#cutlass.pipeline.PipelineState))

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

### *class* cutlass.pipeline.PipelineUmmaAsync(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup))

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

#### \_\_init_\_(sync_object_full: [SyncObject](#cutlass.pipeline.SyncObject), sync_object_empty: [SyncObject](#cutlass.pipeline.SyncObject), num_stages: int, producer_mask: Int32 | None, consumer_mask: Int32 | None, cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) → None

#### \_compute_peer_cta_rank(, ip=None)

Computes a mask to signal release of tmem buffers for 2CTA kernels.

#### \_compute_tmem_sync_mask(, loc=None, ip=None)

Computes a mask to signal completion of tmem buffers for 2CTA kernels.

#### create(, producer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), consumer_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup), barrier_storage: Pointer = None, cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout) | None = None, defer_sync: bool = False, loc=None, ip=None)

Creates an instance of PipelineUmmaAsync with computed attributes.

* **Parameters:**
  * **num_stages** (*int*) – Number of buffer stages for this pipeline
  * **producer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
  * **consumer_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
  * **barrier_storage** (*cute.Pointer* *,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
  * **cta_layout_vmnk** ([*cute.Layout*](cute.md#cutlass.cute.Layout) *,* *optional*) – Layout of the cluster shape
* **Raises:**
  **ValueError** – If barrier_storage is not a cute.Pointer instance
* **Returns:**
  New instance of `PipelineUmmaAsync`
* **Return type:**
  [PipelineUmmaAsync](#cutlass.pipeline.PipelineUmmaAsync)

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)*

#### producer_commit(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

UMMA producer commit buffer full, cta_group needs to be provided.

#### producer_tail(state: [PipelineState](#cutlass.pipeline.PipelineState), , loc=None, ip=None)

Make sure the last used buffer empty signal is visible to producer.
Producer tail is usually executed by producer before exit, to avoid dangling
mbarrier arrive signals after kernel exit.

* **Parameters:**
  **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – The pipeline state that points to next useful buffer

### *class* cutlass.pipeline.PipelineUserType(\*values)

Bases: `Enum`

#### Consumer *= 2*

#### Producer *= 1*

#### ProducerConsumer *= 3*

### *class* cutlass.pipeline.SyncObject

Bases: `ABC`

Abstract base class for hardware synchronization primitives.

This class defines the interface for different types of hardware synchronization
mechanisms including shared memory barriers, named barriers, and fences.

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstractmethod* arrive() → None

#### *abstractmethod* arrive_and_drop() → None

#### *abstractmethod* arrive_and_wait() → None

#### *abstractmethod* get_barrier() → Pointer | int | None

#### *abstractmethod* max() → int | None

#### *abstractmethod* wait() → None

### *class* cutlass.pipeline.TmaStoreFence(num_stages: int = 0)

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

TmaStoreFence is used for a multi-stage epilogue buffer.

#### \_\_init_\_(num_stages: int = 0) → None

#### \_abc_impl *= <_abc._abc_data object>*

#### arrive(, loc=None, ip=None) → None

#### arrive_and_drop(, loc=None, ip=None) → None

#### arrive_and_wait(, loc=None, ip=None) → None

#### get_barrier(, loc=None, ip=None) → None

#### max() → None

#### tail(, loc=None, ip=None) → None

#### wait(, loc=None, ip=None) → None

### cutlass.pipeline.agent_sync(group: [Agent](#cutlass.pipeline.Agent), is_relaxed: bool = False, , loc=None, ip=None)

Syncs all threads within an agent.

### cutlass.pipeline.arrive(barrier_id: int, num_threads: int, , loc=None, ip=None)

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

### cutlass.pipeline.arrive_and_wait(barrier_id: int, num_threads: int, , loc=None, ip=None)

### cutlass.pipeline.arrive_unaligned(barrier_id: int, num_threads: int, , loc=None, ip=None)

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

### cutlass.pipeline.make_pipeline_state(type: [PipelineUserType](#cutlass.pipeline.PipelineUserType), stages: int, , loc=None, ip=None)

Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

### cutlass.pipeline.pipeline_init_arrive(cluster_shape_mn: [Layout](cute.md#cutlass.cute.Layout) | None = None, is_relaxed: bool = False, , loc=None, ip=None)

Fences the mbarrier_init and sends an arrive if using clusters.

### cutlass.pipeline.pipeline_init_wait(cluster_shape_mn: [Layout](cute.md#cutlass.cute.Layout) | None = None, , loc=None, ip=None)

Syncs the threadblock or cluster

### cutlass.pipeline.sync(barrier_id: int = 0, , loc=None, ip=None)

### cutlass.pipeline.wait(, loc=None, ip=None)

NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

### cutlass.pipeline.wait_unaligned(barrier_id: int, num_threads: int, , loc=None, ip=None)
