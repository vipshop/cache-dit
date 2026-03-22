import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from ..logger import init_logger

logger = init_logger(__name__)


ScheduleEntry = Tuple[int, int, List[int]]
MeshKey = Tuple[int, int, Tuple[int, ...]]


@dataclasses.dataclass
class DynamicSPConfig:
    enabled: bool = False
    # Normalized schedule entries:
    # [(ulysses_degree, ring_degree, active_ranks), ...]
    schedule: List[ScheduleEntry] = dataclasses.field(default_factory=list)

    @staticmethod
    def parse_schedule(
        raw_schedule: list,
        default_ulysses: int = 1,
        default_ring: int = 1,
        world_size: Optional[int] = None,
    ) -> List[ScheduleEntry]:
        if raw_schedule is None:
            return []
        if not isinstance(raw_schedule, list) or len(raw_schedule) == 0:
            raise ValueError("dynamic_sp.schedule must be a non-empty list.")

        world = world_size if world_size is not None else dist.get_world_size()
        if world <= 0:
            raise ValueError(f"Invalid world_size for dynamic_sp schedule parsing: {world}")

        def _validate_ranks(ranks: List[int], degree: int, entry: Any):
            if len(ranks) != degree:
                raise ValueError(
                    f"Invalid dynamic_sp schedule entry {entry}: len(ranks)={len(ranks)} "
                    f"must equal degree={degree}."
                )
            if len(set(ranks)) != len(ranks):
                raise ValueError(f"Invalid dynamic_sp schedule entry {entry}: ranks contains duplicates.")
            for rank in ranks:
                if rank < 0 or rank >= world:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: rank {rank} out of range [0, {world})."
                    )

        normalized: List[ScheduleEntry] = []
        for entry in raw_schedule:
            if isinstance(entry, int):
                degree = int(entry)
                if degree < 1:
                    raise ValueError(f"Invalid dynamic_sp degree {degree}. Must be >= 1.")
                if degree > world:
                    raise ValueError(
                        f"Invalid dynamic_sp degree {degree}. Must be <= world_size={world}."
                    )
                if degree == default_ulysses * default_ring and default_ring > 1:
                    ulysses, ring = default_ulysses, default_ring
                else:
                    # Keep the split simple and deterministic for int entries.
                    ulysses, ring = degree, 1
                ranks = list(range(degree))
                _validate_ranks(ranks, degree, entry)
                normalized.append((ulysses, ring, ranks))
                continue

            if not isinstance(entry, dict):
                raise ValueError(
                    f"Invalid dynamic_sp schedule entry: {entry}. "
                    "Expected int, or dict with {degree, ranks} / {ulysses, ring, ranks}."
                )

            if "degree" in entry:
                degree = int(entry["degree"])
                if degree < 1:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: degree must be >= 1."
                    )
                if degree > world:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: degree must be <= world_size={world}."
                    )
                ulysses = int(entry.get("ulysses", degree))
                ring = int(entry.get("ring", 1))
                if ulysses < 1 or ring < 1:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: ulysses/ring must be >= 1."
                    )
                if ulysses * ring != degree:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: "
                        f"ulysses * ring ({ulysses}*{ring}) must equal degree={degree}."
                    )
            elif "ulysses" in entry and "ring" in entry:
                ulysses = int(entry["ulysses"])
                ring = int(entry["ring"])
                if ulysses < 1 or ring < 1:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: ulysses/ring must be >= 1."
                    )
                degree = ulysses * ring
                if degree > world:
                    raise ValueError(
                        f"Invalid dynamic_sp schedule entry {entry}: "
                        f"ulysses*ring={degree} must be <= world_size={world}."
                    )
            else:
                raise ValueError(
                    f"Invalid dynamic_sp schedule entry {entry}: expected "
                    "`degree` or (`ulysses` and `ring`)."
                )

            ranks = entry.get("ranks", list(range(degree)))
            if not isinstance(ranks, list):
                raise ValueError(
                    f"Invalid dynamic_sp schedule entry {entry}: ranks must be a list of ints."
                )
            ranks = [int(rank) for rank in ranks]
            _validate_ranks(ranks, degree, entry)
            normalized.append((ulysses, ring, ranks))

        return normalized


class DynamicSPManager:
    def __init__(
        self,
        config: DynamicSPConfig,
        cp_config,
        rank: int,
        world_size: int,
        device_type: str,
    ):
        self._cp_config = cp_config
        self._rank = rank
        self._world_size = world_size
        self._device_type = device_type
        self._schedule = config.schedule
        self._step = 0
        self._prev_sp_key: Optional[MeshKey] = None
        self._meshes: Dict[MeshKey, Dict[str, Optional[DeviceMesh]]] = {}
        self._pre_create_meshes(device_type=device_type)

    @property
    def step(self) -> int:
        return self._step

    def advance_step(self) -> None:
        self._step += 1

    def reset(self) -> None:
        self._step = 0
        self._prev_sp_key = None

    def _mesh_key(self, entry: ScheduleEntry) -> MeshKey:
        ulysses, ring, ranks = entry
        return (ulysses, ring, tuple(ranks))

    def _pre_create_meshes(self, device_type: str) -> None:
        for entry in self._schedule:
            key = self._mesh_key(entry)
            if key in self._meshes:
                continue

            ulysses, ring, active_ranks = entry
            total_sp = ulysses * ring
            if len(active_ranks) != total_sp:
                raise ValueError(
                    f"Invalid dynamic_sp schedule {entry}: active_ranks size must equal ulysses*ring."
                )

            # Dynamic SP entries may use a degree that does not divide world_size
            # (e.g. degree=3 on world_size=4). Building extra DP planes by padding
            # inactive ranks can make mesh.numel() > world_size and crash DeviceMesh.
            # Use a single DP plane that only contains active ranks for this step.
            #
            mesh_tensor = torch.tensor(active_ranks, dtype=torch.int).reshape(1, ring, ulysses)
            full_mesh = DeviceMesh(
                device_type=device_type,
                mesh=mesh_tensor,
                mesh_dim_names=("dp", "ring", "ulysses"),
            )
            # Important: all ranks must still participate in full mesh creation
            # to keep process-group creation order consistent across ranks.
            # However, on ranks not in active_ranks some PyTorch versions can
            # fail when slicing submeshes. So we only slice on active ranks.
            if self._rank not in active_ranks:
                self._meshes[key] = {
                    "full": full_mesh,
                    "cp": None,
                    "flat_cp": None,
                    "ring_mesh": None,
                    "ulysses_mesh": None,
                }
                continue

            cp_mesh = full_mesh["ring", "ulysses"]
            # NOTE: Avoid creating a "submesh from a submesh" (some PyTorch versions
            # forbid slicing DeviceMesh derived from another slice).
            ring_mesh = full_mesh["ring"]
            ulysses_mesh = full_mesh["ulysses"]
            self._meshes[key] = {
                "full": full_mesh,
                "cp": cp_mesh,
                "flat_cp": cp_mesh._flatten(),
                "ring_mesh": ring_mesh,
                "ulysses_mesh": ulysses_mesh,
            }

    def get_schedule_entry(self, step: int) -> ScheduleEntry:
        if not self._schedule:
            raise ValueError("dynamic_sp.schedule must not be empty when dynamic SP is enabled.")
        return self._schedule[step % len(self._schedule)]

    def is_active(self, step: int) -> bool:
        _, _, active_ranks = self.get_schedule_entry(step)
        return self._rank in active_ranks

    def get_broadcast_src(self, step: int) -> int:
        _, _, active_ranks = self.get_schedule_entry(step)
        return int(active_ranks[0])

    def sp_degree_changed(self, step: int) -> bool:
        key = self._mesh_key(self.get_schedule_entry(step))
        return key != self._prev_sp_key

    def apply_config(self, step: int) -> None:
        entry = self.get_schedule_entry(step)
        key = self._mesh_key(entry)
        if key not in self._meshes:
            raise ValueError(f"Mesh for dynamic_sp entry {entry} is not initialized.")
        if key == self._prev_sp_key:
            return

        ulysses, ring, _ = entry
        bundle = self._meshes[key]
        cp_mesh = bundle["cp"]
        flat_cp = bundle["flat_cp"]
        ring_mesh = bundle["ring_mesh"]
        ulysses_mesh = bundle["ulysses_mesh"]
        if cp_mesh is None or flat_cp is None or ring_mesh is None or ulysses_mesh is None:
            raise RuntimeError(
                f"Rank {self._rank} is not active for dynamic_sp entry {entry}, "
                "but apply_config() was called."
            )

        self._cp_config.ulysses_degree = ulysses
        self._cp_config.ring_degree = ring
        self._cp_config._rank = self._rank
        self._cp_config._world_size = self._world_size
        self._cp_config._device = torch.device(self._device_type, self._rank)
        self._cp_config._mesh = cp_mesh
        self._cp_config._flattened_mesh = flat_cp
        self._cp_config._ring_mesh = ring_mesh
        self._cp_config._ulysses_mesh = ulysses_mesh
        self._cp_config._ring_local_rank = ring_mesh.get_local_rank()
        self._cp_config._ulysses_local_rank = ulysses_mesh.get_local_rank()

        self._prev_sp_key = key

    def _flatten_output_tensors(self, output) -> List[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return [output]
        if isinstance(output, tuple):
            tensors: List[torch.Tensor] = []
            for item in output:
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        "dynamic_sp only supports Tensor or Tuple[Tensor, ...] outputs, "
                        f"but got tuple item type: {type(item)}."
                    )
                tensors.append(item)
            return tensors
        raise TypeError(
            f"Unsupported dynamic_sp output type for broadcast: {type(output)}. "
            "Only Tensor or Tuple[Tensor, ...] are supported."
        )

    def _rebuild_output(self, tensors: List[torch.Tensor]):
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)

    def _dtype_to_code(self, dtype: torch.dtype) -> int:
        mapping = {
            torch.float16: 0,
            torch.bfloat16: 1,
            torch.float32: 2,
            torch.float64: 3,
            torch.int64: 4,
            torch.int32: 5,
            torch.int16: 6,
            torch.int8: 7,
            torch.uint8: 8,
            torch.bool: 9,
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported dtype for dynamic_sp broadcast: {dtype}.")
        return mapping[dtype]

    def _code_to_dtype(self, code: int) -> torch.dtype:
        mapping = {
            0: torch.float16,
            1: torch.bfloat16,
            2: torch.float32,
            3: torch.float64,
            4: torch.int64,
            5: torch.int32,
            6: torch.int16,
            7: torch.int8,
            8: torch.uint8,
            9: torch.bool,
        }
        if code not in mapping:
            raise TypeError(f"Unsupported dtype code for dynamic_sp broadcast: {code}.")
        return mapping[code]

    def _build_spec_tensor(self, tensors: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        # spec format (int64):
        # [num_tensors, dtype_code, ndim, *shape, dtype_code, ndim, *shape, ...]
        spec: List[int] = [len(tensors)]
        for tensor in tensors:
            spec.append(self._dtype_to_code(tensor.dtype))
            spec.append(tensor.dim())
            spec.extend(list(tensor.shape))
        return torch.tensor(spec, dtype=torch.int64, device=device)

    def _parse_spec_tensor(self, spec_tensor: torch.Tensor) -> List[Tuple[torch.dtype, Tuple[int, ...]]]:
        spec_list = spec_tensor.tolist()
        num_tensors = int(spec_list[0])
        idx = 1
        parsed: List[Tuple[torch.dtype, Tuple[int, ...]]] = []
        for _ in range(num_tensors):
            dtype_code = int(spec_list[idx])
            idx += 1
            ndim = int(spec_list[idx])
            idx += 1
            shape = tuple(int(x) for x in spec_list[idx : idx + ndim])
            idx += ndim
            parsed.append((self._code_to_dtype(dtype_code), shape))
        return parsed

    def sync_output(self, output, hidden_states, step: int):
        src = self.get_broadcast_src(step)
        device = hidden_states.device if isinstance(hidden_states, torch.Tensor) else None
        if device is None:
            device = getattr(self._cp_config, "_device", None)
        if device is None:
            device = torch.device(self._device_type, self._rank)

        # 1) Broadcast tensor spec on GPU to guarantee consistent shapes across ranks.
        if self._rank == src:
            tensors = self._flatten_output_tensors(output)
            tensors = [tensor.contiguous() if not tensor.is_contiguous() else tensor for tensor in tensors]
            spec_tensor = self._build_spec_tensor(tensors=tensors, device=device)
            spec_len = torch.tensor([spec_tensor.numel()], dtype=torch.int64, device=device)
        else:
            tensors = []
            spec_len = torch.empty(1, dtype=torch.int64, device=device)

        dist.broadcast(spec_len, src=src)
        if self._rank != src:
            spec_tensor = torch.empty(int(spec_len.item()), dtype=torch.int64, device=device)
        dist.broadcast(spec_tensor, src=src)

        parsed_spec = self._parse_spec_tensor(spec_tensor)
        if self._rank != src:
            tensors = [torch.empty(shape, dtype=dtype, device=device) for dtype, shape in parsed_spec]

        # 2) Broadcast actual tensor payloads.
        for tensor in tensors:
            dist.broadcast(tensor, src=src)

        return self._rebuild_output(tensors)
