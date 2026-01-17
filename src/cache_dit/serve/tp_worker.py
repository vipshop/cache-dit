"""
Tensor Parallelism worker for distributed inference.

This module implements a simple broadcast-based mechanism for TP serving:
- Rank 0 receives HTTP requests and broadcasts them to all ranks
- All ranks execute inference synchronously
- Rank 0 collects and returns the result

Inspired by SGLang's distributed architecture.
"""

import logging
import pickle
import time
import threading

import torch
import torch.distributed as dist

from ..platforms import current_platform
from .model_manager import GenerateRequest, GenerateResponse, ModelManager

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 300
HEARTBEAT_SIZE = -1


class TPCoordinator:
    """
    Coordinator for Tensor Parallelism inference.

    Runs on rank 0 and broadcasts requests to all TP workers.
    """

    def __init__(self, model_manager: ModelManager, rank: int, world_size: int):
        self.model_manager = model_manager
        self.rank = rank
        self.world_size = world_size
        self._last_broadcast_time = time.time()
        self._heartbeat_lock = threading.Lock()
        self._stop_heartbeat = False
        self._heartbeat_thread = None
        logger.info(f"TPCoordinator initialized: rank={rank}, world_size={world_size}")
        self._start_heartbeat()

    @property
    def pipe(self):
        """Expose the underlying model_manager's pipe for compatibility."""
        return self.model_manager.pipe

    def get_model_info(self):
        """Get model information from the underlying model manager."""
        return self.model_manager.get_model_info()

    def _start_heartbeat(self):
        def heartbeat_loop():
            while not self._stop_heartbeat:
                time.sleep(HEARTBEAT_INTERVAL)
                with self._heartbeat_lock:
                    if time.time() - self._last_broadcast_time > HEARTBEAT_INTERVAL:
                        try:
                            size_tensor = torch.tensor(
                                [HEARTBEAT_SIZE],
                                dtype=torch.long,
                                device=current_platform.device_type,
                            )
                            dist.broadcast(size_tensor, src=0)
                            self._last_broadcast_time = time.time()
                            logger.debug("Heartbeat sent to workers")
                        except Exception as e:
                            logger.error(f"Heartbeat failed: {e}")

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"Heartbeat thread started (interval={HEARTBEAT_INTERVAL}s)")

    def stop(self):
        self._stop_heartbeat = True
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate images using TP.

        This method broadcasts the request to all ranks and collects the result.
        """
        with self._heartbeat_lock:
            current_platform.synchronize()

            request_data = pickle.dumps(request)
            request_size = len(request_data)

            size_tensor = torch.tensor(
                [request_size], dtype=torch.long, device=current_platform.device_type
            )
            dist.broadcast(size_tensor, src=0)

            padded_size = (
                (request_size + self.world_size - 1) // self.world_size
            ) * self.world_size
            request_tensor = torch.zeros(
                padded_size, dtype=torch.uint8, device=current_platform.device_type
            )
            request_tensor[:request_size].copy_(torch.frombuffer(request_data, dtype=torch.uint8))
            dist.broadcast(request_tensor, src=0)

            self._last_broadcast_time = time.time()

        # IMPORTANT: Rank 0 must also deserialize the broadcasted request
        # to ensure all ranks use exactly the same request object
        broadcasted_request_data = request_tensor[:request_size].cpu().numpy().tobytes()
        broadcasted_request = pickle.loads(broadcasted_request_data)

        # All ranks execute inference with the broadcasted request
        response = self.model_manager.generate(broadcasted_request)

        # Rank 0 returns the result
        return response


def run_tp_worker(model_manager: ModelManager, rank: int):
    """
    Worker loop for TP ranks > 0.

    Receives requests from rank 0 and executes inference.
    """
    logger.info(f"TP worker {rank} started, waiting for requests...")

    while True:
        try:
            current_platform.synchronize()

            size_tensor = torch.tensor([0], dtype=torch.long, device=current_platform.device_type)
            dist.broadcast(size_tensor, src=0)
            request_size = size_tensor.item()

            if request_size == HEARTBEAT_SIZE:
                logger.debug(f"Rank {rank} received heartbeat")
                continue

            padded_size = (
                (request_size + dist.get_world_size() - 1) // dist.get_world_size()
            ) * dist.get_world_size()
            request_tensor = torch.zeros(
                padded_size, dtype=torch.uint8, device=current_platform.device_type
            )
            dist.broadcast(request_tensor, src=0)

            request_data = request_tensor[:request_size].cpu().numpy().tobytes()
            request = pickle.loads(request_data)

            logger.debug(f"Rank {rank} executing inference...")
            _ = model_manager.generate(request)
            logger.debug(f"Rank {rank} inference completed")

        except KeyboardInterrupt:
            logger.info(f"TP worker {rank} shutting down...")
            break
        except RuntimeError as e:
            if "NCCL" in str(e) or "timeout" in str(e).lower():
                logger.error(f"TP worker {rank} NCCL error: {e}")
                dist.destroy_process_group()
                break
            else:
                logger.exception(f"TP worker {rank} runtime error: {type(e).__name__}: {e}")
                time.sleep(0.1)
        except Exception as e:
            logger.exception(f"TP worker {rank} error: {type(e).__name__}: {e}")
            time.sleep(0.1)
