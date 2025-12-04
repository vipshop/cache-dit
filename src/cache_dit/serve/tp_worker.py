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

import torch
import torch.distributed as dist

from cache_dit.serve.model_manager import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class TPCoordinator:
    """
    Coordinator for Tensor Parallelism inference.

    Runs on rank 0 and broadcasts requests to all TP workers.
    """

    def __init__(self, model_manager, rank: int, world_size: int):
        self.model_manager = model_manager
        self.rank = rank
        self.world_size = world_size
        logger.info(f"TPCoordinator initialized: rank={rank}, world_size={world_size}")

    @property
    def pipe(self):
        """Expose the underlying model_manager's pipe for compatibility."""
        return self.model_manager.pipe

    def get_model_info(self):
        """Get model information from the underlying model manager."""
        return self.model_manager.get_model_info()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate images using TP.

        This method broadcasts the request to all ranks and collects the result.
        """
        # Serialize request
        request_data = pickle.dumps(request)
        request_size = len(request_data)

        # Broadcast request size
        size_tensor = torch.tensor([request_size], dtype=torch.long, device="cuda")
        dist.broadcast(size_tensor, src=0)

        # Broadcast request data
        request_tensor = torch.frombuffer(request_data, dtype=torch.uint8).cuda()
        # Pad to make it divisible by world_size for efficient broadcast
        padded_size = ((request_size + self.world_size - 1) // self.world_size) * self.world_size
        if padded_size > request_size:
            request_tensor = torch.cat(
                [
                    request_tensor,
                    torch.zeros(padded_size - request_size, dtype=torch.uint8, device="cuda"),
                ]
            )
        dist.broadcast(request_tensor, src=0)

        # IMPORTANT: Rank 0 must also deserialize the broadcasted request
        # to ensure all ranks use exactly the same request object
        broadcasted_request_data = request_tensor[:request_size].cpu().numpy().tobytes()
        broadcasted_request = pickle.loads(broadcasted_request_data)

        # All ranks execute inference with the broadcasted request
        response = self.model_manager.generate(broadcasted_request)

        # Rank 0 returns the result
        return response


def run_tp_worker(model_manager, rank: int):
    """
    Worker loop for TP ranks > 0.

    Receives requests from rank 0 and executes inference.
    """
    logger.info(f"TP worker {rank} started, waiting for requests...")

    while True:
        try:
            # Receive request size
            size_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.broadcast(size_tensor, src=0)
            request_size = size_tensor.item()

            # Receive request data
            padded_size = (
                (request_size + dist.get_world_size() - 1) // dist.get_world_size()
            ) * dist.get_world_size()
            request_tensor = torch.zeros(padded_size, dtype=torch.uint8, device="cuda")
            dist.broadcast(request_tensor, src=0)

            # Deserialize request
            request_data = request_tensor[:request_size].cpu().numpy().tobytes()
            request = pickle.loads(request_data)

            # Execute inference
            logger.debug(f"Rank {rank} executing inference...")
            _ = model_manager.generate(request)
            logger.debug(f"Rank {rank} inference completed")

        except KeyboardInterrupt:
            logger.info(f"TP worker {rank} shutting down...")
            break
        except Exception as e:
            logger.error(f"TP worker {rank} error: {e}", exc_info=True)
            time.sleep(0.1)  # Avoid busy loop on repeated errors
