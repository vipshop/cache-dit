from cache_dit.logger import init_logger

# Make all example are registered
from utils import get_args, maybe_postprocess_args
from registers import CacheDiTExampleRegister  # noqa: F403, F401
from helpers import activate_all_examples

activate_all_examples()


def get_example_args():
    parser = get_args(parse=False)
    parser.add_argument(
        "example",
        type=str,
        choices=CacheDiTExampleRegister.list_examples(),
        help="Names of the examples to run. If not specified, run all examples.",
    )
    args = parser.parse_args()
    return maybe_postprocess_args(args)


if __name__ == "__main__":
    logger = init_logger(__name__)
    args = get_example_args()

    example = CacheDiTExampleRegister.get_example(args, args.example)
    example.run()

    # Usage:
    # torchrun --nproc_per_node=4 generate.py zimage --parallel ulysses --ulysses-anything
