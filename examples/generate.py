from cache_dit.logger import init_logger

# Make all example are registered
from utils import get_base_args, maybe_postprocess_args
from registers import CacheDiTExampleRegister  # noqa: F403, F401
from helpers import activate_all_examples

activate_all_examples()

logger = init_logger(__name__)


def get_example_args():
    parser = get_base_args(parse=False)
    parser.add_argument(
        "example",
        type=str,
        default=None,
        choices=[None] + CacheDiTExampleRegister.list_example_names(),
        help="Names of the examples to run. If not specified, run all examples.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List all available examples and exit.",
    )
    args = parser.parse_args()
    return maybe_postprocess_args(args)


if __name__ == "__main__":
    logger = init_logger(__name__)
    args = get_example_args()
    if args.list_examples:
        logger.info("Available examples:")
        for name in CacheDiTExampleRegister.list_examples():
            logger.info(f"- {name}")
        exit(0)
    else:
        if args.example is None:
            logger.error(
                "Please specify an example name to run. Use --list-examples to see all available examples."
            )
            exit(1)

        example = CacheDiTExampleRegister.get_example(args, args.example)
        example.run()

    # Usage:
    # python3 generate.py --list-examples
    # torchrun --nproc_per_node=4 generate.py zimage --parallel ulysses --ulysses-anything
