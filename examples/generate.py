from cache_dit.logger import init_logger
from utils import get_base_args, maybe_postprocess_args
from registers import ExampleRegister  # noqa: F403, F401
from helpers import activate_all_examples

# Make sure all example are registered
activate_all_examples()

logger = init_logger(__name__)


def get_example_args():
    parser = get_base_args(parse=False)
    parser.add_argument(
        "task",
        type=str,
        default="generate",
        choices=["generate", "list"],
        help=(
            "The task to perform. If not specified, run the specified example. "
            "Or, Use 'list' to list all available examples."
        ),
    )
    parser.add_argument(
        "example",
        type=str,
        nargs="?",
        default=None,
        choices=[None] + ExampleRegister.list_examples(),
        help="Names of the examples to run. If not specified, skip running example.",
    )
    args = parser.parse_args()
    return maybe_postprocess_args(args)


if __name__ == "__main__":
    logger = init_logger(__name__)
    args = get_example_args()
    if args.task == "list":
        logger.info("Available examples:")
        for name in ExampleRegister.list_examples():
            logger.info(f"- {name}")
        exit(0)
    else:
        if args.example is None:
            logger.error(
                "Please specify an example name to run. Use --list-examples to "
                "see all available examples."
            )
            exit(1)

        # logging all args with better formatting
        logger.info("Running example with the following arguments:")
        for arg, value in vars(args).items():
            logger.info(f"- {arg}: {value}")

        example = ExampleRegister.get_example(args, args.example)
        example.run()

    # Usage:
    # python3 generate.py list
    # python3 generate.py generate zimage
    # python3 generate.py generate qwen_image_edit_lightning --cpu-offload
    # torchrun --nproc_per_node=4 generate.py generate zimage --parallel ulysses --ulysses-anything
    # torchrun --nproc_per_node=4 generate.py generate zimage --parallel tp --parallel-text
