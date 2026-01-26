from ..logger import init_logger
from .utils import get_base_args, maybe_postprocess_args
from .registers import ExampleRegister  # noqa: F403, F401

logger = init_logger(__name__)


def get_example_args():
    parser = get_base_args(parse=False)
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        default="generate",
        choices=["generate", "list"] + ExampleRegister.list_examples(),
        help=(
            "The task to perform or example name to run. "
            "Use 'list' to list all available examples, "
            "or specify an example name directly (defaults to 'generate' task)."
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

    if args.task in ExampleRegister.list_examples():
        args.example = args.task
        args.task = "generate"

    return maybe_postprocess_args(args)


def entrypoint():
    args = get_example_args()
    if args.task == "list":
        logger.info("Available examples:")
        max_name_len = max(len(name) for name in ExampleRegister.list_examples())
        for name in ExampleRegister.list_examples():
            default = ExampleRegister.get_default(name)
            # format by max_name_len
            info = f"- ✅ {name:<{max_name_len}} - Defalut: {default}"
            logger.info(info)
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


__all__ = ["entrypoint"]
