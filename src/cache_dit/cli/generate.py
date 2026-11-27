import os
from ..logger import (
  init_logger,
  suppress_torch_compile_loggers,
  globally_suppress_loggers,
)

# Prefer to supppress loggers globally for better readability,
# if the environment variable is set to enable loggers suppress.
# Otherwise, users may see messy logs from some noisy loggers,
# which can be confusing.
loggers_suppress_env = os.environ.get(
  "CACHE_DIT_ENABLE_LOGGERS_SUPPRESS",
  None,
)
if loggers_suppress_env is None or bool(int(loggers_suppress_env)):
  globally_suppress_loggers()
  suppress_torch_compile_loggers()

from .utils import get_base_args, maybe_postprocess_args
from .registers import ExampleRegister, ExampleType  # noqa: F403, F401

logger = init_logger(__name__)

_TASK_NAMES = [task.name for task in ExampleType]

# Examples whose registered default is not the main checkpoint but LoRA / ControlNet / distilled
# transformer weights: a model path for them goes to that slot instead of `model_path`, where the
# factory loads it from. A test runs every example to keep this table exact.
_PATH_SLOT_EXAMPLES = {
  "causvid_wan": "transformer_path",
  "flux_nunchaku": "transformer_path",
  "qwen_image_lightning": "lora_path",
  "qwen_image_edit_lightning": "lora_path",
  "qwen_image_edit_2511_lightning": "lora_path",
  "qwen_image_controlnet": "controlnet_path",
  "zimage_turbo_nunchaku": "transformer_path",
  "zimage_turbo_controlnet_2.0": "controlnet_path",
  "zimage_turbo_controlnet_2.1": "controlnet_path",
}


def _is_model_path(selector: str) -> bool:
  """Whether a positional is a model path rather than an example name.

  A registered name is an example name (several examples branch on `args.example`); everything
  else, e.g. `/data/FLUX.1-dev`, `./flux` or `org/name`, is matched against the default models.

  :param selector: First positional argument.
  :returns: True when the selector should be matched against the registered default models.
  """

  return selector not in ExampleRegister.list_examples()


def get_example_args():
  parser = get_base_args(parse=False)
  parser.add_argument(
    "task",
    type=str,
    nargs="?",
    default="generate",
    help=("Run an example by three forms: `list` to list all available examples; an example "
          "name, e.g. `flux`; or a model directory / repo id, e.g. `/data/FLUX.1-dev` or "
          "`black-forest-labs/FLUX.1-dev`, whose example is matched from the model "
          "(defaults to 'generate' task)."),
  )
  parser.add_argument(
    "example",
    type=str,
    nargs="?",
    default=None,
    help="Example to run, for the `generate <example>` form. If not specified, skip running.",
  )
  parser.add_argument(
    "--task",
    dest="task_type",
    type=str.upper,
    choices=_TASK_NAMES,
    default=None,
    help=("Task the example drives, e.g. T2I, T2V, I2V, IE2I. Only needed when a model path "
          "matches several examples; without it the model's default task is used "
          "(T2I for image models, then the video tasks)."),
  )
  parser.add_argument(
    "--example-summary",
    "--esummary",  # short alias for better readability when used together with --cache-summary
    action="store_true",
    default=False,
    help="Enable example summary logging",
  )
  args = parser.parse_args()

  selector, second = args.task, args.example
  if selector == "generate":
    # The `generate <example>` form: the selector is the second positional.
    selector, second = second, None
  if selector == "list":
    args.task = "list"
  elif selector is None:
    args.task = "generate"
    args.example = None
    if args.task_type is not None:
      parser.error("--task needs an example name or a model path.")
  elif _is_model_path(selector):
    if second is not None:
      parser.error(f"'{second}' cannot follow the model path '{selector}'; the example is matched "
                   "from the model path, pass --task to pick a task instead.")
    task = ExampleType[args.task_type] if args.task_type is not None else None
    try:
      name, candidates = ExampleRegister.select_example(selector, task)
    except KeyError as error:
      parser.error(f"{error.args[0]} Run 'python -m cache_dit.generate list' to see the examples "
                   "and their default models.")
    except LookupError as error:
      parser.error(str(error))
    model_path = os.path.expandvars(os.path.expanduser(selector))
    slot = _PATH_SLOT_EXAMPLES.get(name, "model_path")
    if getattr(args, slot) is not None:
      parser.error(f"--{slot.replace('_', '-')} cannot be combined with a model path argument; "
                   f"'{selector}' is the model '{name}' loads through --{slot.replace('_', '-')}, "
                   "pass one of them.")
    args.task = "generate"
    args.example = name
    setattr(args, slot, model_path)
    others = ", ".join(f"{other} ({ExampleRegister.get_task(other).name})" for other in candidates
                       if other != name)
    logger.info(f"Model path '{selector}' matched example '{name}' "
                f"({ExampleRegister.get_task(name).name}).")
    if others:
      logger.info(f"Other examples for this model: {others}; pass --task to pick one of them.")
  else:
    if second is not None:
      parser.error(f"unrecognized arguments: {second}")
    if selector not in ExampleRegister.list_examples():
      parser.error(f"Unknown example '{selector}'. Run 'python -m cache_dit.generate list' to see "
                   "the available examples.")
    args.task = "generate"
    args.example = selector
    if args.task_type is not None:
      actual = ExampleRegister.get_task(selector)
      if actual.name != args.task_type:
        parser.error(f"--task {args.task_type} does not match example '{selector}', "
                     f"which drives {actual.name}.")

  return maybe_postprocess_args(args)


def entrypoint():
  args = get_example_args()
  if args.task == "list":
    logger.info("Available examples:")
    max_name_len = max(len(name) for name in ExampleRegister.list_examples())
    for name in ExampleRegister.list_examples():
      default = ExampleRegister.get_default(name)
      task = ExampleRegister.get_task(name).name
      # format by max_name_len
      info = f"- ✅ {name:<{max_name_len}} [{task}] - Default: {default}"
      logger.info(info)
    logger.info("Pass a model directory or repo id instead of an example name to match the "
                "example from the model, e.g. `python3 -m cache_dit.generate /path/to/FLUX.1-dev`; "
                "add `--task T2I` (T2V, I2V, ...) when the model serves several tasks.")
    exit(0)
  else:
    if args.example is None:
      logger.error("Please specify an example name or a model path to run. Use "
                   "'python3 -m cache_dit.generate list' to see all available examples.")
      exit(1)

    if args.cache_summary or args.example_summary:
      # Only logging all args when the 'summary' flag is set for better readability.
      logger.info("Running example with the following arguments:")
      for arg, value in vars(args).items():
        logger.info(f"- {arg}: {value}")

    example = ExampleRegister.get_example(args, args.example)
    example.run()


__all__ = ["entrypoint"]
