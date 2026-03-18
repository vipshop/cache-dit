import importlib
from ..logger import init_logger

logger = init_logger(__name__)


def import_error_metric_func(*args, **kwargs):
    raise ImportError(
        "This metric function requires additional dependencies that are not installed. "
        "Please check the documentation for installation instructions."
    )


def _safe_import(module_name: str, func_name: str):
    """Helper function to safely import a function from a module."""
    try:
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_func = getattr(module, func_name)
        return target_func
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {func_name} from {module_name}: {e}")
        return import_error_metric_func
