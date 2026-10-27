"""Image generation pipelines shipped by cache-dit itself.

Some checkpoints ship their own `pipeline_<model>.py` and point `model_index.json` at it, which a
plain `DiffusionPipeline.from_pretrained` cannot import. Those pipelines are vendored here so
cache-dit can run them without `trust_remote_code` or a remote `custom_pipeline` module.
"""

__all__ = []
