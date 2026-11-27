"""Apply `--attn-params` to the process-wide attention backend configuration."""

from .params import resolve_attn_params


def _apply_attn_params(args) -> None:
  """Resolve `--attn-params` and push the options into the process-wide backend configuration.

  The `type` key selects the backend and is exposed as `args.attn`; the ffpa and sol_attn options
  are module-level state, so they apply to both the direct and the context-parallel attention
  paths.

  :param args: Parsed CLI namespace; `attn` and `attn_fields` are set here.
  """

  args.attn, fields = resolve_attn_params(args.attn_params)
  args.attn_fields = fields
  if args.attn == "sol_attn":
    # Runs without options too: it validates the Sol-Attn runtime up front.
    from ..attention import set_sol_attn_config

    set_sol_attn_config(**fields)
  elif fields:
    from ..attention.backends import ffpa

    for key, value in fields.items():
      getattr(ffpa, f"set_ffpa_{key}")(value)
