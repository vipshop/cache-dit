import pytest
from torch._inductor import config as inductor_config

from cache_dit.compile.utils import set_compile_configs
from cache_dit.envs import ENV


@pytest.mark.parametrize("env_enabled,override,expected", [
  (True, None, True),
  (False, None, False),
  (True, False, False),
  (False, True, True),
])
def test_epilogue_prologue_fusion_controls(monkeypatch, env_enabled, override, expected) -> None:
  monkeypatch.setattr(ENV, "CACHE_DIT_EPILOGUE_PROLOGUE_FUSION", env_enabled)
  monkeypatch.setattr(inductor_config, "epilogue_fusion", None)
  monkeypatch.setattr(inductor_config, "prologue_fusion", None)
  monkeypatch.setattr(inductor_config, "epilogue_fusion_first", None)
  kwargs = {} if override is None else {"epilogue_prologue_fusion": override}
  set_compile_configs(**kwargs)
  assert inductor_config.epilogue_fusion is expected
  assert inductor_config.prologue_fusion is expected
  assert inductor_config.epilogue_fusion_first is expected
