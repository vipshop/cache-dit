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
  # The fusion switches are independent of descent tuning, but
  # `epilogue_fusion_first` still belongs to the aggressive tuning set.
  set_compile_configs(descent_tuning=True, **kwargs)
  assert inductor_config.epilogue_fusion is expected
  assert inductor_config.prologue_fusion is expected
  assert inductor_config.epilogue_fusion_first is expected


def test_fusion_and_cudagraph_switches_ignore_descent_tuning(monkeypatch) -> None:
  monkeypatch.setattr(ENV, "CACHE_DIT_EPILOGUE_PROLOGUE_FUSION", False)
  monkeypatch.setattr(inductor_config, "triton.cudagraphs", False)
  monkeypatch.setattr(inductor_config, "epilogue_fusion", None)
  monkeypatch.setattr(inductor_config, "prologue_fusion", None)
  monkeypatch.setattr(inductor_config, "epilogue_fusion_first", None)
  set_compile_configs(cuda_graphs=True)
  assert inductor_config.triton.cudagraphs is True
  assert inductor_config.epilogue_fusion is False
  assert inductor_config.prologue_fusion is False
  assert inductor_config.epilogue_fusion_first is None


def test_descent_tuning_is_disabled_by_default(monkeypatch) -> None:
  # Sentinel: `fx_graph_cache` is written unconditionally inside the inductor block,
  # so it proves the block ran instead of being swallowed by the surrounding try.
  monkeypatch.setattr(inductor_config, "fx_graph_cache", None)
  monkeypatch.setattr(inductor_config, "coordinate_descent_tuning", None)
  monkeypatch.setattr(inductor_config, "coordinate_descent_check_all_directions", None)
  monkeypatch.setattr(inductor_config, "conv_1x1_as_mm", None)
  monkeypatch.setattr(inductor_config, "dce", None)
  set_compile_configs()
  assert inductor_config.fx_graph_cache is True
  assert inductor_config.coordinate_descent_tuning is None
  assert inductor_config.coordinate_descent_check_all_directions is None
  assert inductor_config.conv_1x1_as_mm is None
  assert inductor_config.dce is None


def test_cache_and_fast_math_toggles_ignore_descent_tuning(monkeypatch) -> None:
  monkeypatch.setattr(inductor_config, "force_disable_caches", False)
  monkeypatch.setattr(inductor_config, "fx_graph_cache", True)
  monkeypatch.setattr(inductor_config, "fx_graph_remote_cache", None)
  monkeypatch.setattr(inductor_config, "autotune_local_cache", True)
  monkeypatch.setattr(inductor_config, "use_fast_math", False)
  monkeypatch.setattr(inductor_config, "cuda.use_fast_math", False)
  set_compile_configs(
    descent_tuning=False,
    force_disable_compile_caches=True,
    use_fast_math=True,
  )
  assert inductor_config.force_disable_caches is True
  assert inductor_config.fx_graph_cache is False
  assert inductor_config.fx_graph_remote_cache is False
  assert inductor_config.autotune_local_cache is False
  assert inductor_config.use_fast_math is True
  assert inductor_config.cuda.use_fast_math is True
