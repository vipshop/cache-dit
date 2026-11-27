"""CLI coverage for `--cache-params`."""

import pytest

from cache_dit.cli import utils as cli_utils

parser = cli_utils.get_args(parse=False)


def resolve(argv):
  return cli_utils.maybe_postprocess_args(parser.parse_args(argv))


def test_defaults_are_written_back_without_the_flag():
  """The DBCache defaults stay readable even when no cache is requested."""

  args = resolve([])
  assert args.cache is False
  assert args.cache_type is None
  assert args.Fn_compute_blocks == 1
  assert args.Bn_compute_blocks == 0
  assert args.residual_diff_threshold == 0.12
  assert (args.max_warmup_steps, args.warmup_interval) == (8, 1)
  assert (args.max_cached_steps, args.max_continuous_cached_steps) == (-1, 3)
  assert (args.steps_mask, args.mask_policy) == (False, None)
  assert (args.seacache, args.seacache_mode, args.softcache) == (False, "flow", False)


def test_type_selects_the_cache():
  """One flag enables the cache; `type` seeds the SeaCache and SoftCache switches."""

  db = resolve(["--cache-params", "type=DBCache"])
  assert (db.cache, db.cache_type, db.seacache, db.softcache) == (True, "dbcache", False, False)

  sea = resolve(["--cache-params", "type=SeaCache"])
  assert (sea.cache, sea.cache_type, sea.seacache, sea.softcache) == (True, "seacache", True, False)

  soft = resolve(["--cache-params", "type=SoftCache"])
  assert (soft.cache, soft.cache_type, soft.seacache, soft.softcache) == (True, "softcache", True,
                                                                          True)


def test_short_and_long_knob_names_are_equivalent():
  """`Fn`/`rdt`/...

  and the DBCache field names reach the same attributes.
  """

  short = resolve(["--cache-params", "Fn=2,Bn=1,rdt=0.2,ws=4,wi=2,mc=5,mcc=1,scm=f"])
  long = resolve([
    "--cache-params",
    "Fn_compute_blocks=2,Bn_compute_blocks=1,residual_diff_threshold=0.2,max_warmup_steps=4,"
    "warmup_interval=2,max_cached_steps=5,max_continuous_cached_steps=1,mask_policy=fast"
  ])
  fields = (
    "Fn_compute_blocks",
    "Bn_compute_blocks",
    "residual_diff_threshold",
    "max_warmup_steps",
    "warmup_interval",
    "max_cached_steps",
    "max_continuous_cached_steps",
    "mask_policy",
  )
  assert {key: getattr(short, key) for key in fields} == {key: getattr(long, key) for key in fields}
  assert short.mask_policy == "fast"


def test_mask_policy_implies_the_steps_mask():
  """A policy used to imply the mask switch, and still does; aliases are expanded."""

  args = resolve(["--cache-params", "scm=s"])
  assert (args.steps_mask, args.mask_policy) == (True, "slow")


def test_sea_and_soft_extras():
  """SeaCache and SoftCache options land on their plain attributes."""

  args = resolve([
    "--cache-params",
    "type=SoftCache,seacache_mode=flow,seacache_feature_shape=8x8x2,seacache_power_exp=3,"
    "softcache_tau=0.4,softcache_jf_max=6"
  ])
  assert args.seacache_feature_shape == [8, 8, 2]
  assert (args.seacache_power_exp, args.softcache_tau, args.softcache_jf_max) == (3.0, 0.4, 6.0)


def test_db_cache_conflict_with_config_path_is_allowed():
  """Only the SeaCache options conflict with a YAML config, as before."""

  args = resolve(["--cache-params", "rdt=0.2", "--config-path", "some.yaml"])
  assert (args.cache, args.residual_diff_threshold) == (True, 0.2)


def test_postprocess_is_idempotent():
  """Resolving the params twice keeps the same attributes."""

  once = resolve(["--cache-params", "rdt=0.2,Fn=2"])
  twice = cli_utils.maybe_postprocess_args(once)
  assert (once.cache, once.residual_diff_threshold, once.Fn_compute_blocks) == (True, 0.2, 2)
  assert (twice.cache, twice.residual_diff_threshold, twice.Fn_compute_blocks) == (True, 0.2, 2)


@pytest.mark.parametrize("argv", [
  ["--cache-params", ""],
  ["--cache-params", "type=nope"],
])
def test_argparse_rejects_a_bad_selector(argv):
  """The flag is the switch, so an empty or unknown type fails while parsing."""

  with pytest.raises(SystemExit):
    parser.parse_args(argv)


@pytest.mark.parametrize("argv, match", [
  (["--cache-params", "bogus=1"], "Unknown --cache-params key 'bogus'"),
  (["--cache-params", "fn=1,Fn_compute_blocks=2"], "given twice"),
  (["--cache-params", "seacache_mode=vp,softcache=true"], "requires SeaCache"),
  (["--cache-params", "type=SoftCache,seacache_mode=vp"], "seacache_mode=flow"),
  (["--cache-params", "type=SoftCache,softcache_tau=0"], "must be positive"),
  (["--cache-params", "seacache_feature_shape=8x8"], "requires a SeaCache"),
  (["--cache-params", "type=SeaCache,seacache_feature_shape=0x8"], "positive"),
  (["--cache-params", "type=SeaCache", "--config-path", "some.yaml"
    ], "cannot be combined with --config-path"),
])
def test_invalid_options_are_rejected(argv, match):
  """Bad values fail before the model is loaded."""

  with pytest.raises(ValueError, match=match):
    resolve(argv)
