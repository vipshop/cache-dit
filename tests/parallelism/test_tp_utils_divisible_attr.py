"""
Minimal runnable test script (non-pytest style), consistent with other files in `tests/`.

Run:
  python3 tests/parallelism/test_tp_utils_divisible_attr.py
"""

from cache_dit.parallelism.transformers.native_pytorch.tensor_parallelism.tp_utils import (
    shard_divisible_attr,
)


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_shard_divisible_attr_success_updates_value():
    attn = Dummy(heads=30)
    new_heads = shard_divisible_attr(attn, "heads", 5, what="attn", context="test")
    assert new_heads == 6
    assert attn.heads == 6


def test_shard_divisible_attr_raises_on_not_divisible():
    attn = Dummy(heads=30)
    try:
        shard_divisible_attr(attn, "heads", 4, what="attn", context="test")
        raise AssertionError("Expected ValueError for non-divisible heads/tp_size, but got none.")
    except ValueError as e:
        # should be a clear, startup-time error message
        msg = str(e)
        assert "tp_size=4" in msg
        assert "heads=30" in msg


def test_shard_divisible_attr_raises_on_missing_attr():
    attn = Dummy()
    try:
        shard_divisible_attr(attn, "heads", 2, what="attn", context="test")
        raise AssertionError("Expected AttributeError for missing attr, but got none.")
    except AttributeError:
        pass


def main():
    tests = [
        test_shard_divisible_attr_success_updates_value,
        test_shard_divisible_attr_raises_on_not_divisible,
        test_shard_divisible_attr_raises_on_missing_attr,
    ]

    print("== cache-dit TP utils self-check ==")
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"Summary: passed={passed}, failed={failed}, total={len(tests)}")
    if failed != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
