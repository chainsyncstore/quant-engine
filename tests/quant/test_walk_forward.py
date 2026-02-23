import numpy as np
import pytest

from quant.validation.walk_forward import _iter_purged_kfold_splits


def test_purged_kfold_splits_apply_embargo():
    total_bars = 120
    embargo = 3
    splits = _iter_purged_kfold_splits(total_bars=total_bars, n_splits=6, embargo_bars=embargo)

    assert len(splits) == 6

    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        test_start = int(test_idx.min())
        test_end = int(test_idx.max()) + 1
        purge_low = max(0, test_start - embargo)
        purge_high = min(total_bars, test_end + embargo)

        assert not np.any((train_idx >= purge_low) & (train_idx < purge_high))


def test_purged_kfold_splits_validate_args():
    with pytest.raises(ValueError):
        _iter_purged_kfold_splits(total_bars=100, n_splits=1, embargo_bars=2)
