from __future__ import annotations

import pandas as pd

from quant_v2.research.retrain_pipeline import _build_labels


def test_legacy_retrain_labels_drop_unlabeled_tail_rows() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})

    labels = _build_labels(frame, horizon=2)

    assert labels.iloc[:3].notna().all()
    assert labels.iloc[-2:].isna().all()
