import pandas as pd

from tta.preprocessing.tobit_prep_04 import apply_manual_trend_breaks


def test_apply_manual_trend_breaks_assigns_term_by_date_range():
    df = pd.DataFrame(
        {
            "NAME": ["W1", "W1", "W1"],
            "EVENT": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "TERM": [1, 1, 1],
        }
    )

    breaks = pd.DataFrame(
        {
            "NAME": ["W1"],
            "TREND": [2],
            "START": ["01/01/2021"],
            "END": ["01/01/2022"],
        }
    )

    out = apply_manual_trend_breaks(df, breaks)

    assert out["TERM"].tolist() == [1, 2, 1]
