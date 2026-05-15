import pandas as pd

from tta.preprocessing.prepare_chromium_chemistry import prepare_chromium_chemistry


def test_prepare_chromium_keeps_filtered_chromium_and_hexchrom(tmp_path):
    raw = pd.DataFrame(
        {
            "NAME": ["W1", "W1", "W1"],
            "EVENT": ["01/01/2024 00:00:00"] * 3,
            "LOAD_DATE_TIME": ["01/02/2024 00:00:00"] * 3,
            "ANALYTE": ["Chromium", "Chromium", "Hexavalent Chromium"],
            "FILTERED": ["Y", "N", "N"],
            "VAL": [10.0, 99.0, 5.0],
            "MDL": [1.0, 1.0, 1.0],
            "DEPTH": [None, None, None],
        }
    )

    path = tmp_path / "chem.txt"
    raw.to_csv(path, index=False)

    out = prepare_chromium_chemistry(
        chem_files=[path],
        year=2024,
    )

    assert len(out) == 2
    assert set(out["ANALYTE_ORG"]) == {"Chromium", "Hexavalent Chromium"}
    assert (out["ANALYTE"] == "Hex. Chromium and Fil. Chromium").all()
    assert 99.0 not in set(out["VAL"])
