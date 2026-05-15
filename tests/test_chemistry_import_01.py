import pandas as pd

from tta.preprocessing.chemistry_import_01 import (
    ChemistryImportConfig,
    run_chemistry_import,
)


def test_chemistry_import_applies_duplicate_priority_after_filters(tmp_path):
    chem = pd.DataFrame(
        {
            "NAME": ["W1", "W1", "W1", "W1"],
            "EVENT": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "ANALYTE": ["Analyte", "Analyte", "Analyte", "Analyte"],
            "ANALYTE_ORG": [
                "Chromium",
                "Hexavalent Chromium",
                "Hexavalent Chromium",
                "Hexavalent Chromium",
            ],
            "DUPLICATE_PRIORITY": [0, 1, 1, 1],
            "FILTERED": ["Y", "N", "N", "N"],
            "VAL": [10.0, 5.0, 6.0, 7.0],
            "LABQ": ["", "", "", ""],
            "UNIT": ["ug/L", "ug/L", "ug/L", "ug/L"],
            "MDL": [1.0, 1.0, 1.0, 1.0],
            "REVIEWQ": ["", "", "", ""],
            "VALIDATION_QUALIFIER": ["", "", "", ""],
            "COLLECTION_PURPOSE": ["", "", "", ""],
        }
    )

    chem_path = tmp_path / "prepared.parquet"
    chem.to_parquet(chem_path, index=False)

    stage = pd.DataFrame(
        {
            "EVENT": pd.date_range("2024-01-01", "2024-01-03", freq="D"),
            "PRD": [100.0, 101.0, 102.0],
        }
    )

    dist = pd.DataFrame({"NAME": ["W1"], "DIST": [10.0]})
    stagedist = pd.DataFrame({"NAME": ["W1"], "STAGE": ["PRD"]})
    well = pd.DataFrame(
        {
            "NAME": ["W1"],
            "XCOORDS": [0.0],
            "YCOORDS": [0.0],
            "ZCOORDS": [0.0],
            "OU": ["OU1"],
        }
    )
    screen = pd.DataFrame({"NAME": ["W1"], "TOP": [10.0], "BOT": [0.0]})

    out = run_chemistry_import(
        chem_files=[chem_path],
        stage_comb=stage,
        dist=dist,
        stagedist=stagedist,
        well=well,
        screen=screen,
        yr=2024,
        cfg=ChemistryImportConfig(trend_min_year=2024),
    )
    obs = out.loc[out["VAL"].notna()].copy()

    assert len(obs) == 3

    jan1 = obs.loc[obs["EVENT"] == pd.Timestamp("2024-01-01")]

    assert len(jan1) == 1
    assert jan1["VAL"].iloc[0] == 5.0
