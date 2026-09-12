import ast
import json
from pathlib import Path


def test_multispectral_trust_notebook_is_clean_and_complete():
    root = Path(__file__).resolve().parents[1]
    path = root / "kaggle/GeoDiff_TrustMoE_Landsat_Sentinel_RGB_Multispectral_128_3x.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    for required in (
        "TRAIN_LR_CROP = 128",
        "'--include-multispectral'",
        "condition_key='lr_ms'",
        "target_key='hr_ms'",
        "dense_transformer",
        "sparse_transformer",
        "sparse_no_trust",
        "spectral_index_study.csv",
        "def show_test_index(index=0",
        "bundle_results",
    ):
        assert required in source
