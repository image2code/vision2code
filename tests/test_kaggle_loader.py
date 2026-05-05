from pathlib import Path
from vision2code.data.load_kaggle_dataset import check_layout, load_manifest_csv
from vision2code.data.validate_manifest import validate_kaggle_dir

def test_fixture_kaggle_loads():
    root=Path(__file__).resolve().parents[1]/'data'/'fixture_kaggle'
    assert check_layout(root)==[]
    assert len(load_manifest_csv(root))==3
    assert validate_kaggle_dir(root,allow_small=True)==[]
