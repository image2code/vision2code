from image2code.data.build_manifest import load_filtered_manifest, validate_filtered_manifests

def test_filtered_manifest_counts_and_subset():
    assert len(load_filtered_manifest('test-mini'))==539
    assert len(load_filtered_manifest('test'))==2169
    assert validate_filtered_manifests()==[]

def test_no_absolute_paths_in_filtered_manifests():
    for split in ['test-mini','test']:
        for row in load_filtered_manifest(split):
            for key in ['question_dir','metadata_path']:
                assert not str(row.get(key) or '').startswith('/')
