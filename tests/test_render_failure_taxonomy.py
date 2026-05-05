from vision2code.rendering.failure_taxonomy import classify_failure

def test_failure_taxonomy_examples():
    assert classify_failure('timeout')=='timeout'
    assert classify_failure('SyntaxError: invalid syntax')=='syntax_truncation'
    assert classify_failure('ModuleNotFoundError: no module named foo')=='missing_dependency_file'
    assert classify_failure('AttributeError: module has no attribute bar')=='hallucinated_api_argument'
    assert classify_failure('ValueError: shape mismatch')=='shape_3d_geometry'
