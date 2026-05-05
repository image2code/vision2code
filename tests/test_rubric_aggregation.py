from vision2code.evaluation.dataset_rubrics import DATASET_RUBRICS, aggregate_rating
from vision2code.evaluation.generic_rubric import aggregate_generic_rating

def test_dataset_rubric_aggregation_smoke():
    rubric=DATASET_RUBRICS['ChartQA']
    parsed={'category_scores':{c['id']:4.0 for c in rubric['categories']},'rationales':{c['id']:'ok' for c in rubric['categories']},'strengths':['clear'],'issues':['minor'],'overall_summary':'good'}
    result=aggregate_rating(parsed,dataset_name='ChartQA')
    assert result['rubric_dataset']=='ChartQA'
    assert 0.0 <= result['final_rating_0_to_5'] <= 5.0

def test_generic_rubric_aggregation_smoke():
    parsed={'category_scores':{'core_information_fidelity':3.0,'structure_layout_fidelity':3.0,'text_annotation_accuracy':3.0,'visual_completeness_cleanliness':3.0},'rationales':{'core_information_fidelity':'ok','structure_layout_fidelity':'ok','text_annotation_accuracy':'ok','visual_completeness_cleanliness':'ok'},'strengths':[],'issues':[],'overall_summary':'ok'}
    assert aggregate_generic_rating(parsed)['final_rating_0_to_5']==3.0
