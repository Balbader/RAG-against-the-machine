from typing import List, Dict, Any
from ..models.data_models import MinimalSource


def calculate_overlap(source1: MinimalSource, source2: MinimalSource) -> float:
    """Calculate overlap percentage between two sources"""
    if source1.file_path != source2.file_path:
        return 0.0

    # Calculate character overlap
    start = max(source1.first_character_index, source2.first_character_index)
    end = min(source1.last_character_index, source2.last_character_index)

    if start >= end:
        return 0.0

    overlap_length = end - start
    source1_length =\
        source1.last_character_index - source1.first_character_index

    if source1_length == 0:
        return 0.0

    return overlap_length / source1_length


def calculate_recall_at_k(
    retrieved_sources: List[MinimalSource],
    correct_sources: List[MinimalSource],
    overlap_threshold: float = 0.05
) -> float:
    """Calculate recall@k for a single question"""
    if not correct_sources:
        return 1.0  # No correct sources to find

    found_sources = 0
    for correct_source in correct_sources:
        for retrieved_source in retrieved_sources:
            if calculate_overlap(retrieved_source, correct_source) >=\
                    overlap_threshold:
                found_sources += 1
                break

    return found_sources / len(correct_sources)


def evaluate_dataset_recall(search_results_file: str,
                            ground_truth_file: str) -> float:
    """Evaluate recall@k on entire dataset"""
    import json

    with open(search_results_file, 'r') as f:
        search_results = json.load(f)

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    # Create lookup for ground truth
    truth_lookup = {}
    for question in ground_truth['rag_questions']:
        if 'sources' in question:
            truth_lookup[question['question_id']] = [
                MinimalSource(**source) for source in question['sources']
            ]

    total_recall = 0
    valid_questions = 0

    for result in search_results['search_results']:
        question_id = result['question_id']
        if question_id in truth_lookup:
            retrieved = [MinimalSource(**source)
                            for source in result['retrieved_sources']]
            correct = truth_lookup[question_id]

            recall = calculate_recall_at_k(retrieved, correct)
            total_recall += recall
            valid_questions += 1

    return total_recall / valid_questions if valid_questions > 0 else 0.0
