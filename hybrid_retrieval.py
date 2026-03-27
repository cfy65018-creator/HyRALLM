"""
Hybrid Retrieval Module.
Responsible for fusing scores from parallel Dense and Sparse (BM25) retreival efforts, 
normalizing the confidence coefficients into a cohesive selection index.
"""

from typing import List, Dict

def get_document_id(result: Dict) -> str:
    for id_field in ['docid', 'doc_id', 'id', 'index']:
        if id_field in result:
            return str(result[id_field])


    content = result.get('content', result.get('text', ''))
    if content:
        return str(hash(content))

    return str(hash(frozenset(result.items())))


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return scores

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def combine_retrieval_results(
    sparse_results: List[Dict],
    dense_results: List[Dict],
    alpha: float = 0.5,
    top_k: int = 50
) -> List[Dict]:
        sparse_scores = {}
    dense_scores = {}


    sparse_score_list = [result['score'] for result in sparse_results]
    normalized_sparse = normalize_scores(sparse_score_list)

    for result, norm_score in zip(sparse_results, normalized_sparse):
        doc_id = get_document_id(result)
        sparse_scores[doc_id] = norm_score


    dense_score_list = [result['score'] for result in dense_results]
    normalized_dense = normalize_scores(dense_score_list)

    for result, norm_score in zip(dense_results, normalized_dense):
        doc_id = get_document_id(result)
        dense_scores[doc_id] = norm_score


    all_docs = set(sparse_scores.keys()) | set(dense_scores.keys())
    combined_results = []

    for doc_id in all_docs:
        sparse_score = sparse_scores.get(doc_id, 0.0)
        dense_score = dense_scores.get(doc_id, 0.0)


        final_score = alpha * sparse_score + (1 - alpha) * dense_score


        doc_content = None
        for result in sparse_results:
            if get_document_id(result) == doc_id:
                doc_content = result
                break

        if doc_content is None:
            for result in dense_results:
                if get_document_id(result) == doc_id:
                    doc_content = result
                    break

        if doc_content:
            combined_result = doc_content.copy()
            combined_result['final_score'] = final_score
            combined_result['sparse_score'] = sparse_score
            combined_result['dense_score'] = dense_score
            combined_results.append(combined_result)


    combined_results.sort(key=lambda x: x['final_score'], reverse=True)

    return combined_results[:top_k]


def perform_hybrid_retrieval_fusion(
    dense_results: List[List[Dict]],
    sparse_results: List[List[Dict]],
    alpha: float = 0.5,
    topk: int = 50
) -> List[List[Dict]]:
        print("  🔀 Fusing retrieval results...")
    hybrid_results = []

    total_queries = len(dense_results)

    for i in range(total_queries):
        dense_query_results = dense_results[i] if i < len(dense_results) else []
        sparse_query_results = sparse_results[i] if i < len(sparse_results) else []

        combined = combine_retrieval_results(sparse_query_results, dense_query_results, alpha, topk)
        hybrid_results.append(combined)

        if (i + 1) % 100 == 0:
            print(f"    Fusion progress: {i + 1}/{total_queries}")

    return hybrid_results


def reciprocal_rank_fusion(
    sparse_results: List[Dict],
    dense_results: List[Dict],
    k: float = 60.0,
    top_k: int = 50
) -> List[Dict]:
        doc_scores = {}


    for rank, result in enumerate(sparse_results):
        doc_id = get_document_id(result)
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {'sparse_rrf': 0.0, 'dense_rrf': 0.0, 'doc_info': result}
        doc_scores[doc_id]['sparse_rrf'] = rrf_score


    for rank, result in enumerate(dense_results):
        doc_id = get_document_id(result)
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {'sparse_rrf': 0.0, 'dense_rrf': 0.0, 'doc_info': result}
        doc_scores[doc_id]['dense_rrf'] = rrf_score


    final_results = []
    for doc_id, scores in doc_scores.items():
        final_score = scores['sparse_rrf'] + scores['dense_rrf']
        result = scores['doc_info'].copy()
        result['final_score'] = final_score
        result['sparse_rrf'] = scores['sparse_rrf']
        result['dense_rrf'] = scores['dense_rrf']
        final_results.append(result)


    final_results.sort(key=lambda x: x['final_score'], reverse=True)

    return final_results[:top_k]




