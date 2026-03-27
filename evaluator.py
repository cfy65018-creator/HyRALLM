"""
Evaluation module for summarization tasks.
Provides standard NLP metric calculation protocols commonly used in code summarization literature, 
including token-level F1, ROUGE-L, BLEU-4, and METEOR.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn.functional as F
import string


def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0]


    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()


    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def calculate_token_f1(generated_text: str, reference_text: str) -> Dict[str, float]:
    def preprocess_and_tokenize(text: str) -> set:
                if not text:
            return set()


        text = text.lower()



        translator = str.maketrans('', '', string.punctuation.replace('_', '').replace('-', ''))
        text = text.translate(translator)


        tokens = set(text.split())


        tokens.discard('')

        return tokens


    gen_tokens = preprocess_and_tokenize(generated_text)
    ref_tokens = preprocess_and_tokenize(reference_text)


    if len(gen_tokens) == 0 and len(ref_tokens) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


    intersection = gen_tokens & ref_tokens


    precision = len(intersection) / len(gen_tokens)
    recall = len(intersection) / len(ref_tokens)


    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }






def calculate_rouge_simple(predictions: List[str], references: List[str]) -> Dict[str, float]:
    def lcs_length(s1_words: List[str], s2_words: List[str]) -> int:
                m, n = len(s1_words), len(s2_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1_words[i-1] == s2_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            rougeL_scores.append(0.0)
            continue


        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        lcs_len = lcs_length(pred_words, ref_words)

        if len(pred_words) > 0 and len(ref_words) > 0:
            precision = lcs_len / len(pred_words)
            recall = lcs_len / len(ref_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            rougeL_scores.append(f1)
        else:
            rougeL_scores.append(0.0)


    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    return {
        'rougeL_f1': avg_rougeL
    }


def calculate_rouge_single(prediction: str, reference: str) -> float:
    result = calculate_rouge_simple([prediction], [reference])
    return result['rougeL_f1']






def calculate_bleu4(predictions: List[str], references: List[str]) -> float:
    try:

        preds = [pred.lower().split() for pred in predictions]
        refs = [[ref.lower().split()] for ref in references]


        return corpus_bleu(refs, preds)

    except ImportError:
        print("  ⚠️ NLTK is not installed, cannot compute BLEU-4")
        print("  Tip: run 'pip install nltk'")
        return 0.0
    except Exception as e:
        print(f"  ⚠️ BLEU-4 calculation failed: {e}")
        return 0.0


def calculate_bleu4_single(prediction: str, reference: str) -> float:
    return calculate_bleu4([prediction], [reference])






def calculate_meteor(predictions: List[str], references: List[str]) -> float:
    try:

        from meteor_hf import calculate_meteor_hf

        return calculate_meteor_hf(
            predictions=predictions,
            references=references,
            alpha=0.9,
            beta=3.0,
            gamma=0.5
        )

    except ImportError:

        print("  ℹ️ meteor_hf module not found, falling back to direct NLTK calculation...")

        try:
            from nltk.translate.meteor_score import single_meteor_score
            from nltk import word_tokenize
            import nltk
            from packaging import version
            import importlib.metadata as importlib_metadata


            custom_nltk_path = '/home/jovyan/work/nltk_data'
            if custom_nltk_path not in nltk.data.path:
                nltk.data.path.insert(0, custom_nltk_path)


            NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))


            scores = []
            for pred, ref in zip(predictions, references):
                if NLTK_VERSION >= version.parse("3.6.5"):

                    score = single_meteor_score(
                        word_tokenize(ref),
                        word_tokenize(pred),
                        alpha=0.9,
                        beta=3.0,
                        gamma=0.5
                    )
                else:

                    score = single_meteor_score(
                        ref,
                        pred,
                        alpha=0.9,
                        beta=3.0,
                        gamma=0.5
                    )
                scores.append(score)

            return np.mean(scores)

        except ImportError:
            print("  ⚠️ NLTK is not installed, cannot compute METEOR")
            print("  Tip: run 'pip install nltk'")
            return 0.0

    except Exception as e:
        print(f"  ⚠️ METEOR calculation failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        return 0.0


def calculate_meteor_single(prediction: str, reference: str) -> float:
    return calculate_meteor([prediction], [reference])






def calculate_metrics(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ['rougeL', 'bleu4']
) -> Dict[str, float]:
        results = {}

    if 'rougeL' in metrics or 'rouge' in metrics:
        rouge_result = calculate_rouge_simple(predictions, references)
        results['rougeL_f1'] = rouge_result['rougeL_f1']

    if 'bleu4' in metrics or 'bleu' in metrics:
        bleu4_score = calculate_bleu4(predictions, references)
        results['bleu4'] = bleu4_score






    return results


def calculate_metrics_single(
    prediction: str,
    reference: str,
    metrics: List[str] = ['rougeL', 'bleu4']
) -> Dict[str, float]:
        return calculate_metrics([prediction], [reference], metrics)






def evaluate_generation_quality(
    predictions: List[str],
    references: List[str],
    detailed: bool = False
) -> Dict[str, Any]:

    overall_metrics = calculate_metrics(predictions, references)

    result = {
        'overall': overall_metrics,
        'num_samples': len(predictions)
    }


    if detailed:
        detailed_scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            sample_metrics = calculate_metrics_single(pred, ref)
            detailed_scores.append({
                'index': i,
                'prediction': pred,
                'reference': ref,
                'scores': sample_metrics
            })
        result['detailed'] = detailed_scores

    return result





if __name__ == '__main__':
    test_evaluator()





