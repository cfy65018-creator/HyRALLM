"""
Main execution script for the Hybrid Retrieval-Augmented Code Summarization pipeline.
This module coordinates the complete evaluation flow including dense/sparse retrieval,
LLM-based generation, and results evaluation using standard NLP metrics.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any
import torch
from transformers import RobertaTokenizer, T5Config

from contrastlearn import ContrastiveGenerator, PairTextDataset, train_contrastive
from dense_retrieval import VectorDatabase, FaissVectorDatabase, perform_dense_retrieval
from sparse_retriever import BM25SparseRetriever
from hybrid_retrieval import perform_hybrid_retrieval_fusion
from generate import SummaryGenerator
from evaluator import (
    calculate_rouge_simple,
    calculate_bleu4,
    calculate_metrics,
    calculate_metrics_single,
    calculate_token_f1,
)
import json


# ============================================================
# ============================================================

class Config:
    """
    Central configuration class containing paths, model configs,
    training parameters, and retrieval settings.
    """
    
    
    DATASET_ROOT = '/home/jovyan/work/dataset'
    MODEL_NAME_OR_PATH = '/home/jovyan/work/models/codet5-base'
    MODEL_DIR = '/home/jovyan/work/saved_models/contrastive_encoder_JCSD_3epochs'
    RESULT_SAVE_DIR = '/home/jovyan/work'
    
    BATCH_SIZE = 16
    MAX_SOURCE_LENGTH = 256
    MAX_TARGET_LENGTH = 128
    
    RETRAIN = False
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    SEED = 42
    NUM_WORKERS = 4
    LOG_INTERVAL = 100
    
    AUGMENTATION_FACTOR = 1
    AUGMENT_CODE = False
    MLM_PROBABILITY = 0.15
    TEMPERATURE = 0.2
    
    DATASET = 'JCSD'
    EVALUATION_MODE = 'test_only'  # 'validation_only', 'test_only', 'train_only'
    TEST_LIMIT = None
    
    RETRIEVAL_METHOD = 'dense'  # 'dense', 'sparse', 'hybrid'
    TOPK = 50
    NUM_REFERENCES = 1
    
    
    USE_FAISS = False
    USE_GPU_FAISS = False
    
    BM25_K1 = 1.2
    BM25_B = 0.8
    USE_QUERY_EXPANSION = True
    
    ALPHA = 0.4
    
    NEWAPI_BASE_URL = ""
    NEWAPI_API_KEY = ""
    
    MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

    
    API_TYPE = "auto"
    
    NUM_EXAMPLES_FOR_GENERATION = 1
    
    MAX_WORKERS = 6
    
    ENABLE_INCREMENTAL_SAVE = True
    INCREMENTAL_SAVE_FILENAME = None
    AUTO_RESUME = False
    
    ENABLE_POST_PROCESSING = False
    
    LLM_POST_TEMPERATURE = 0.0
    LLM_POST_MAX_TOKENS = 5
    
    SAVE_DETAILED_RESULTS = True
    LOCAL_FILES_ONLY = True
    
    @classmethod
    def get_dataset_language(cls, dataset_name: str) -> str:
        
        language_map = {'PCSD': 'python', 'JCSD': 'java'}
        return language_map.get(dataset_name.upper(), 'java')
    
    @classmethod
    def print_config(cls):
        
        print("\n" + "="*60)
        print("⚙️  System Configuration")
        print("="*60)
        print(f"\n📚 Dataset: {cls.DATASET}")
        print(f"🎯 Evaluation mode: {cls.EVALUATION_MODE}")
        print(f"🔍 Retrieval method: {cls.RETRIEVAL_METHOD}")
        print(f"📊 Top-K: {cls.TOPK}")
        print(f"💾 Batch size: {cls.BATCH_SIZE}")
        
        if cls.RETRIEVAL_METHOD in ['dense', 'hybrid']:
            print(f"\n⚡ Dense Retrieval:")
            print(f"  - Use Faiss: {'Yes' if cls.USE_FAISS else 'No'}")
            if cls.USE_FAISS:
                print(f"  - GPU acceleration: {'Yes' if cls.USE_GPU_FAISS else 'No'}")
        
        if cls.RETRIEVAL_METHOD in ['sparse', 'hybrid']:
            print(f"\n📈 Sparse Retrieval (BM25):")
            print(f"  - k1: {cls.BM25_K1}")
            print(f"  - b: {cls.BM25_B}")
            print(f"  - Query expansion: {'Yes' if cls.USE_QUERY_EXPANSION else 'No'}")
        
        if cls.RETRIEVAL_METHOD == 'hybrid':
            print(f"\n🔀 Hybrid Retrieval:")
            print(f"  - Alpha (sparse weight): {cls.ALPHA}")
        
        print(f"\n🎯 Working Mode:")
        if cls.RETRIEVAL_ONLY_MODE:
            print(f"  - Mode: Retrieval-only (use retrieved summaries directly)")
            print(f"  - Use summary from retrieval rank #{cls.RETRIEVAL_SUMMARY_RANK}")
            print(f"  - LLM generation: disabled")
            print(f"  - Post-processing: disabled")
        else:
            print(f"  - Mode: RAG generation (retrieval + LLM generation)")
        
        if not cls.RETRIEVAL_ONLY_MODE:
            print(f"\n🤖 LLM Generation Settings:")
            print(f"  - Model: {cls.MODEL_NAME}")
            print(f"  - API type: {cls.API_TYPE}")
            print(f"  - API endpoint: {cls.NEWAPI_BASE_URL}")
            if cls.NUM_EXAMPLES_FOR_GENERATION == 1:
                print(f"  - Generation strategy: direct generation with top-1 example")
            else:
                print(f"  - Generation strategy: Multi-Generation (generate from each of top-{cls.NUM_EXAMPLES_FOR_GENERATION} examples)")
                print(f"  - Selection metric: token-level F1")
            
            print(f"\n⏱️  API Concurrency Settings:")
            print(f"  - Worker threads: {cls.MAX_WORKERS}")
            
            if cls.ENABLE_POST_PROCESSING:
                print(f"\n🔧 Post-processing Settings:")
                print(f"  - Enabled: Yes")
                print(f"  - Mode: LLM evaluation")
                print(f"  - LLM decision: evaluate semantic consistency and information completeness of retrieved summaries")
                print(f"  - LLM Temperature: {cls.LLM_POST_TEMPERATURE}")
                print(f"  - LLM Max Tokens: {cls.LLM_POST_MAX_TOKENS}")
            else:
                print(f"\n🔧 Post-processing Settings: disabled")
        
        print("="*60 + "\n")


# ============================================================
# ============================================================

def load_contrastive_encoder(model_name_or_path, load_path, local_files_only=True):
    
    print(f"Loading trained contrastive encoder from {load_path}.")
    
    model_dir = os.path.dirname(load_path)
    tokenizer_loaded = False
    
    if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
        try:
            print(f"Loading tokenizer from saved model directory: {model_dir}")
            tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
            tokenizer_loaded = True
            print(f"✅ Tokenizer loaded from model directory")
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer from model directory: {e}")
            tokenizer_loaded = False
    
    if not tokenizer_loaded:
        print(f"Loading tokenizer from base model path: {model_name_or_path}")
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"  - Vocab size: {len(tokenizer)}")
    
    state_dict = torch.load(load_path, map_location='cpu')
    saved_vocab_size = state_dict['shared.weight'].shape[0]
    
    print(f"Saved model vocab size: {saved_vocab_size}")
    
    config = T5Config.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    config.vocab_size = saved_vocab_size
    
    print(f"Creating model with vocab size: {config.vocab_size}")
    model = ContrastiveGenerator(config, tokenizer)
    
    if len(tokenizer) != saved_vocab_size:
        print(f"⚠️ Tokenizer vocab size ({len(tokenizer)}) does not match saved model ({saved_vocab_size})")
        print(f"Resizing embedding layer to match saved model...")
        model.resize_token_embeddings(saved_vocab_size)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Trying non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}...")
        print("✅ Model loaded (non-strict mode)")
        
    model.eval()
    return model, tokenizer


def build_dense_vectors(model, tokenizer, codes, batch_size=32, max_source_length=256, device='cuda', use_faiss=False, use_gpu=False):
    
    print(f"   Building vector database ({'Faiss-GPU' if use_faiss and use_gpu else 'Faiss-CPU' if use_faiss else 'Numpy'})")
    
    if use_faiss:
        vector_db = FaissVectorDatabase.build_from_model(
            model, tokenizer, codes, batch_size, max_source_length, device
        )
        vector_db = FaissVectorDatabase(vector_db.vectors, use_gpu=use_gpu)
    else:
        vector_db = VectorDatabase.build_from_model(
            model, tokenizer, codes, batch_size, max_source_length, device
        )
    
    return vector_db


def sparse_retrieval_bm25(query_codes, db_codes, topk=50, language='python', k1=1.2, b=0.75, use_query_expansion=True):
    
    print(f"  🔍 Running sparse retrieval with BM25 (query expansion: {'on' if use_query_expansion else 'off'})...")
    
    bm25_retriever = BM25SparseRetriever.from_texts(db_codes, language=language)
    
    bm25_retriever.set_bm25_params(k1=k1, b=b)
    
    results = []
    for i, query in enumerate(query_codes):
        query_results = bm25_retriever.search(query, k=topk, use_query_expansion=use_query_expansion)
        formatted_results = []
        for result in query_results:
            formatted_results.append({
                "docid": f"doc_{result['docid']}",
                "score": float(result['score'])
            })
        results.append(formatted_results)
        
        if (i + 1) % 100 == 0:
            print(f"  BM25 retrieval progress: {i + 1}/{len(query_codes)}")
    
    print(f"  ✅ BM25 retrieval complete, processed {len(query_codes)} queries")
    return results


def get_eval_split_info(evaluation_mode: str):
    
    split_map = {
        'validation_only': ('valid', '验证集'),
        'test_only': ('test', '测试集'),
        'train_only': ('train', '训练集')
    }
    
    if evaluation_mode not in split_map:
        raise ValueError(f"不支持的评估模式: {evaluation_mode}")
    
    return split_map[evaluation_mode]


def load_datasets(dataset_root: str, dataset: str, evaluation_mode: str, test_limit: int = None):
    
    eval_split, eval_name = get_eval_split_info(evaluation_mode)
    
    print(f"📁 Loading datasets...")
    db_set = PairTextDataset(dataset_root, dataset, 'train', augmentation_factor=1, augment_code=False)
    eval_set = PairTextDataset(dataset_root, dataset, eval_split, augmentation_factor=1, augment_code=False)
    
    db_codes, db_texts = db_set.codes, db_set.texts
    eval_codes, eval_texts = eval_set.codes, eval_set.texts
    
    if test_limit and test_limit < len(eval_codes):
        print(f"  ⚠️ Limiting {eval_name} to first {test_limit} samples")
        eval_codes = eval_codes[:test_limit]
        eval_texts = eval_texts[:test_limit]
    
    print(f"  Retrieval database: {len(db_codes)} samples")
    print(f"  {eval_name}: {len(eval_codes)} samples")
    
    return db_codes, db_texts, eval_codes, eval_texts, eval_split, eval_name


def perform_hybrid_retrieval(query_codes, db_vector_db, db_codes, model, tokenizer, topk, batch_size, max_source_length, device, language, alpha=0.5, bm25_k1=1.2, bm25_b=0.75, use_query_expansion=True, exclude_self=False, use_faiss=False, use_gpu=False):
    
    print("  Running hybrid retrieval...")
    
    dense_results = perform_dense_retrieval(query_codes, db_vector_db, model, tokenizer, topk, batch_size, max_source_length, device, exclude_self, use_faiss, use_gpu)
    
    sparse_results = sparse_retrieval_bm25(query_codes, db_codes, topk, language, k1=bm25_k1, b=bm25_b, use_query_expansion=use_query_expansion)
    
    hybrid_results = perform_hybrid_retrieval_fusion(dense_results, sparse_results, alpha, topk)
    
    print(f"\n📈 Retrieval statistics:")
    print(f"  - Hybrid retrieval results: {len(hybrid_results)}")
    
    return hybrid_results, dense_results, sparse_results


def retrieval_only(
    model, tokenizer, dataset_root, dataset, evaluation_mode='test_only', 
    topk=50, max_source_length=256, batch_size=32, test_limit=None,
    use_faiss=False, use_gpu_faiss=False, retrieval_method='dense', alpha=0.5, 
    bm25_k1=1.2, bm25_b=0.75, use_query_expansion=True
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    language = Config.get_dataset_language(dataset)
    
    db_codes, db_texts, eval_codes, eval_texts, eval_split, eval_name = load_datasets(
        dataset_root, dataset, evaluation_mode, test_limit
    )
    
    print(f"\n🚀 Starting retrieval on {eval_name}")
    print(f"  - Dataset: {dataset} ({language})")
    print(f"  - Retrieval method: {retrieval_method}")
    if retrieval_method == 'hybrid':
        print(f"  - Fusion weights: sparse={alpha:.2f}, dense={1-alpha:.2f}")
    print(f"  - Top-K: {topk}")
    if test_limit:
        print(f"  - Test sample limit: {test_limit}")
    print(f"  - Device: {device}")
    
    if retrieval_method == 'dense':
        print("\n🧮 Building dense vector representations...")
        print("  Building database vectors...")
        db_vectors = build_dense_vectors(
            model, tokenizer, db_codes, batch_size, max_source_length, 
            device, use_faiss, use_gpu_faiss
        )
        
        print(f"\n🔧 Running dense retrieval on {eval_name}...")
        results = perform_dense_retrieval(
            eval_codes, db_vectors, model, tokenizer, topk, batch_size, 
            max_source_length, device, exclude_self=False, 
            use_faiss=use_faiss, use_gpu=use_gpu_faiss
        )
        
    elif retrieval_method == 'sparse':
        print(f"\n🔧 Running sparse retrieval on {eval_name}...")
        results = sparse_retrieval_bm25(
            eval_codes, db_codes, topk, language, 
            k1=bm25_k1, b=bm25_b, use_query_expansion=use_query_expansion
        )
        
    elif retrieval_method == 'hybrid':
        print("\n🧮 Building dense vector representations...")
        print("  Building database vectors...")
        db_vectors = build_dense_vectors(
            model, tokenizer, db_codes, batch_size, max_source_length, 
            device, use_faiss, use_gpu_faiss
        )
        
        print(f"\n🔧 Running hybrid retrieval on {eval_name}...")
        results, dense_results, sparse_results = perform_hybrid_retrieval(
            eval_codes, db_vectors, db_codes, model, tokenizer, topk, batch_size, 
            max_source_length, device, language, alpha, 
            bm25_k1=bm25_k1, bm25_b=bm25_b, use_query_expansion=use_query_expansion,
            exclude_self=False, use_faiss=use_faiss, use_gpu=use_gpu_faiss
        )
    else:
        raise ValueError(f"不支持的检索方法: {retrieval_method}")
    
    print(f"\n✅ Retrieval complete, processed {len(results)} queries")
    
    return results, eval_codes, eval_texts, db_codes, db_texts


# ============================================================
# ============================================================

def evaluate_retrieval_only(
    results, eval_codes, eval_texts, db_codes, db_texts, 
    retrieval_summary_rank=1
):
    
    print(f"\n📋 Using retrieved summaries directly (retrieval-only mode)...")
    print(f"  - Using summary from retrieval rank #{retrieval_summary_rank}")
    
    retrieved_summaries = []
    detailed_results = []
    
    print(f"\n📊 Extracting retrieval results...")
    for i, (query_code, query_results) in enumerate(zip(eval_codes, results)):
        query_ref_summary = eval_texts[i]
        
        if len(query_results) >= retrieval_summary_rank:
            result_item = query_results[retrieval_summary_rank - 1]
            
            if isinstance(result_item, dict) and 'docid' in result_item:
                doc_idx = int(result_item['docid'].split('_')[1])
            else:
                doc_idx = result_item
            
            retrieved_summary = db_texts[doc_idx]
            retrieved_code = db_codes[doc_idx]
        else:
            if len(query_results) == 0:
                print(f"  ❌ Sample {i}: retrieval result is empty, skipping")
                retrieved_summary = ""
                retrieved_code = ""
            else:
                print(f"  ⚠️ Sample {i}: fewer than {retrieval_summary_rank} retrieval results, using the last one")
                result_item = query_results[-1]
                if isinstance(result_item, dict) and 'docid' in result_item:
                    doc_idx = int(result_item['docid'].split('_')[1])
                else:
                    doc_idx = result_item
                retrieved_summary = db_texts[doc_idx]
                retrieved_code = db_codes[doc_idx]
        
        retrieved_summaries.append(retrieved_summary)
        
        individual_metrics = calculate_metrics_single(retrieved_summary, query_ref_summary)
        
        detailed_results.append({
            'query_code': query_code,
            'reference_text': query_ref_summary,
            'retrieved_summary': retrieved_summary,
            'retrieved_code': retrieved_code,
            'retrieved_index': doc_idx,
            'retrieval_rank': retrieval_summary_rank,
            'scores': {
                'rougeL_f1': individual_metrics['rougeL_f1'],
                'bleu4': individual_metrics['bleu4']
            }
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed: {i + 1}/{len(eval_codes)}")
    
    print(f"  ✅ Retrieval extraction complete")
    
    print(f"\n📊 Calculating overall evaluation metrics...")
    metrics = calculate_metrics(retrieved_summaries, eval_texts)
    
    print(f"\n🎉 Retrieval-only evaluation results:")
    print(f"  - ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
    print(f"  - BLEU-4: {metrics['bleu4']:.4f}")
    
    return metrics, detailed_results


def generate_and_evaluate(
    results, eval_codes, eval_texts, db_codes, db_texts, 
    num_examples_for_generation=3,
    api_base_url="", 
    api_key="", model_name="",
    api_type="openai",
    enable_incremental_save=True,
    incremental_save_path=None,
    auto_resume=True
):
    
    print(f"\n🤖 Starting LLM summary generation...")
    print(f"  - Model: {model_name}")
    print(f"  - API type: {api_type}")
    if num_examples_for_generation == 1:
        print(f"  - Generation strategy: direct generation with top-1 example")
    else:
        print(f"  - Generation strategy: Multi-Generation (generate separately for top-{num_examples_for_generation} examples)")
        print(f"  - Selection metric: token-level F1 (fixed)")
    
    if enable_incremental_save:
        if incremental_save_path is None:
            if auto_resume:
                import glob
                all_files = glob.glob("generated_incremental_*.json")
                existing_files = [f for f in all_files if 'post_processed' not in f]
                
                if existing_files:
                    existing_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    latest_file = existing_files[0]
                    
                    print(f"\n📂 Found {len(existing_files)} unfinished save files:")
                    for i, f in enumerate(existing_files[:3], 1):
                        mtime = os.path.getmtime(f)
                        from datetime import datetime
                        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                        size_kb = os.path.getsize(f) / 1024
                        print(f"  {i}. {f} (modified: {time_str}, size: {size_kb:.1f} KB)")
                    
                    incremental_save_path = latest_file
                    print(f"\n✅ Auto-selected newest file to resume: {incremental_save_path}")
                else:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    incremental_save_path = f"generated_incremental_{timestamp}.json"
                    print(f"  - ✅ Incremental save enabled, created new file: {incremental_save_path}")
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                incremental_save_path = f"generated_incremental_{timestamp}.json"
                print(f"  - ✅ Incremental save enabled, created new file: {incremental_save_path}")
                print(f"  - ℹ️  AUTO_RESUME=False, not resuming historical incremental files")
        else:
            print(f"  - ✅ Incremental save enabled: {incremental_save_path}")
    
    completed_indices = set()
    if enable_incremental_save and os.path.exists(incremental_save_path):
        print(f"\n📂 Found unfinished save file, loading completed samples...")
        try:
            with open(incremental_save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                completed_indices = set(saved_data.get('completed_indices', []))
                print(f"  - ✅ Loaded {len(completed_indices)} completed samples")
        except Exception as e:
            print(f"  - ⚠️  Load failed: {e}, starting from scratch")
    
    generator = SummaryGenerator(
        api_base_url=api_base_url,
        api_key=api_key,
        model=model_name,
        api_type=api_type
    )
    
    detailed_results = []
    
    print(f"\n📊 Step 1: Extract top-{num_examples_for_generation} retrieval results for each query...")
    
    for i, (query_code, query_results) in enumerate(zip(eval_codes, results)):
        query_ref_summary = eval_texts[i]
        
        top_n_codes = []
        top_n_summaries = []
        top_n_indices = []
        
        for j in range(min(num_examples_for_generation, len(query_results))):
            result_item = query_results[j]
            if isinstance(result_item, dict) and 'docid' in result_item:
                doc_idx = int(result_item['docid'].split('_')[1])
            else:
                doc_idx = result_item
            
            top_n_indices.append(doc_idx)
            top_n_codes.append(db_codes[doc_idx])
            top_n_summaries.append(db_texts[doc_idx])
        
        detailed_results.append({
            'query_code': query_code,
            'reference_text': query_ref_summary,
            'top_n_indices': top_n_indices,
            'top_n_codes': top_n_codes,
            'top_n_summaries': top_n_summaries
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed: {i + 1}/{len(eval_codes)}")
    
    print(f"  ✅ Retrieval extraction complete")
    
    total_samples = len(eval_codes)
    samples_to_process = total_samples - len(completed_indices)
    
    print(f"\n🔄 Step 2: Generate summaries per query (total: {total_samples}, completed: {len(completed_indices)}, pending: {samples_to_process})...")
    if num_examples_for_generation == 1:
        print(f"  - Generation mode: direct generation with top-1 example")
    else:
        print(f"  - Generation mode: generate {num_examples_for_generation} times per query (one for each top-{num_examples_for_generation} example)")
        print(f"  - Selection metric: token-level F1 (fixed)")
    
    generated_summaries = [None] * total_samples
    
    def save_incremental_result(idx, summary, scores):
        
        if not enable_incremental_save:
            return
        
        try:
            if os.path.exists(incremental_save_path):
                with open(incremental_save_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                if 'results' not in saved_data:
                    saved_data['results'] = []
                if 'completed_indices' not in saved_data:
                    saved_data['completed_indices'] = []
                if 'metadata' not in saved_data:
                    saved_data['metadata'] = {
                        'model': model_name,
                        'api_type': api_type,
                        'num_examples_for_generation': num_examples_for_generation,
                        'total_samples': total_samples
                    }
            else:
                saved_data = {
                    'completed_indices': [],
                    'results': [],
                    'metadata': {
                        'model': model_name,
                        'api_type': api_type,
                        'num_examples_for_generation': num_examples_for_generation,
                        'total_samples': total_samples
                    }
                }
            
            result_entry = {
                'index': idx,
                'query': {
                    'code': detailed_results[idx]['query_code'],
                    'reference_summary': detailed_results[idx]['reference_text']
                },
                'selected_example': {
                    'example_idx': detailed_results[idx].get('selected_example_idx', 0),
                    'code': detailed_results[idx]['top_n_codes'][detailed_results[idx].get('selected_example_idx', 0)],
                    'summary': detailed_results[idx]['top_n_summaries'][detailed_results[idx].get('selected_example_idx', 0)]
                },
                'generated': {
                    'summary': summary,
                    'scores': scores
                }
            }
            
            result_exists = False
            for i, r in enumerate(saved_data['results']):
                if r['index'] == idx:
                    saved_data['results'][i] = result_entry
                    result_exists = True
                    break
            
            if not result_exists:
                saved_data['results'].append(result_entry)
                saved_data['completed_indices'].append(idx)
            
            with open(incremental_save_path, 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"  ⚠️  Save failed (sample {idx}): {e}")
    
    import time
    processed_count = 0
    start_time = time.time()
    
    completed_results = {}
    if completed_indices and os.path.exists(incremental_save_path):
        try:
            print(f"  📥 Preloading completed sample results...")
            with open(incremental_save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                for r in saved_data['results']:
                    completed_results[r['index']] = r
            print(f"  ✅ Preloaded {len(completed_results)} completed samples")
        except Exception as e:
            print(f"  ⚠️  Preload failed: {e}")
    
    for i in range(total_samples):
        if i in completed_indices:
            if i in completed_results:
                r = completed_results[i]
                if 'generated' in r and 'summary' in r['generated'] and 'scores' in r['generated']:
                    generated_summaries[i] = r['generated']['summary']
                    detailed_results[i]['generated_summary'] = r['generated']['summary']
                    detailed_results[i]['generation_scores'] = r['generated']['scores']
                else:
                    print(f"  ⚠️  Sample {i} data is incomplete, skipping restore and regenerating")
                    completed_indices.discard(i)
                    continue
            else:
                print(f"  ⚠️  Sample {i} marked as completed but result not found")
            continue
        
        try:
            query_code = eval_codes[i]
            query_ref = eval_texts[i]
            top_n_codes = detailed_results[i]['top_n_codes']
            top_n_summaries = detailed_results[i]['top_n_summaries']
            
            if num_examples_for_generation == 1:
                print(f"\n  🔄 Sample {i}: generate directly with top-1 example...")
                
                try:
                    example_code = top_n_codes[0]
                    example_summary = top_n_summaries[0]
                    
                    summary = generator.generate(
                        source_code=query_code,
                        example_code=example_code,
                        example_summary=example_summary
                    )
                    
                    individual_metrics = calculate_metrics([summary], [query_ref])
                    
                    print(f"    ✅ Generation completed")
                    
                    generated_summaries[i] = summary
                    detailed_results[i]['generated_summary'] = summary
                    detailed_results[i]['generation_scores'] = {
                        'rougeL_f1': individual_metrics['rougeL_f1'],
                        'bleu4': individual_metrics['bleu4']
                    }
                    detailed_results[i]['selected_example_idx'] = 0
                    
                    save_incremental_result(i, summary, detailed_results[i]['generation_scores'])
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    ❌ Generation failed: {e}")
                    generated_summaries[i] = ""
                    detailed_results[i]['generated_summary'] = ""
                    detailed_results[i]['generation_scores'] = {
                        'rougeL_f1': 0.0,
                        'bleu4': 0.0
                    }
                    detailed_results[i]['selected_example_idx'] = 0
                    
            else:
                candidate_summaries = []
                candidate_scores = []
                
                print(f"\n  🔄 Sample {i}: generating separately for {len(top_n_codes)} examples...")
                
                for j, (example_code, example_summary) in enumerate(zip(top_n_codes, top_n_summaries)):
                    try:
                        summary = generator.generate(
                            source_code=query_code,
                            example_code=example_code,
                            example_summary=example_summary
                        )
                        
                        individual_metrics = calculate_metrics([summary], [query_ref])
                        token_f1_result = calculate_token_f1(summary, example_summary)
                        score = token_f1_result['f1']
                        
                        candidate_summaries.append(summary)
                        candidate_scores.append({
                            'summary': summary,
                            'score': score,
                            'example_idx': j,
                            'metrics': {
                                'rougeL_f1': individual_metrics.get('rougeL_f1', 0.0),
                                'bleu4': individual_metrics.get('bleu4', 0.0),
                                'example_token_f1_score': score
                            }
                        })
                        
                        print(f"    - Example {j+1}: token-level f1={score:.4f}")
                        
                    except Exception as e:
                        print(f"    ❌ Example {j+1} generation failed: {e}")
                        candidate_summaries.append("")
                        candidate_scores.append({
                            'summary': "",
                            'score': 0.0,
                            'example_idx': j,
                            'metrics': {
                                'rougeL_f1': 0.0, 
                                'bleu4': 0.0,
                                'example_token_f1_score': 0.0
                            }
                        })
                
                if candidate_scores:
                    best_candidate = max(candidate_scores, key=lambda x: x['score'])
                    best_summary = best_candidate['summary']
                    best_metrics = best_candidate['metrics']
                    best_example_idx = best_candidate['example_idx']
                    
                    print(f"    ✅ Selected generation from example {best_example_idx+1} (token-level f1={best_candidate['score']:.4f})")
                    
                    generated_summaries[i] = best_summary
                    detailed_results[i]['generated_summary'] = best_summary
                    detailed_results[i]['generation_scores'] = best_metrics
                    detailed_results[i]['selected_example_idx'] = best_example_idx
                    detailed_results[i]['all_candidates'] = candidate_scores
                    
                    save_incremental_result(i, best_summary, best_metrics)
                    
                    processed_count += 1
                else:
                    raise Exception("所有示例生成均失败")
            
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / processed_count if processed_count > 0 else 0
            remaining_samples = samples_to_process - processed_count
            estimated_remaining_time = avg_time_per_sample * remaining_samples
            
            if processed_count == 1 or processed_count % 5 == 0 or processed_count == samples_to_process:
                print(f"\n  📊 Overall progress: {processed_count}/{samples_to_process} "
                      f"({processed_count/samples_to_process*100:.1f}%), "
                      f"avg: {avg_time_per_sample:.1f}s/sample, "
                      f"ETA: {estimated_remaining_time/3600:.1f} hours")
        
        except Exception as e:
            print(f"  ❌ Sample {i} generation failed: {e}")
            generated_summaries[i] = ""
            detailed_results[i]['generated_summary'] = ""
            detailed_results[i]['generation_scores'] = {
                'rougeL_f1': 0.0,
                'bleu4': 0.0
            }
    
    print(f"  ✅ Generation completed: {processed_count}/{samples_to_process} new samples")

    post_processed_count = 0
    post_processed_indices = []
    
    if Config.ENABLE_POST_PROCESSING:
        print(f"\n🔍 Step 2.5: LLM post-processing evaluation (whether retrieved summary should replace generated summary)...")
        
        try:
            from llm_postprocessor import LLMPostProcessor, apply_llm_postprocessing
            
            processor = LLMPostProcessor(
                api_base_url=Config.NEWAPI_BASE_URL,
                api_key=Config.NEWAPI_API_KEY,
                model_name=Config.MODEL_NAME,
                temperature=Config.LLM_POST_TEMPERATURE,
                max_tokens=Config.LLM_POST_MAX_TOKENS,
                max_workers=Config.MAX_WORKERS
            )
            
            generated_summaries, post_processed_count, post_processed_indices = apply_llm_postprocessing(
                detailed_results,
                generated_summaries,
                processor,
                verbose=True
            )
            
            if post_processed_count > 0:
                print(f"  ✅ LLM post-processing completed: {post_processed_count}/{len(generated_summaries)} samples replaced")
            else:
                print(f"  ✅ LLM post-processing completed: no replacements required")
                
        except Exception as e:
            print(f"  ❌ LLM post-processing failed: {e}")
            print(f"  ℹ️  Skipping post-processing step")
    else:
        print(f"  ℹ️  Post-processing is disabled, skipping this step")
    
    for i, summary in enumerate(generated_summaries):
        if summary is None:
            continue
        detailed_results[i]['generated_summary'] = summary
    
    if Config.ENABLE_POST_PROCESSING and post_processed_count > 0 and enable_incremental_save and os.path.exists(incremental_save_path):
        print(f"\n💾 Step 2.6: Updating post-processed results in incremental save file...")
        try:
            with open(incremental_save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            for idx in post_processed_indices:
                for result in saved_data['results']:
                    if result['index'] == idx:
                        result['generated']['summary'] = generated_summaries[idx]
                        result['generated']['scores'] = detailed_results[idx]['generation_scores']
                        result['post_processed'] = True
                        result['replacement_reason'] = detailed_results[idx]['replacement_reason']
                        break
            
            if 'metadata' not in saved_data:
                saved_data['metadata'] = {}
            saved_data['metadata']['post_processed_count'] = post_processed_count
            saved_data['metadata']['post_processed_indices'] = post_processed_indices
            
            with open(incremental_save_path, 'w', encoding='utf-8') as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Incremental save file updated: {incremental_save_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to update incremental save file: {e}")
    
    print(f"\n📊 Step 3: Calculating overall evaluation metrics...")
    metrics = calculate_metrics(generated_summaries, eval_texts)
    
    
    print(f"\n🎉 Generation evaluation results:")
    print(f"  - ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
    print(f"  - BLEU-4: {metrics['bleu4']:.4f}")
    
    return metrics, detailed_results, generator



# ============================================================
# ============================================================

if __name__ == '__main__':
    Config.print_config()
    
    config = {
        'retrain': Config.RETRAIN,
        'model_name_or_path': Config.MODEL_NAME_OR_PATH,
        'model_dir': Config.MODEL_DIR,
        'dataset_root': Config.DATASET_ROOT,
        'dataset': Config.DATASET,
        'evaluation_mode': Config.EVALUATION_MODE,
        'test_limit': Config.TEST_LIMIT,
        'topk': Config.TOPK,
        'num_references': Config.NUM_REFERENCES,
        'batch_size': Config.BATCH_SIZE,
        'max_source_length': Config.MAX_SOURCE_LENGTH,
        'save_detailed_results': Config.SAVE_DETAILED_RESULTS,
        'use_faiss': Config.USE_FAISS,
        'use_gpu_faiss': Config.USE_GPU_FAISS,
        'retrieval_method': Config.RETRIEVAL_METHOD,
        'alpha': Config.ALPHA,
        'bm25_k1': Config.BM25_K1,
        'bm25_b': Config.BM25_B,
        'use_query_expansion': Config.USE_QUERY_EXPANSION,
        'result_save_dir': Config.RESULT_SAVE_DIR,
        'newapi_base_url': Config.NEWAPI_BASE_URL,
        'newapi_api_key': Config.NEWAPI_API_KEY,
        'model_name': Config.MODEL_NAME,
        'api_type': Config.API_TYPE
    }
    
    if config['retrain']:
        print("🚀 Starting model training...")
        
        class TrainingArgs:
            def __init__(self):
                self.dataset_root = config['dataset_root']
                self.dataset = config['dataset']
                self.model_name_or_path = config['model_name_or_path']
                self.output_dir = config['model_dir']
                
                self.split = 'train'
                
                self.epochs = Config.NUM_EPOCHS
                self.batch_size = config['batch_size']
                self.lr = Config.LEARNING_RATE
                self.weight_decay = Config.WEIGHT_DECAY
                self.warmup_ratio = Config.WARMUP_RATIO
                self.max_source_length = config['max_source_length']
                self.max_target_length = Config.MAX_TARGET_LENGTH
                
                self.augmentation_factor = Config.AUGMENTATION_FACTOR
                self.augment_code = Config.AUGMENT_CODE
                self.mlm_probability = Config.MLM_PROBABILITY
                self.temperature = Config.TEMPERATURE
                
                self.seed = Config.SEED
                self.num_workers = Config.NUM_WORKERS
                self.log_interval = Config.LOG_INTERVAL
                self.local_files_only = Config.LOCAL_FILES_ONLY
        
        args = TrainingArgs()
        
        print(f"\n🏋️ Starting contrastive learning training...")
        train_contrastive(args)
        
        print(f"\n✅ Training complete! Model saved to: {config['model_dir']}")
        
        model_path = os.path.join(config['model_dir'], 'pytorch_model.bin')
        print(f"\n📂 Loading trained model for evaluation: {model_path}")
        model, tokenizer = load_contrastive_encoder(
            model_name_or_path=config['model_name_or_path'],
            load_path=model_path
        )
    else:
        model_path = os.path.join(config['model_dir'], 'pytorch_model.bin')
        print(f"\n📂 Loading model: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Error: model file does not exist: {model_path}")
            print(f"💡 Please train the model first (set Config.RETRAIN = True) or check the model path")
            print(f"📂 Current model directory: {config['model_dir']}")
            if os.path.exists(config['model_dir']):
                print(f"📁 Directory contents: {os.listdir(config['model_dir'])}")
            else:
                print(f"📁 Model directory does not exist")
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        model, tokenizer = load_contrastive_encoder(
            model_name_or_path=config['model_name_or_path'],
            load_path=model_path
        )
    
    if config['evaluation_mode'] in ['validation_only', 'test_only', 'train_only']:
        print(f"\n🎯 Step 1: Execute retrieval...")
        
        results, eval_codes, eval_texts, db_codes, db_texts = retrieval_only(
                model=model,
                tokenizer=tokenizer,
                dataset_root=config['dataset_root'],
                dataset=config['dataset'],
                evaluation_mode=config['evaluation_mode'],
                topk=config['topk'],
                batch_size=config['batch_size'],
                max_source_length=config['max_source_length'],
                test_limit=config['test_limit'],
                use_faiss=config['use_faiss'],
                use_gpu_faiss=config['use_gpu_faiss'],
                retrieval_method=config['retrieval_method'],
                alpha=config['alpha'],
                bm25_k1=config['bm25_k1'],
                bm25_b=config['bm25_b'],
            use_query_expansion=config['use_query_expansion']
        )
        
        if config['evaluation_mode'] == 'validation_only':
            result_type = "validation"
        elif config['evaluation_mode'] == 'test_only':
            result_type = "test"
        else:  # train_only
            result_type = "train"
        
        if Config.RETRIEVAL_ONLY_MODE:
            print(f"\n🎯 Step 2: Retrieval-only mode - use retrieved summaries directly...")
            
            scores, detailed_results = evaluate_retrieval_only(
                results=results,
                eval_codes=eval_codes,
                eval_texts=eval_texts,
                db_codes=db_codes,
                db_texts=db_texts,
                retrieval_summary_rank=Config.RETRIEVAL_SUMMARY_RANK
            )
            
            
        else:
            print(f"\n🎯 Step 2: Generate summaries with LLM and evaluate...")
            
            scores, detailed_results, generator = generate_and_evaluate(
                results=results,
                eval_codes=eval_codes,
                eval_texts=eval_texts,
                db_codes=db_codes,
                db_texts=db_texts,
                num_examples_for_generation=Config.NUM_EXAMPLES_FOR_GENERATION,
                api_base_url=config['newapi_base_url'],
                api_key=config['newapi_api_key'],
                model_name=config['model_name'],
                api_type=config['api_type'],
                auto_resume=Config.AUTO_RESUME
            )
            
            import glob
            all_files = glob.glob("generated_incremental_*.json")
            existing_files = [f for f in all_files if 'post_processed' not in f]
            incremental_save_path = existing_files[0] if existing_files else None
            
        
    else:
        raise ValueError(
            f"不支持的评估模式: {config['evaluation_mode']}. "
            f"支持的模式: 'validation_only', 'test_only', 'train_only'"
        )
    
    if config['save_detailed_results'] and detailed_results:
        mode_prefix = "retrieval_only" if Config.RETRIEVAL_ONLY_MODE else "generated"
        result_filename = (
            f"{mode_prefix}_{config['retrieval_method']}_{result_type}_"
            f"{config['dataset']}_rougeL_{scores['rougeL_f1']:.4f}_bleu4_{scores['bleu4']:.4f}.json"
        )
        result_path = os.path.join(config['result_save_dir'], result_filename)
        
        if Config.RETRIEVAL_ONLY_MODE:
            save_data = {
                'overall_scores': scores,
                'detailed_results': detailed_results,
                'config': {
                    'mode': 'retrieval_only',
                    'dataset': config['dataset'],
                    'retrieval_method': config['retrieval_method'],
                    'evaluation_mode': config['evaluation_mode'],
                    'topk': config['topk'],
                    'retrieval_summary_rank': Config.RETRIEVAL_SUMMARY_RANK,
                    'alpha': config['alpha'] if config['retrieval_method'] == 'hybrid' else None,
                    'bm25_k1': config['bm25_k1'] if config['retrieval_method'] in ['sparse', 'hybrid'] else None,
                    'bm25_b': config['bm25_b'] if config['retrieval_method'] in ['sparse', 'hybrid'] else None,
                }
            }
        else:
            save_data = {
                'overall_scores': scores,
                'detailed_results': detailed_results,
                'config': {
                    'mode': 'rag_generation',
                    'dataset': config['dataset'],
                    'retrieval_method': config['retrieval_method'],
                    'evaluation_mode': config['evaluation_mode'],
                    'topk': config['topk'],
                    'num_examples_for_generation': Config.NUM_EXAMPLES_FOR_GENERATION,
                    'prompt_strategy': 'multi_generation',
                    'llm_model': config['model_name'],
                    'api_type': config['api_type'],
                    'alpha': config['alpha'] if config['retrieval_method'] == 'hybrid' else None,
                    'bm25_k1': config['bm25_k1'] if config['retrieval_method'] in ['sparse', 'hybrid'] else None,
                    'bm25_b': config['bm25_b'] if config['retrieval_method'] in ['sparse', 'hybrid'] else None,
                }
            }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved: {result_path}")
    
    print("\n" + "="*60)
    if Config.RETRIEVAL_ONLY_MODE:
        print("🏆 Retrieval-only Evaluation Summary")
    elif Config.NUM_EXAMPLES_FOR_GENERATION == 1:
        print("🏆 Generation Evaluation Summary (Direct Top-1)")
    else:
        print("🏆 Generation Evaluation Summary (Multi-Generation)")
    print("="*60)
    print(f"\n📊 Evaluation Configuration:")
    print(f"  - Evaluation mode: {config['evaluation_mode']}")
    print(f"  - Dataset: {config['dataset']}")
    print(f"  - Retrieval method: {config['retrieval_method']}")
    print(f"  - Test samples: {len(eval_codes)}")
    print(f"  - Retrieval Top-K: {config['topk']}")
    
    if Config.RETRIEVAL_ONLY_MODE:
        print(f"  - Working mode: retrieval-only (use retrieval results directly)")
        print(f"  - Using summary from retrieval rank #{Config.RETRIEVAL_SUMMARY_RANK}")
    elif Config.NUM_EXAMPLES_FOR_GENERATION == 1:
        print(f"  - Generation strategy: direct generation with top-1 example")
        print(f"  - Prompt strategy: direct Top-1 generation")
    else:
        print(f"  - Generation strategy: generate separately with top-{Config.NUM_EXAMPLES_FOR_GENERATION} examples per query")
        print(f"  - Prompt strategy: Multi-Generation")
    
    if not Config.RETRIEVAL_ONLY_MODE:
        print(f"  - LLM model: {config['model_name']}")
        print(f"  - API type: {config['api_type']}")
    
    if config['retrieval_method'] == 'hybrid':
        print(f"  - Fusion weights: sparse={config['alpha']:.2f}, dense={1-config['alpha']:.2f}")
    
    if config['retrieval_method'] in ['dense', 'hybrid']:
        if config['use_faiss']:
            print(f"  - Dense retrieval: Faiss-{'GPU' if config['use_gpu_faiss'] else 'CPU'}")
        else:
            print(f"  - Dense retrieval: Numpy (reference-project style)")
    
    if config['retrieval_method'] in ['sparse', 'hybrid']:
        print(f"  - BM25 params: k1={config['bm25_k1']}, b={config['bm25_b']}")
        print(f"  - Query expansion: {'enabled' if config['use_query_expansion'] else 'disabled'}")
    
    print(f"\n📏 Generation Quality Metrics:")
    print(f"  - ROUGE-L F1: {scores['rougeL_f1']:.4f}")
    print(f"  - BLEU-4: {scores['bleu4']:.4f}")
    
    if config['save_detailed_results'] and detailed_results:
        print(f"\n📈 Detailed Statistics:")
        print(f"  - Total samples: {len(detailed_results)}")
        
        if Config.RETRIEVAL_ONLY_MODE:
            rouge_scores_list = [r['scores'].get('rougeL_f1', 0.0) for r in detailed_results]
            bleu_scores_list = [r['scores'].get('bleu4', 0.0) for r in detailed_results]
        else:
            rouge_scores_list = [r['generation_scores'].get('rougeL_f1', 0.0) for r in detailed_results]
            bleu_scores_list = [r['generation_scores'].get('bleu4', 0.0) for r in detailed_results]
        
        import numpy as np
        print(f"\n  Generation quality distribution:")
        print(f"    ROUGE-L F1:")
        print(f"      - Min: {min(rouge_scores_list):.4f}")
        print(f"      - Median: {np.median(rouge_scores_list):.4f}")
        print(f"      - Max: {max(rouge_scores_list):.4f}")
        print(f"      - Std: {np.std(rouge_scores_list):.4f}")
        
        print(f"    BLEU-4:")
        print(f"      - Min: {min(bleu_scores_list):.4f}")
        print(f"      - Median: {np.median(bleu_scores_list):.4f}")
        print(f"      - Max: {max(bleu_scores_list):.4f}")
        print(f"      - Std: {np.std(bleu_scores_list):.4f}")
        
        high_rouge_count = sum(1 for s in rouge_scores_list if s >= 0.4)
        high_rouge_pct = high_rouge_count / len(rouge_scores_list) * 100
        print(f"\n  High-quality samples (ROUGE-L >= 0.4): {high_rouge_count}/{len(rouge_scores_list)} ({high_rouge_pct:.1f}%)")
    
    print("="*60)
    print("\n✅ Program execution completed!")    