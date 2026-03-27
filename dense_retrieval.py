"""
Dense Retrieval Module.
Implements vector database operations leveraging Faiss for GPU/CPU accelerated semantic search 
over embedded code snippets and summaries.
"""

import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import DataLoader


# ============================================================
# ============================================================

class VectorDatabase:
    
    
    def __init__(self, vectors: np.ndarray):
        
        self.vectors = vectors.astype(np.float32)
        self.n_docs, self.embedding_dim = self.vectors.shape
        print(f"  📚 Vector DB initialized: {self.n_docs} docs, {self.embedding_dim} dims")
    
    def __len__(self):
        return self.n_docs
    
    def search(self, query_vectors: np.ndarray, top_k: int, exclude_self: bool = False) -> List[List[int]]:
        
        n_queries = query_vectors.shape[0]
        print(f"  🔍 Running dense retrieval (numpy dot product, exclude_self={exclude_self})...")
        print(f"    Queries: {n_queries}, Docs: {self.n_docs}, Dim: {self.embedding_dim}")
        
        batch_size = 50
        results = []
        k = top_k + 1 if exclude_self else top_k
        
        for batch_start in range(0, n_queries, batch_size):
            batch_end = min(batch_start + batch_size, n_queries)
            query_batch = query_vectors[batch_start:batch_end]
            
            scores = np.matmul(query_batch, self.vectors.T)  # (batch_size, n_docs)
            
            for i in range(scores.shape[0]):
                if self.n_docs > 10000:
                    top_indices = np.argpartition(-scores[i], k-1)[:k]
                    top_indices = top_indices[np.argsort(-scores[i][top_indices])]
                else:
                    top_indices = np.argsort(-scores[i])[:k]
                
                if exclude_self:
                    results.append(top_indices[1:].tolist())
                else:
                    results.append(top_indices.tolist())
            
            print(f"    Retrieval progress: {batch_end}/{n_queries} ({batch_end/n_queries*100:.1f}%)")
        
        print(f"  ✅ Retrieval complete, processed {len(results)} queries")
        return results
    
    @staticmethod
    def build_from_model(model, tokenizer, texts: List[str], batch_size: int = 32, 
                        max_length: int = 256, device: str = 'cuda') -> 'VectorDatabase':
        
        model.to(device)
        model.eval()
        
        dataloader = DataLoader(
            texts, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: tokenizer(
                list(batch), 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            )
        )
        
        vectors = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                vec = model.sentence_emb(input_ids, attention_mask)
                vectors.append(vec.cpu().numpy())
                
                if (i + 1) % 10 == 0:
                    print(f"    Build progress: {(i+1)*batch_size}/{len(texts)}")
        
        vectors = np.concatenate(vectors, axis=0)
        return VectorDatabase(vectors)


class FaissVectorDatabase(VectorDatabase):
    
    
    def __init__(self, vectors: np.ndarray, use_gpu: bool = False):
        
        super().__init__(vectors)
        
        try:
            import faiss
        except ImportError:
            raise ImportError("Faiss未安装。请使用: pip install faiss-cpu 或 pip install faiss-gpu")
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        print(f"  🚀 Building Faiss index (GPU: {self.use_gpu})")
        
        if self.n_docs < 10000:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.n_docs < 100000:
            nlist = min(int(np.sqrt(self.n_docs)), 4096)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        else:
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(self.vectors)
        
        self.index.add(self.vectors)
        print(f"  ✅ Faiss index built")
    
    def search(self, query_vectors: np.ndarray, top_k: int, exclude_self: bool = False) -> List[List[int]]:
        
        query_vectors = query_vectors.astype(np.float32)
        
        k = top_k + 1 if exclude_self else top_k
        scores, indices = self.index.search(query_vectors, k)
        
        results = []
        for i in range(len(indices)):
            if exclude_self:
                results.append(indices[i][1:].tolist())
            else:
                results.append(indices[i].tolist())
        
        return results


# ============================================================
# ============================================================

def build_dense_vectors(model, tokenizer, codes, batch_size=32, max_source_length=256, 
                       device='cuda', use_faiss=False, use_gpu=False):
    
    print(f"  📊 Building vector DB ({'Faiss-GPU' if use_faiss and use_gpu else 'Faiss-CPU' if use_faiss else 'Numpy'})")
    
    if use_faiss:
        vector_db = VectorDatabase.build_from_model(
            model, tokenizer, codes, batch_size, max_source_length, device
        )
        vector_db = FaissVectorDatabase(vector_db.vectors, use_gpu=use_gpu)
    else:
        vector_db = VectorDatabase.build_from_model(
            model, tokenizer, codes, batch_size, max_source_length, device
        )
    
    return vector_db


def dense_retrieval(query_vector_db, db_vector_db, topk=50, exclude_self=False):
    
    indices_list = db_vector_db.search(query_vector_db.vectors, topk, exclude_self=exclude_self)
    
    results = []
    for indices in indices_list:
        query_results = [{"docid": f"doc_{idx}", "score": 1.0} for idx in indices]
        results.append(query_results)
    
    return results


def perform_dense_retrieval(query_codes, db_vector_db, model, tokenizer, topk, batch_size, 
                           max_source_length, device, exclude_self=False, use_faiss=False, use_gpu=False):
    
    print("  📝 Building query vectors...")
    
    query_vector_db = build_dense_vectors(
        model, tokenizer, query_codes, 
        batch_size=batch_size, max_source_length=max_source_length, 
        device=device, use_faiss=use_faiss, use_gpu=use_gpu
    )
    
    dense_results = dense_retrieval(query_vector_db, db_vector_db, topk, exclude_self=exclude_self)
    
    print(f"\n📈 Retrieval summary:")
    print(f"  - Dense retrieval results: {len(dense_results)}")
    
    return dense_results


if __name__ == '__main__':
    print("🧪 Testing dense retrieval module...")
    
    test_vectors = np.random.randn(100, 768).astype(np.float32)
    
    print("\n=== Test Numpy version ===")
    db = VectorDatabase(test_vectors)
    query = np.random.randn(5, 768).astype(np.float32)
    results = db.search(query, top_k=10)
    print(f"Retrieval results: {len(results)} queries, each returns {len(results[0])} results")
    
    results_exclude = db.search(test_vectors[:5], top_k=10, exclude_self=True)
    print(f"Exclude self retrieval: first result index={results_exclude[0][0]} (should not be 0)")
    
    print("\n✅ Test complete")
