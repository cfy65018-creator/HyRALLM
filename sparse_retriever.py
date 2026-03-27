import os
import json
import re
import logging
import tempfile
import subprocess
from typing import List, Dict, Optional
import tree_sitter
from tree_sitter import Language, Parser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25SparseRetriever:

    def __init__(self, corpus_path: str = None, index_dir: str = None, language: str = "python"):
        self.corpus_path = corpus_path
        self.language = language.lower()


        if index_dir is None:
            import uuid
            self.index_dir = os.path.join(tempfile.gettempdir(), f"bm25_index_{self.language}_{uuid.uuid4().hex[:8]}")
        else:
            self.index_dir = index_dir

        self.parser = None
        self.tree_sitter_language = None
        self.searcher = None


        self._init_tree_sitter()


        if corpus_path:
            self._build_or_load_index()

    def _init_tree_sitter(self):
        try:
            if self.language == 'python':
                from tree_sitter_python import language
                self.tree_sitter_language = Language(language())
            elif self.language == 'java':
                from tree_sitter_java import language
                self.tree_sitter_language = Language(language())
            else:
                raise ValueError(f"Unsupported language: {self.language}")


            self.parser = Parser(self.tree_sitter_language)
            logger.info(f"Tree-sitter initialized for {self.language}")

        except ImportError as e:
            logger.warning(f"Tree-sitter not available for {self.language}: {e}")
            self.parser = None
        except Exception as e:
            logger.warning(f"Tree-sitter initialization failed: {e}")
            self.parser = None

    def _preprocess_code(self, code: str) -> str:
        if not self.parser:
            return self._enhanced_simple_preprocess(code)

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            extracted_features = []
            self._extract_enhanced_features(tree.root_node, code.encode('utf8'), extracted_features)
            return " ".join(extracted_features)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed: {e}, falling back to enhanced preprocessing")
            return self._enhanced_simple_preprocess(code)

    def _extract_enhanced_features(self, node, code_bytes: bytes, features: List[str]):
        if node.type in ['comment', 'string', 'string_literal']:
            return


        if node.type == 'identifier':
            identifier = code_bytes[node.start_byte:node.end_byte].decode('utf8')
            if len(identifier) > 1:
                features.append(identifier)

                import re
                camel_split = re.sub('([a-z0-9])([A-Z])', r'\1_\2', identifier).lower()
                if '_' in camel_split:
                    features.extend(camel_split.split('_'))


        elif node.type in ['function_definition', 'method_definition']:
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
                    features.append(f"function_{func_name}")

                    action_hints = self._extract_action_hints(func_name)
                    features.extend(action_hints)


        elif node.type == 'class_definition':
            for child in node.children:
                if child.type == 'identifier':
                    class_name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
                    features.append(f"class_{class_name}")


        elif node.type in ['if_statement', 'for_statement', 'while_statement']:
            features.append(f"control_{node.type}")
        elif node.type in ['return_statement']:
            features.append("returns")
        elif node.type in ['assignment']:
            features.append("assigns")

        for child in node.children:
            self._extract_enhanced_features(child, code_bytes, features)

    def _extract_action_hints(self, func_name: str) -> List[str]:
        action_patterns = {
            'get': ['retrieve', 'access', 'fetch'],
            'set': ['assign', 'update'],
            'add': ['insert', 'create'],
            'remove': ['delete', 'eliminate'],
            'find': ['search', 'locate'],
            'check': ['verify', 'validate'],
            'calculate': ['compute'],
            'process': ['handle'],
            'parse': ['analyze'],
            'load': ['read'],
            'save': ['write'],
            'sort': ['order'],
            'filter': ['select'],
        }

        hints = []
        func_lower = func_name.lower()
        for pattern, actions in action_patterns.items():
            if pattern in func_lower:
                hints.extend(actions[:1])
        return hints

    def _enhanced_simple_preprocess(self, code: str) -> str:
        import re


        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', '', code)
        code = re.sub(r"'[^']*'", '', code)

        features = []


        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        for identifier in identifiers:
            if len(identifier) > 1:
                features.append(identifier)

                camel_split = re.sub('([a-z0-9])([A-Z])', r'\1_\2', identifier).lower()
                if '_' in camel_split:
                    features.extend(camel_split.split('_'))


        func_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        for func_name in func_matches:
            features.append(f"function_{func_name}")
            features.extend(self._extract_action_hints(func_name))


        class_matches = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        for class_name in class_matches:
            features.append(f"class_{class_name}")

        return " ".join(features)

    def _extract_identifiers(self, node, code_bytes: bytes, identifiers: List[str]):
        if node.type in ['comment', 'string', 'string_literal']:
            return

        if node.type == 'identifier':
            identifier = code_bytes[node.start_byte:node.end_byte].decode('utf8')
            if len(identifier) > 2:
                identifiers.append(identifier)

        for child in node.children:
            self._extract_identifiers(child, code_bytes, identifiers)

    def _simple_preprocess(self, code: str) -> str:

        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)


        code = re.sub(r'"[^"]*"', '', code)
        code = re.sub(r"'[^']*'", '', code)


        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', code)
        return " ".join(identifiers)

    def _build_or_load_index(self):

        if os.path.exists(self.index_dir) and len(os.listdir(self.index_dir)) > 0:
            logger.info(f"Loading existing Pyserini BM25 index: {self.index_dir}")
            self._load_searcher()
        else:
            logger.info(f"Building new Pyserini BM25 index: {self.index_dir}")
            self._build_index()

    def _build_index(self):
        try:

            corpus_dir = os.path.join(self.index_dir, "corpus")
            os.makedirs(corpus_dir, exist_ok=True)

            json_file = os.path.join(corpus_dir, "corpus.jsonl")
            self._create_jsonl_corpus(json_file)


            self._build_index_with_pyserini(corpus_dir, self.index_dir)


            self._load_searcher()

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise RuntimeError(f"Index building failed: {e}")

    def _create_jsonl_corpus(self, json_file: str):
        logger.info(f"Creating JSONL corpus file: {json_file}")

        doc_count = 0
        with open(json_file, 'w', encoding='utf-8') as f:
            if os.path.isfile(self.corpus_path):
                with open(self.corpus_path, 'r', encoding='utf-8') as corpus_file:
                    for line_num, line in enumerate(corpus_file):
                        line = line.strip()
                        if line:
                            processed = self._preprocess_code(line)
                            doc = {
                                "id": str(line_num),
                                "contents": processed,
                                "raw": line
                            }
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                            doc_count += 1

        logger.info(f"Created JSONL corpus with {doc_count} documents")

    def _build_index_with_pyserini(self, corpus_dir: str, index_dir: str):
        logger.info("Building index with Pyserini...")


        cmd = [
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', corpus_dir,
            '--index', index_dir,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', '1'
        ]

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Index built successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Command line indexing failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")


            self._build_index_python_api(corpus_dir)

    def _build_index_python_api(self, corpus_dir: str):
        logger.info("Trying alternative indexing method using Python API...")

        try:
            from pyserini.index.lucene import LuceneIndexer


            indexer = LuceneIndexer(self.index_dir, collection='JsonCollection')


            json_file = os.path.join(corpus_dir, "corpus.jsonl")
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    indexer.add_doc(doc)


            indexer.close()
            logger.info("Index built successfully using Python API")

        except ImportError as e:
            logger.error(f"Required pyserini modules not available: {e}")
            raise ImportError(f"Cannot build index with Pyserini: {e}")
        except Exception as e:
            logger.error(f"Python API indexing failed: {e}")
            raise RuntimeError(f"All indexing methods failed: {e}")

    def _load_searcher(self):
        try:
            from pyserini.search.lucene import LuceneSearcher

            self.searcher = LuceneSearcher(self.index_dir)


            self.searcher.set_bm25(k1=1.2, b=0.75)

            logger.info(f"Loaded Pyserini searcher from {self.index_dir}")

        except Exception as e:
            logger.error(f"Failed to load searcher: {e}")
            raise RuntimeError(f"Cannot load Pyserini searcher: {e}")

    def search(self, query: str, k: int = 10, use_query_expansion: bool = True) -> List[Dict]:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        try:
            preprocessed_query = self._preprocess_code(query)


            if use_query_expansion:
                expanded_query = self._expand_query(preprocessed_query)
            else:
                expanded_query = preprocessed_query


            hits = self.searcher.search(expanded_query, k=k)

            results = []
            for i, hit in enumerate(hits):
                try:

                    doc = self.searcher.doc(hit.docid)


                    doc_content = None
                    processed_content = ""

                    if hasattr(doc, 'raw') and doc.raw():

                        try:
                            doc_dict = json.loads(doc.raw())
                            doc_content = doc_dict.get('raw', doc_dict.get('contents', ''))
                            processed_content = doc_dict.get('contents', '')
                        except (json.JSONDecodeError, TypeError):

                            doc_content = doc.raw()
                            processed_content = self._preprocess_code(doc_content)
                    elif hasattr(doc, 'contents') and doc.contents():

                        doc_content = doc.contents()
                        processed_content = self._preprocess_code(doc_content)
                    else:

                        doc_content = f"Document {hit.docid}"
                        processed_content = ""

                    result = {
                        'rank': i + 1,
                        'docid': hit.docid,
                        'score': hit.score,
                        'content': doc_content or f"Document {hit.docid}",
                        'processed_content': processed_content
                    }
                    results.append(result)

                except Exception as doc_error:
                    logger.warning(f"Failed to retrieve document {hit.docid}: {doc_error}")

                    result = {
                        'rank': i + 1,
                        'docid': hit.docid,
                        'score': hit.score,
                        'content': f"Document {hit.docid} (content unavailable)",
                        'processed_content': ""
                    }
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Pyserini BM25 search failed: {e}")
            return []

    def close(self):
        if self.searcher:
            try:
                self.searcher.close()
                logger.info("Pyserini searcher closed")
            except Exception as e:
                logger.error(f"Error closing searcher: {e}")

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def set_bm25_params(self, k1: float = 1.2, b: float = 0.75):
        logger.info(f"translatedBM25translated: k1={k1}, b={b}")
        self.searcher.set_bm25(k1, b)

    def _expand_query(self, query: str) -> str:

        code_to_nl_mapping = {

            'get': ['retrieve', 'fetch', 'obtain', 'access'],
            'set': ['assign', 'update', 'modify', 'change'],
            'add': ['insert', 'append', 'create', 'include'],
            'remove': ['delete', 'eliminate', 'erase', 'drop'],
            'find': ['search', 'locate', 'discover', 'identify'],
            'check': ['verify', 'validate', 'test', 'examine'],
            'process': ['handle', 'manage', 'execute', 'run'],
            'calculate': ['compute', 'determine', 'evaluate'],
            'parse': ['analyze', 'interpret', 'decode'],
            'load': ['read', 'import', 'open'],
            'save': ['write', 'export', 'store'],
            'sort': ['order', 'arrange', 'organize'],
            'filter': ['select', 'screen', 'refine'],
            'merge': ['combine', 'join', 'unite'],
            'split': ['divide', 'separate', 'break'],


            'list': ['array', 'sequence', 'collection'],
            'dict': ['dictionary', 'map', 'mapping', 'hash'],
            'set': ['collection', 'group'],
            'tree': ['hierarchy', 'structure'],
            'node': ['element', 'item', 'vertex'],


            'init': ['initialize', 'create', 'setup'],
            'start': ['begin', 'launch', 'initiate'],
            'stop': ['end', 'terminate', 'halt'],
            'run': ['execute', 'perform', 'operate'],
            'build': ['construct', 'create', 'generate'],
            'test': ['check', 'verify', 'validate'],
        }

        expanded_terms = []
        query_words = query.lower().split()

        for word in query_words:
            expanded_terms.append(word)


            if word in code_to_nl_mapping:
                expanded_terms.extend(code_to_nl_mapping[word][:2])


            if any(c.isupper() for c in word):

                import re
                camel_split = re.sub('([a-z0-9])([A-Z])', r'\1 \2', word).lower()
                expanded_terms.extend(camel_split.split())


        unique_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = ' '.join(unique_terms)

        logger.debug(f"translated: '{query}' -> '{expanded_query}'")
        return expanded_query

    @classmethod
    def from_texts(cls, texts: List[str], doc_ids=None, language="python", index_dir=None):

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            for text in texts:
                tmp_file.write(text.strip() + '\n')
            tmp_corpus_path = tmp_file.name

        try:

            retriever = cls(
                corpus_path=tmp_corpus_path,
                index_dir=index_dir,
                language=language
            )
            return retriever
        finally:

            if os.path.exists(tmp_corpus_path):
                os.remove(tmp_corpus_path)



if __name__ == "__main__":
    print("🧪 Testing BM25 retriever based on Pyserini...")


    test_documents = [
        "def calculate_sum(a, b): return a + b",
        "class Calculator: def add(self, x, y): return x + y",
        "def find_max(numbers): return max(numbers)",
        "def sort_list(items): return sorted(items)",
        "class DataProcessor: def process_data(self, data): return data.strip()"
    ]

    try:
        print("📝 Creating BM25 retriever...")
        retriever = BM25SparseRetriever.from_texts(
            texts=test_documents,
            language="python"
        )

        print("✅ Created successfully!")


        queries = ["calculate sum", "class add", "find max"]

        for query in queries:
            print(f"\n🔍 Search: '{query}'")
            results = retriever.search(query, k=3)

            for result in results:
                print(f"  📊 Score {result['score']:.3f}: {result['content']}")

        print("\n🎉 Test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()



