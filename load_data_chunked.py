#!/usr/bin/env python3
"""
GitHub to Milvus Chunked Vector Loader
=====================================

This enhanced script experiments with different chunking strategies:
- By file (original approach)
- By function/method
- By class
- By semantic blocks
- By fixed size
- By logical sections

Author: AI Assistant
Date: 2024
"""

import os
import json
import time
import logging
import re
import ast
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import requests
from dataclasses import dataclass
import hashlib
import base64
from collections import defaultdict
from enum import Enum

# Third-party imports
try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, MilvusException
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from scipy.sparse import csr_matrix
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"Missing required dependencies. Please install: {e}")
    print("Run: pip install pymilvus scikit-learn numpy scipy pandas requests sentence-transformers torch")
    exit(1)

# Configuration
MILVUS_URI = "https://in03-28c44765e826237.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
CLUSTER_NAME = "vibecoding_072025"
COLLECTION_NAME = "rag_01_chunked"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Dense vector configuration
DENSE_VECTOR_DIM = 384  # all-MiniLM-L6-v2 embedding dimension
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_milvus_chunked_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Different types of chunking strategies."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    SEMANTIC_BLOCK = "semantic_block"
    FIXED_SIZE = "fixed_size"
    LOGICAL_SECTION = "logical_section"

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    repo_name: str
    file_path: str
    chunk_type: ChunkType
    chunk_id: str
    content: str
    start_line: int
    end_line: int
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    chunk_size: int = 0
    file_sha: str = ""
    url: str = ""

@dataclass
class SparseVector:
    """Represents a sparse vector with indices and values."""
    indices: List[int]
    values: List[float]
    
    def to_milvus_format(self):
        """Convert to Milvus sparse vector format (scipy csr_matrix)."""
        from scipy.sparse import csr_matrix
        import numpy as np
        
        if not self.indices:
            # Empty vector
            return csr_matrix((1, 1))
            
        # Create a csr_matrix with shape (1, max_index + 1)
        max_dim = max(self.indices) + 1
        
        # Create row, col, data for csr_matrix
        row = [0] * len(self.indices)  # All in first row
        col = self.indices
        data = self.values
        
        return csr_matrix((data, (row, col)), shape=(1, max_dim))

@dataclass
class DenseVector:
    """Represents a dense vector for semantic similarity."""
    embedding: List[float]
    
    def to_milvus_format(self) -> List[float]:
        """Convert to Milvus dense vector format."""
        return self.embedding
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return len(self.embedding)

class EmbeddingGenerator:
    """Generates both sparse and dense embeddings from text."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding generators."""
        logger.info(f"🤖 Loading embedding model: {model_name}")
        
        # Dense embeddings using Sentence Transformers
        self.dense_model = SentenceTransformer(model_name)
        self.dense_model.eval()  # Set to evaluation mode
        
        # Sparse embeddings using TF-IDF
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 2 to 1 for smaller datasets
            max_df=0.95  # Changed from 0.8 to 0.95 for more flexibility
        )
        self.sparse_fitted = False
        
        logger.info(f"✅ Embedding model loaded. Dense dim: {self.dense_model.get_sentence_embedding_dimension()}")
    
    def fit_sparse_vectorizer(self, texts: List[str]):
        """Fit the sparse vectorizer on the corpus."""
        if not self.sparse_fitted:
            logger.info("🔧 Fitting TF-IDF vectorizer...")
            processed_texts = [self._preprocess_text(text) for text in texts]
            self.sparse_vectorizer.fit(processed_texts)
            self.sparse_fitted = True
            logger.info(f"✅ TF-IDF fitted with {len(self.sparse_vectorizer.vocabulary_)} features")
    
    def generate_dense_embedding(self, text: str) -> DenseVector:
        """Generate dense semantic embedding."""
        try:
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.dense_model.encode(processed_text, normalize_embeddings=True)
            
            # Convert to Python list with proper type handling
            if hasattr(embedding, 'tolist'):
                # Handle numpy arrays
                embedding_list = embedding.tolist()
            elif hasattr(embedding, 'cpu'):
                # Handle PyTorch tensors
                embedding_list = embedding.cpu().tolist()
            else:
                # Already a list
                embedding_list = list(embedding)
            
            # Ensure all elements are floats
            embedding_list = [float(x) for x in embedding_list]
            
            return DenseVector(embedding=embedding_list)
            
        except Exception as e:
            logger.error(f"Error generating dense embedding: {e}")
            # Return zero vector as fallback
            return DenseVector(embedding=[0.0] * DENSE_VECTOR_DIM)
    
    def generate_sparse_embedding(self, text: str) -> SparseVector:
        """Generate sparse TF-IDF embedding."""
        try:
            if not self.sparse_fitted:
                raise ValueError("Sparse vectorizer not fitted. Call fit_sparse_vectorizer first.")
            
            processed_text = self._preprocess_text(text)
            
            # Generate TF-IDF vector
            tfidf_matrix = self.sparse_vectorizer.transform([processed_text])
            
            # Ensure it's a sparse matrix and convert to CSR format
            from scipy.sparse import issparse
            if issparse(tfidf_matrix):
                csr_matrix_result = tfidf_matrix.tocsr()
            else:
                # Convert dense to sparse if needed
                from scipy.sparse import csr_matrix
                csr_matrix_result = csr_matrix(tfidf_matrix)
            
            # Extract non-zero indices and values
            indices = csr_matrix_result.indices.tolist()
            values = csr_matrix_result.data.tolist()
            
            return SparseVector(indices=indices, values=values)
            
        except Exception as e:
            logger.error(f"Error generating sparse embedding: {e}")
            return SparseVector(indices=[], values=[])
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Truncate if too long (to avoid memory issues)
        max_length = 8000  # characters
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text

class CodeChunker:
    """Handles different code chunking strategies."""
    
    def __init__(self):
        self.chunk_strategies = {
            ChunkType.FILE: self.chunk_by_file,
            ChunkType.FUNCTION: self.chunk_by_function,
            ChunkType.CLASS: self.chunk_by_class,
            ChunkType.SEMANTIC_BLOCK: self.chunk_by_semantic_block,
            ChunkType.FIXED_SIZE: self.chunk_by_fixed_size,
            ChunkType.LOGICAL_SECTION: self.chunk_by_logical_section
        }
    
    def chunk_code(self, file_content: str, file_path: str, repo_name: str, 
                   chunk_type: ChunkType, file_sha: str = "", url: str = "") -> List[CodeChunk]:
        """Main method to chunk code based on strategy."""
        language = self._detect_language(file_path)
        chunker = self.chunk_strategies.get(chunk_type)
        
        if not chunker:
            logger.warning(f"Unknown chunk type: {chunk_type}")
            return []
            
        return chunker(file_content, file_path, repo_name, language, file_sha, url)
    
    def chunk_by_file(self, content: str, file_path: str, repo_name: str, 
                     language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Original strategy: entire file as one chunk."""
        lines = content.split('\n')
        chunk_id = hashlib.md5(f"{repo_name}:{file_path}:file".encode()).hexdigest()[:8]
        
        return [CodeChunk(
            repo_name=repo_name,
            file_path=file_path,
            chunk_type=ChunkType.FILE,
            chunk_id=chunk_id,
            content=content,
            start_line=1,
            end_line=len(lines),
            language=language,
            chunk_size=len(content),
            file_sha=file_sha,
            url=url
        )]
    
    def chunk_by_function(self, content: str, file_path: str, repo_name: str,
                         language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Chunk by functions/methods."""
        chunks = []
        lines = content.split('\n')
        
        if language == 'python':
            chunks.extend(self._extract_python_functions(content, file_path, repo_name, file_sha, url))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._extract_js_functions(content, file_path, repo_name, file_sha, url))
        elif language == 'java':
            chunks.extend(self._extract_java_methods(content, file_path, repo_name, file_sha, url))
        else:
            # Generic function extraction using regex
            chunks.extend(self._extract_generic_functions(content, file_path, repo_name, language, file_sha, url))
        
        # If no functions found, fall back to file chunking
        if not chunks:
            return self.chunk_by_file(content, file_path, repo_name, language, file_sha, url)
            
        return chunks
    
    def chunk_by_class(self, content: str, file_path: str, repo_name: str,
                      language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Chunk by classes."""
        chunks = []
        
        if language == 'python':
            chunks.extend(self._extract_python_classes(content, file_path, repo_name, file_sha, url))
        elif language == 'java':
            chunks.extend(self._extract_java_classes(content, file_path, repo_name, file_sha, url))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._extract_js_classes(content, file_path, repo_name, file_sha, url))
        
        # If no classes found, fall back to function chunking
        if not chunks:
            return self.chunk_by_function(content, file_path, repo_name, language, file_sha, url)
            
        return chunks
    
    def chunk_by_semantic_block(self, content: str, file_path: str, repo_name: str,
                               language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Chunk by semantic code blocks (imports, functions, classes, etc.)."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_type = "misc"
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Detect semantic boundaries
            if self._is_import_line(stripped_line, language):
                if current_chunk and current_type != "imports":
                    chunks.append(self._create_semantic_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "imports"
            elif self._is_function_start(stripped_line, language):
                if current_chunk:
                    chunks.append(self._create_semantic_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "function"
            elif self._is_class_start(stripped_line, language):
                if current_chunk:
                    chunks.append(self._create_semantic_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "class"
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_semantic_chunk(
                current_chunk, start_line, len(lines), current_type,
                file_path, repo_name, language, file_sha, url
            ))
        
        return chunks if chunks else self.chunk_by_file(content, file_path, repo_name, language, file_sha, url)
    
    def chunk_by_fixed_size(self, content: str, file_path: str, repo_name: str,
                           language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Chunk by fixed character/line size."""
        chunks = []
        chunk_size = 1000  # characters
        overlap = 100      # character overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if not chunk_content.strip():
                continue
                
            start_line = content[:i].count('\n') + 1
            end_line = content[:i + len(chunk_content)].count('\n') + 1
            
            chunk_id = hashlib.md5(f"{repo_name}:{file_path}:fixed:{i}".encode()).hexdigest()[:8]
            
            chunks.append(CodeChunk(
                repo_name=repo_name,
                file_path=file_path,
                chunk_type=ChunkType.FIXED_SIZE,
                chunk_id=chunk_id,
                content=chunk_content,
                start_line=start_line,
                end_line=end_line,
                language=language,
                chunk_size=len(chunk_content),
                file_sha=file_sha,
                url=url
            ))
        
        return chunks
    
    def chunk_by_logical_section(self, content: str, file_path: str, repo_name: str,
                                language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Chunk by logical sections (comments, docstrings, code blocks)."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_type = "code"
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Detect section boundaries
            if self._is_comment_block_start(stripped_line, language):
                if current_chunk and current_type != "comments":
                    chunks.append(self._create_logical_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "comments"
            elif self._is_docstring_start(stripped_line, language):
                if current_chunk and current_type != "documentation":
                    chunks.append(self._create_logical_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "documentation"
            elif stripped_line and not stripped_line.startswith(('*', '//', '#', '"""', "'''")):
                if current_chunk and current_type != "code":
                    chunks.append(self._create_logical_chunk(
                        current_chunk, start_line, i-1, current_type,
                        file_path, repo_name, language, file_sha, url
                    ))
                    current_chunk = []
                    start_line = i
                current_type = "code"
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_logical_chunk(
                current_chunk, start_line, len(lines), current_type,
                file_path, repo_name, language, file_sha, url
            ))
        
        return chunks if chunks else self.chunk_by_file(content, file_path, repo_name, language, file_sha, url)
    
    def _extract_python_functions(self, content: str, file_path: str, repo_name: str, 
                                 file_sha: str, url: str) -> List[CodeChunk]:
        """Extract Python functions using AST parsing."""
        chunks = []
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    function_content = '\n'.join(lines[start_line-1:end_line])
                    chunk_id = hashlib.md5(f"{repo_name}:{file_path}:func:{node.name}".encode()).hexdigest()[:8]
                    
                    chunks.append(CodeChunk(
                        repo_name=repo_name,
                        file_path=file_path,
                        chunk_type=ChunkType.FUNCTION,
                        chunk_id=chunk_id,
                        content=function_content,
                        start_line=start_line,
                        end_line=end_line,
                        language='python',
                        function_name=node.name,
                        chunk_size=len(function_content),
                        file_sha=file_sha,
                        url=url
                    ))
        except SyntaxError:
            logger.warning(f"Could not parse Python file: {file_path}")
        
        return chunks
    
    def _extract_python_classes(self, content: str, file_path: str, repo_name: str,
                               file_sha: str, url: str) -> List[CodeChunk]:
        """Extract Python classes using AST parsing."""
        chunks = []
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    class_content = '\n'.join(lines[start_line-1:end_line])
                    chunk_id = hashlib.md5(f"{repo_name}:{file_path}:class:{node.name}".encode()).hexdigest()[:8]
                    
                    chunks.append(CodeChunk(
                        repo_name=repo_name,
                        file_path=file_path,
                        chunk_type=ChunkType.CLASS,
                        chunk_id=chunk_id,
                        content=class_content,
                        start_line=start_line,
                        end_line=end_line,
                        language='python',
                        class_name=node.name,
                        chunk_size=len(class_content),
                        file_sha=file_sha,
                        url=url
                    ))
        except SyntaxError:
            logger.warning(f"Could not parse Python file: {file_path}")
        
        return chunks
    
    def _extract_js_functions(self, content: str, file_path: str, repo_name: str,
                             file_sha: str, url: str) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript functions using regex."""
        chunks = []
        lines = content.split('\n')
        
        # Patterns for different function types
        patterns = [
            r'^\s*function\s+(\w+)\s*\(',      # function name()
            r'^\s*const\s+(\w+)\s*=\s*\(',     # const name = (
            r'^\s*let\s+(\w+)\s*=\s*\(',       # let name = (
            r'^\s*var\s+(\w+)\s*=\s*\(',       # var name = (
            r'^\s*(\w+):\s*function\s*\(',     # name: function(
            r'^\s*async\s+function\s+(\w+)',   # async function name
            r'^\s*(\w+)\s*\(.*\)\s*=>',        # arrow function name() =>
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    start_line = i + 1
                    end_line = self._find_js_function_end(lines, i)
                    
                    function_content = '\n'.join(lines[i:end_line])
                    chunk_id = hashlib.md5(f"{repo_name}:{file_path}:func:{func_name}".encode()).hexdigest()[:8]
                    
                    chunks.append(CodeChunk(
                        repo_name=repo_name,
                        file_path=file_path,
                        chunk_type=ChunkType.FUNCTION,
                        chunk_id=chunk_id,
                        content=function_content,
                        start_line=start_line,
                        end_line=end_line,
                        language=self._detect_language(file_path),
                        function_name=func_name,
                        chunk_size=len(function_content),
                        file_sha=file_sha,
                        url=url
                    ))
                    break
        
        return chunks
    
    def _extract_js_classes(self, content: str, file_path: str, repo_name: str,
                           file_sha: str, url: str) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript classes using regex."""
        chunks = []
        lines = content.split('\n')
        
        class_pattern = r'^\s*class\s+(\w+)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                class_name = match.group(1)
                start_line = i + 1
                end_line = self._find_js_class_end(lines, i)
                
                class_content = '\n'.join(lines[i:end_line])
                chunk_id = hashlib.md5(f"{repo_name}:{file_path}:class:{class_name}".encode()).hexdigest()[:8]
                
                chunks.append(CodeChunk(
                    repo_name=repo_name,
                    file_path=file_path,
                    chunk_type=ChunkType.CLASS,
                    chunk_id=chunk_id,
                    content=class_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self._detect_language(file_path),
                    class_name=class_name,
                    chunk_size=len(class_content),
                    file_sha=file_sha,
                    url=url
                ))
        
        return chunks
    
    def _extract_java_methods(self, content: str, file_path: str, repo_name: str,
                             file_sha: str, url: str) -> List[CodeChunk]:
        """Extract Java methods using regex."""
        chunks = []
        lines = content.split('\n')
        
        # Java method pattern
        method_pattern = r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\('
        
        for i, line in enumerate(lines):
            match = re.search(method_pattern, line)
            if match:
                method_name = match.group(3)
                start_line = i + 1
                end_line = self._find_java_method_end(lines, i)
                
                method_content = '\n'.join(lines[i:end_line])
                chunk_id = hashlib.md5(f"{repo_name}:{file_path}:method:{method_name}".encode()).hexdigest()[:8]
                
                chunks.append(CodeChunk(
                    repo_name=repo_name,
                    file_path=file_path,
                    chunk_type=ChunkType.FUNCTION,
                    chunk_id=chunk_id,
                    content=method_content,
                    start_line=start_line,
                    end_line=end_line,
                    language='java',
                    function_name=method_name,
                    chunk_size=len(method_content),
                    file_sha=file_sha,
                    url=url
                ))
        
        return chunks
    
    def _extract_java_classes(self, content: str, file_path: str, repo_name: str,
                             file_sha: str, url: str) -> List[CodeChunk]:
        """Extract Java classes using regex."""
        chunks = []
        lines = content.split('\n')
        
        class_pattern = r'^\s*(public|private)?\s*class\s+(\w+)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                class_name = match.group(2)
                start_line = i + 1
                end_line = self._find_java_class_end(lines, i)
                
                class_content = '\n'.join(lines[i:end_line])
                chunk_id = hashlib.md5(f"{repo_name}:{file_path}:class:{class_name}".encode()).hexdigest()[:8]
                
                chunks.append(CodeChunk(
                    repo_name=repo_name,
                    file_path=file_path,
                    chunk_type=ChunkType.CLASS,
                    chunk_id=chunk_id,
                    content=class_content,
                    start_line=start_line,
                    end_line=end_line,
                    language='java',
                    class_name=class_name,
                    chunk_size=len(class_content),
                    file_sha=file_sha,
                    url=url
                ))
        
        return chunks
    
    def _extract_generic_functions(self, content: str, file_path: str, repo_name: str,
                                  language: str, file_sha: str, url: str) -> List[CodeChunk]:
        """Generic function extraction for other languages."""
        chunks = []
        lines = content.split('\n')
        
        # Generic patterns
        patterns = [
            r'^\s*def\s+(\w+)',           # Python
            r'^\s*func\s+(\w+)',          # Go
            r'^\s*fn\s+(\w+)',            # Rust
            r'^\s*sub\s+(\w+)',           # Perl
            r'^\s*function\s+(\w+)',      # Generic
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    start_line = i + 1
                    end_line = min(i + 50, len(lines))  # Limit to 50 lines
                    
                    function_content = '\n'.join(lines[i:end_line])
                    chunk_id = hashlib.md5(f"{repo_name}:{file_path}:func:{func_name}".encode()).hexdigest()[:8]
                    
                    chunks.append(CodeChunk(
                        repo_name=repo_name,
                        file_path=file_path,
                        chunk_type=ChunkType.FUNCTION,
                        chunk_id=chunk_id,
                        content=function_content,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        function_name=func_name,
                        chunk_size=len(function_content),
                        file_sha=file_sha,
                        url=url
                    ))
                    break
        
        return chunks
    
    def _find_js_function_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of JavaScript function by tracking braces."""
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > start_idx:
                return i + 1
        return len(lines)
    
    def _find_js_class_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of JavaScript class by tracking braces."""
        return self._find_js_function_end(lines, start_idx)
    
    def _find_java_method_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of Java method by tracking braces."""
        return self._find_js_function_end(lines, start_idx)
    
    def _find_java_class_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of Java class by tracking braces."""
        return self._find_js_function_end(lines, start_idx)
    
    def _create_semantic_chunk(self, lines: List[str], start_line: int, end_line: int,
                              chunk_type: str, file_path: str, repo_name: str, 
                              language: str, file_sha: str, url: str) -> CodeChunk:
        """Create a semantic chunk."""
        content = '\n'.join(lines)
        chunk_id = hashlib.md5(f"{repo_name}:{file_path}:semantic:{chunk_type}:{start_line}".encode()).hexdigest()[:8]
        
        return CodeChunk(
            repo_name=repo_name,
            file_path=file_path,
            chunk_type=ChunkType.SEMANTIC_BLOCK,
            chunk_id=chunk_id,
            content=content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_size=len(content),
            file_sha=file_sha,
            url=url
        )
    
    def _create_logical_chunk(self, lines: List[str], start_line: int, end_line: int,
                             chunk_type: str, file_path: str, repo_name: str,
                             language: str, file_sha: str, url: str) -> CodeChunk:
        """Create a logical chunk."""
        content = '\n'.join(lines)
        chunk_id = hashlib.md5(f"{repo_name}:{file_path}:logical:{chunk_type}:{start_line}".encode()).hexdigest()[:8]
        
        return CodeChunk(
            repo_name=repo_name,
            file_path=file_path,
            chunk_type=ChunkType.LOGICAL_SECTION,
            chunk_id=chunk_id,
            content=content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_size=len(content),
            file_sha=file_sha,
            url=url
        )
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
            '.php': 'php', '.rb': 'ruby', '.go': 'go', '.rs': 'rust',
            '.kt': 'kotlin', '.swift': 'swift', '.scala': 'scala',
            '.md': 'markdown', '.html': 'html', '.css': 'css',
            '.sql': 'sql', '.yml': 'yaml', '.yaml': 'yaml',
            '.json': 'json', '.xml': 'xml', '.sh': 'shell'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_to_lang.get(ext, 'text')
    
    def _is_import_line(self, line: str, language: str) -> bool:
        """Check if line is an import statement."""
        if language == 'python':
            return line.startswith(('import ', 'from '))
        elif language in ['javascript', 'typescript']:
            return line.startswith(('import ', 'require(', 'const ')) and 'require(' in line
        elif language == 'java':
            return line.startswith('import ')
        return False
    
    def _is_function_start(self, line: str, language: str) -> bool:
        """Check if line starts a function."""
        if language == 'python':
            return line.startswith('def ') or line.startswith('async def ')
        elif language in ['javascript', 'typescript']:
            return ('function ' in line or '=>' in line or 
                   line.startswith('const ') and '=' in line and '(' in line)
        elif language == 'java':
            return ('(' in line and ')' in line and 
                   any(modifier in line for modifier in ['public', 'private', 'protected']))
        return False
    
    def _is_class_start(self, line: str, language: str) -> bool:
        """Check if line starts a class."""
        if language == 'python':
            return line.startswith('class ')
        elif language in ['javascript', 'typescript']:
            return line.startswith('class ')
        elif language == 'java':
            return 'class ' in line and any(modifier in line for modifier in ['public', 'private'])
        return False
    
    def _is_comment_block_start(self, line: str, language: str) -> bool:
        """Check if line starts a comment block."""
        if language == 'python':
            return line.startswith('#') or line.startswith('"""') or line.startswith("'''")
        elif language in ['javascript', 'typescript', 'java', 'c', 'cpp']:
            return line.startswith('//') or line.startswith('/*')
        return line.startswith('#')
    
    def _is_docstring_start(self, line: str, language: str) -> bool:
        """Check if line starts a docstring."""
        if language == 'python':
            return line.startswith('"""') or line.startswith("'''")
        elif language in ['javascript', 'typescript']:
            return line.startswith('/**')
        return False

# Continue with GitHubDataFetcher and other classes from original script...
class GitHubDataFetcher:
    """Handles fetching data from GitHub repositories."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"token {token}"})
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Milvus-GitHub-Chunked-Loader/1.0"
        })
        
    def get_repo_files(self, repo_name: str, max_files: int = 50) -> List[Dict]:
        """Fetch files from a GitHub repository."""
        logger.info(f"Fetching files from repository: {repo_name}")
        files = []
        
        try:
            # First, get the default branch name
            repo_url = f"https://api.github.com/repos/{repo_name}"
            repo_response = self.session.get(repo_url)
            repo_response.raise_for_status()
            
            default_branch = repo_response.json().get("default_branch", "main")
            logger.info(f"Using default branch: {default_branch}")
            
            # Get repository tree using the correct branch
            tree_url = f"https://api.github.com/repos/{repo_name}/git/trees/{default_branch}?recursive=1"
            response = self.session.get(tree_url)
            response.raise_for_status()
            
            tree_data = response.json()
            file_count = 0
            
            for item in tree_data.get("tree", []):
                if file_count >= max_files:
                    break
                    
                if item["type"] == "blob" and self._is_text_file(item["path"]):
                    file_data = self._fetch_file_content(repo_name, item)
                    if file_data:
                        files.append(file_data)
                        file_count += 1
                        
                    # Rate limiting
                    time.sleep(0.1)
                    
        except requests.RequestException as e:
            logger.error(f"Error fetching repository {repo_name}: {e}")
            
        logger.info(f"Successfully fetched {len(files)} files from {repo_name}")
        return files
    
    def _is_text_file(self, file_path: str) -> bool:
        """Check if file is likely to contain text content."""
        text_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php',
            '.rb', '.go', '.rs', '.kt', '.swift', '.scala', '.clj', '.r',
            '.md', '.txt', '.rst', '.tex', '.yml', '.yaml', '.json', '.xml',
            '.html', '.css', '.scss', '.sass', '.sql', '.sh', '.bat', '.ps1'
        }
        
        ext = Path(file_path).suffix.lower()
        if ext in text_extensions:
            return True
            
        filename = Path(file_path).name.lower()
        config_files = {
            'readme', 'license', 'dockerfile', 'makefile', 'gemfile',
            'requirements.txt', 'package.json', 'composer.json'
        }
        
        return any(config in filename for config in config_files)
    
    def _fetch_file_content(self, repo_name: str, item: Dict) -> Optional[Dict]:
        """Fetch content of a specific file."""
        try:
            content_url = f"https://api.github.com/repos/{repo_name}/contents/{item['path']}"
            response = self.session.get(content_url)
            response.raise_for_status()
            
            content_data = response.json()
            
            if content_data.get("encoding") == "base64":
                content = base64.b64decode(content_data["content"]).decode('utf-8', errors='ignore')
                
                if len(content) > 50000:  # 50KB limit
                    return None
                    
                return {
                    "repo_name": repo_name,
                    "file_path": item["path"],
                    "content": content,
                    "size": item["size"],
                    "sha": item["sha"],
                    "url": content_data["html_url"]
                }
                
        except Exception as e:
            logger.warning(f"Error fetching file {item['path']}: {e}")
            
        return None

class MilvusManager:
    """Manages Milvus connections and collections for both sparse and dense vectors."""
    
    def __init__(self, uri: str, token: str):
        """Initialize Milvus connection."""
        self.uri = uri
        self.token = token
        self.connected = False
        
    def connect(self):
        """Connect to Milvus."""
        try:
            logger.info(f"🔌 Connecting to Milvus: {self.uri}")
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            self.connected = True
            logger.info("✅ Successfully connected to Milvus")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def create_sparse_collection(self, collection_name: str) -> Collection:
        """Create collection for sparse vectors."""
        if collection_name in utility.list_collections():
            logger.info(f"Collection {collection_name} already exists, checking schema...")
            
            # Check if schema is compatible
            collection = Collection(collection_name)
            schema = collection.schema
            
            # Check for required fields
            required_fields = ["id", "repo_name", "file_path", "chunk_id", "content", 
                             "chunk_type", "language", "sparse_vector"]
            existing_fields = [field.name for field in schema.fields]
            
            if not all(field in existing_fields for field in required_fields):
                logger.warning(f"Schema incompatible. Dropping collection {collection_name}")
                utility.drop_collection(collection_name)
            else:
                logger.info(f"✅ Using existing collection: {collection_name}")
                return collection
        
        # Create new collection schema
        logger.info(f"📋 Creating sparse collection schema: {collection_name}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=32000),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="start_line", dtype=DataType.INT64),
            FieldSchema(name="end_line", dtype=DataType.INT64),
            FieldSchema(name="chunk_size", dtype=DataType.INT64),
            FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Sparse vector collection for {collection_name}"
        )
        
        collection = Collection(collection_name, schema)
        logger.info(f"✅ Created sparse collection: {collection_name}")
        
        # Create index for sparse vectors
        index_params = {
            "metric_type": "IP",  # Inner Product for sparse vectors
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        }
        
        collection.create_index("sparse_vector", index_params)
        logger.info("📊 Created sparse vector index")
        
        return collection
    
    def create_dense_collection(self, collection_name: str) -> Collection:
        """Create collection for dense vectors."""
        if collection_name in utility.list_collections():
            logger.info(f"Collection {collection_name} already exists, checking schema...")
            
            collection = Collection(collection_name)
            schema = collection.schema
            
            # Check for required fields
            required_fields = ["id", "repo_name", "file_path", "chunk_id", "content", 
                             "chunk_type", "language", "dense_vector"]
            existing_fields = [field.name for field in schema.fields]
            
            if not all(field in existing_fields for field in required_fields):
                logger.warning(f"Schema incompatible. Dropping collection {collection_name}")
                utility.drop_collection(collection_name)
            else:
                logger.info(f"✅ Using existing collection: {collection_name}")
                return collection
        
        # Create new collection schema
        logger.info(f"📋 Creating dense collection schema: {collection_name}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=32000),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="start_line", dtype=DataType.INT64),
            FieldSchema(name="end_line", dtype=DataType.INT64),
            FieldSchema(name="chunk_size", dtype=DataType.INT64),
            FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIM)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Dense vector collection for {collection_name}"
        )
        
        collection = Collection(collection_name, schema)
        logger.info(f"✅ Created dense collection: {collection_name}")
        
        # Create index for dense vectors
        index_params = {
            "metric_type": "COSINE",  # Cosine similarity for dense vectors
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        collection.create_index("dense_vector", index_params)
        logger.info("📊 Created dense vector index")
        
        return collection
    
    def insert_sparse_data(self, collection: Collection, chunks: List[CodeChunk], 
                          sparse_vectors: List[SparseVector]):
        """Insert sparse vector data into collection."""
        logger.info(f"📥 Inserting {len(chunks)} sparse vectors...")
        
        # Prepare data in list format for Milvus
        ids = []
        repo_names = []
        file_paths = []
        chunk_ids = []
        contents = []
        chunk_types = []
        languages = []
        start_lines = []
        end_lines = []
        chunk_sizes = []
        function_names = []
        class_names = []
        sparse_vector_data = []
        
        for chunk, sparse_vec in zip(chunks, sparse_vectors):
            ids.append(f"{chunk.repo_name}:{chunk.chunk_id}:sparse")
            repo_names.append(chunk.repo_name)
            file_paths.append(chunk.file_path)
            chunk_ids.append(chunk.chunk_id)
            contents.append(chunk.content[:32000])  # Truncate if necessary
            chunk_types.append(chunk.chunk_type.value)
            languages.append(chunk.language)
            start_lines.append(chunk.start_line)
            end_lines.append(chunk.end_line)
            chunk_sizes.append(chunk.chunk_size)
            function_names.append(chunk.function_name or "")
            class_names.append(chunk.class_name or "")
            sparse_vector_data.append(sparse_vec.to_milvus_format())
        
        # Insert data
        data = [
            ids, repo_names, file_paths, chunk_ids, contents, chunk_types, 
            languages, start_lines, end_lines, chunk_sizes, function_names, 
            class_names, sparse_vector_data
        ]
        
        mr = collection.insert(data)
        collection.flush()
        
        logger.info(f"✅ Inserted {len(chunks)} sparse vectors. Insert IDs: {len(mr.primary_keys)}")
        return mr
    
    def insert_dense_data(self, collection: Collection, chunks: List[CodeChunk], 
                         dense_vectors: List[DenseVector]):
        """Insert dense vector data into collection."""
        logger.info(f"📥 Inserting {len(chunks)} dense vectors...")
        
        # Prepare data in list format for Milvus
        ids = []
        repo_names = []
        file_paths = []
        chunk_ids = []
        contents = []
        chunk_types = []
        languages = []
        start_lines = []
        end_lines = []
        chunk_sizes = []
        function_names = []
        class_names = []
        dense_vector_data = []
        
        for chunk, dense_vec in zip(chunks, dense_vectors):
            ids.append(f"{chunk.repo_name}:{chunk.chunk_id}:dense")
            repo_names.append(chunk.repo_name)
            file_paths.append(chunk.file_path)
            chunk_ids.append(chunk.chunk_id)
            contents.append(chunk.content[:32000])  # Truncate if necessary
            chunk_types.append(chunk.chunk_type.value)
            languages.append(chunk.language)
            start_lines.append(chunk.start_line)
            end_lines.append(chunk.end_line)
            chunk_sizes.append(chunk.chunk_size)
            function_names.append(chunk.function_name or "")
            class_names.append(chunk.class_name or "")
            dense_vector_data.append(dense_vec.to_milvus_format())
        
        # Insert data
        data = [
            ids, repo_names, file_paths, chunk_ids, contents, chunk_types,
            languages, start_lines, end_lines, chunk_sizes, function_names,
            class_names, dense_vector_data
        ]
        
        mr = collection.insert(data)
        collection.flush()
        
        logger.info(f"✅ Inserted {len(chunks)} dense vectors. Insert IDs: {len(mr.primary_keys)}")
        return mr
    
    def load_collection(self, collection: Collection):
        """Load collection into memory."""
        collection.load()
        logger.info(f"🔄 Loaded collection: {collection.name}")
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get collection statistics."""
        if collection_name not in utility.list_collections():
            return {"exists": False}
            
        collection = Collection(collection_name)
        stats = collection.get_compaction_state()
        
        return {
            "exists": True,
            "entities": collection.num_entities,
            "loaded": collection.has_index(),
            "compaction_state": stats
        }

def main():
    """Main function to load GitHub data into Milvus with both sparse and dense vectors."""
    
    if not MILVUS_TOKEN:
        logger.error("❌ MILVUS_TOKEN environment variable is required")
        return
    
    # Configuration
    chunk_strategy = ChunkType.FUNCTION  # Can be changed to other strategies
    sparse_collection_name = f"rag_chunks_{chunk_strategy.value}_sparse"
    dense_collection_name = f"rag_chunks_{chunk_strategy.value}_dense"
    
    # Test repositories
    test_repos = [
        "fastapi/fastapi",
        "microsoft/vscode",
        "pytorch/pytorch"
    ]
    
    logger.info("🚀 Starting GitHub to Milvus Chunked Vector Loader")
    logger.info(f"📊 Strategy: {chunk_strategy.value}")
    logger.info(f"🎯 Collections: {sparse_collection_name}, {dense_collection_name}")
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        fetcher = GitHubDataFetcher(GITHUB_TOKEN)
        chunker = CodeChunker()
        embedding_gen = EmbeddingGenerator()
        milvus_manager = MilvusManager(MILVUS_URI, MILVUS_TOKEN)
        
        # Connect to Milvus
        milvus_manager.connect()
        
        # Create collections
        sparse_collection = milvus_manager.create_sparse_collection(sparse_collection_name)
        dense_collection = milvus_manager.create_dense_collection(dense_collection_name)
        
        # Process repositories
        all_chunks = []
        all_texts = []
        
        for repo in test_repos:
            logger.info(f"\n📂 Processing repository: {repo}")
            
            # Fetch files
            files = fetcher.get_repo_files(repo, max_files=20)
            logger.info(f"📄 Fetched {len(files)} files from {repo}")
            
            # Generate chunks
            repo_chunks = []
            for file_data in files:
                chunks = chunker.chunk_code(
                    file_content=file_data["content"],
                    file_path=file_data["file_path"],
                    repo_name=file_data["repo_name"],
                    chunk_type=chunk_strategy,
                    file_sha=file_data["sha"],
                    url=file_data["url"]
                )
                repo_chunks.extend(chunks)
            
            logger.info(f"✂️ Generated {len(repo_chunks)} chunks from {repo}")
            all_chunks.extend(repo_chunks)
            
            # Collect texts for TF-IDF fitting
            repo_texts = [chunk.content for chunk in repo_chunks]
            all_texts.extend(repo_texts)
        
        logger.info(f"\n📊 Total chunks generated: {len(all_chunks)}")
        logger.info(f"📊 Total texts for vectorization: {len(all_texts)}")
        
        # Fit sparse vectorizer on all texts
        logger.info("\n🔧 Fitting sparse vectorizer...")
        embedding_gen.fit_sparse_vectorizer(all_texts)
        
        # Generate embeddings in batches
        batch_size = 50  # Process in smaller batches
        sparse_vectors = []
        dense_vectors = []
        
        logger.info(f"\n🧮 Generating embeddings in batches of {batch_size}...")
        
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_end = min(i + batch_size, len(all_chunks))
            
            logger.info(f"Processing batch {i//batch_size + 1}: chunks {i+1}-{batch_end}")
            
            # Generate sparse vectors
            batch_sparse = []
            for chunk in batch_chunks:
                sparse_vec = embedding_gen.generate_sparse_embedding(chunk.content)
                batch_sparse.append(sparse_vec)
            sparse_vectors.extend(batch_sparse)
            
            # Generate dense vectors
            batch_dense = []
            for chunk in batch_chunks:
                dense_vec = embedding_gen.generate_dense_embedding(chunk.content)
                batch_dense.append(dense_vec)
            dense_vectors.extend(batch_dense)
            
            logger.info(f"✅ Batch {i//batch_size + 1} completed")
        
        logger.info(f"\n📊 Generated {len(sparse_vectors)} sparse vectors")
        logger.info(f"📊 Generated {len(dense_vectors)} dense vectors")
        
        # Insert data into Milvus
        logger.info("\n💾 Inserting data into Milvus...")
        
        # Insert sparse vectors
        sparse_result = milvus_manager.insert_sparse_data(
            sparse_collection, all_chunks, sparse_vectors
        )
        
        # Insert dense vectors  
        dense_result = milvus_manager.insert_dense_data(
            dense_collection, all_chunks, dense_vectors
        )
        
        # Load collections
        milvus_manager.load_collection(sparse_collection)
        milvus_manager.load_collection(dense_collection)
        
        # Get statistics
        sparse_stats = milvus_manager.get_collection_stats(sparse_collection_name)
        dense_stats = milvus_manager.get_collection_stats(dense_collection_name)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 CHUNKED VECTOR LOADING COMPLETED!")
        logger.info("="*60)
        logger.info(f"📊 Strategy: {chunk_strategy.value}")
        logger.info(f"📂 Repositories processed: {len(test_repos)}")
        logger.info(f"📄 Total files processed: {sum(len(fetcher.get_repo_files(repo, max_files=20)) for repo in test_repos)}")
        logger.info(f"✂️ Total chunks generated: {len(all_chunks)}")
        logger.info(f"🔤 Sparse vectors inserted: {len(sparse_vectors)}")
        logger.info(f"🧠 Dense vectors inserted: {len(dense_vectors)}")
        logger.info(f"📊 Sparse collection entities: {sparse_stats.get('entities', 0)}")
        logger.info(f"📊 Dense collection entities: {dense_stats.get('entities', 0)}")
        logger.info("="*60)
        
        # Example search queries for testing
        logger.info("\n🔍 Example search queries to test:")
        example_queries = [
            "FastAPI authentication middleware",
            "async function database connection",
            "error handling exception patterns",
            "dependency injection container",
            "HTTP request validation"
        ]
        
        for i, query in enumerate(example_queries, 1):
            logger.info(f"  {i}. \"{query}\"")
        
        logger.info(f"\n🎯 Collections created:")
        logger.info(f"  • Sparse: {sparse_collection_name}")
        logger.info(f"  • Dense: {dense_collection_name}")
        
        logger.info("\n✅ Ready for hybrid search and reranking!")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 