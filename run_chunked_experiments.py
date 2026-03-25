#!/usr/bin/env python3
"""
Production Chunked Loader - Multiple Strategies
==============================================

Loads GitHub data using different chunking strategies into separate collections
for comparison and experimentation.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Dict

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from load_data_chunked import (
        ChunkType, CodeChunker, GitHubDataFetcher, SparseVector,
        logger
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    from scipy.sparse import csr_matrix
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)

# Configuration
MILVUS_URI = "https://in03-28c44765e826237.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

class ChunkedVectorEncoder:
    """Enhanced vector encoder for chunked content."""
    
    def __init__(self, max_features: int = 5000):
        self.vectorizers = {}
        self.max_features = max_features
    
    def fit_transform_chunks(self, chunks: List, chunk_type: ChunkType) -> List[SparseVector]:
        """Fit vectorizer and transform chunks to sparse vectors."""
        
        # Create or get vectorizer for this chunk type
        vectorizer_key = chunk_type.value
        if vectorizer_key not in self.vectorizers:
            self.vectorizers[vectorizer_key] = TfidfVectorizer(
                max_features=self.max_features,
                min_df=1,  # Lower threshold for smaller chunks
                max_df=0.9,
                stop_words='english',
                ngram_range=(1, 2),
                token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            )
        
        vectorizer = self.vectorizers[vectorizer_key]
        
        # Extract content from chunks
        documents = [self._preprocess_chunk(chunk) for chunk in chunks]
        
        logger.info(f"Vectorizing {len(documents)} {chunk_type.value} chunks...")
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(documents)
        logger.info(f"Vocabulary size for {chunk_type.value}: {len(vectorizer.vocabulary_)}")
        
        return self._matrix_to_sparse_vectors(tfidf_matrix)
    
    def _preprocess_chunk(self, chunk) -> str:
        """Preprocess chunk content for vectorization."""
        content = chunk.content
        
        # Add metadata as context
        metadata_parts = []
        if chunk.function_name:
            metadata_parts.append(f"function_{chunk.function_name}")
        if chunk.class_name:
            metadata_parts.append(f"class_{chunk.class_name}")
        
        # Add language context
        metadata_parts.append(f"lang_{chunk.language}")
        
        # Combine content with metadata
        enhanced_content = content
        if metadata_parts:
            enhanced_content = " ".join(metadata_parts) + " " + content
        
        return enhanced_content[:5000]  # Limit length
    
    def _matrix_to_sparse_vectors(self, matrix: csr_matrix) -> List[SparseVector]:
        """Convert scipy sparse matrix to list of SparseVector objects."""
        sparse_vectors = []
        
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            indices = row.indices.tolist()
            values = row.data.tolist()
            
            # Filter low values
            threshold = 0.01
            filtered_indices = []
            filtered_values = []
            
            for idx, val in zip(indices, values):
                if val > threshold:
                    filtered_indices.append(idx)
                    filtered_values.append(float(val))
            
            sparse_vectors.append(SparseVector(filtered_indices, filtered_values))
        
        return sparse_vectors

class ChunkedMilvusManager:
    """Manages multiple Milvus collections for different chunk types."""
    
    def __init__(self, base_collection_name: str):
        self.base_collection_name = base_collection_name
        self.collections = {}
    
    def connect(self, uri: str, token: str) -> bool:
        """Connect to Milvus."""
        try:
            connections.connect(uri=uri, token=token)
            logger.info("Connected to Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def create_collection_for_chunk_type(self, chunk_type: ChunkType) -> bool:
        """Create a collection specifically for a chunk type."""
        collection_name = f"{self.base_collection_name}_{chunk_type.value}"
        
        try:
            # Drop existing collection if it exists
            if utility.has_collection(collection_name):
                logger.info(f"Dropping existing collection: {collection_name}")
                utility.drop_collection(collection_name)
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="start_line", dtype=DataType.INT64),
                FieldSchema(name="end_line", dtype=DataType.INT64),
                FieldSchema(name="chunk_size", dtype=DataType.INT64),
                FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"GitHub code chunks - {chunk_type.value} strategy"
            )
            
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "IP",
                "index_type": "SPARSE_INVERTED_INDEX"
            }
            
            collection.create_index(field_name="sparse_vector", index_params=index_params)
            
            self.collections[chunk_type] = collection
            logger.info(f"Created collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection for {chunk_type.value}: {e}")
            return False
    
    def insert_chunks(self, chunks: List, vectors: List[SparseVector], chunk_type: ChunkType) -> bool:
        """Insert chunks into the appropriate collection."""
        collection = self.collections.get(chunk_type)
        if not collection:
            logger.error(f"No collection found for chunk type: {chunk_type}")
            return False
        
        try:
            # Prepare data
            insert_data = [
                [chunk.chunk_id for chunk in chunks],                    # id
                [chunk.repo_name for chunk in chunks],                   # repo_name
                [chunk.file_path for chunk in chunks],                   # file_path
                [chunk.chunk_type.value for chunk in chunks],            # chunk_type
                [chunk.language for chunk in chunks],                    # language
                [chunk.content[:65535] for chunk in chunks],             # content
                [vector.to_milvus_format() for vector in vectors],       # sparse_vector
                [chunk.start_line for chunk in chunks],                  # start_line
                [chunk.end_line for chunk in chunks],                    # end_line
                [chunk.chunk_size for chunk in chunks],                  # chunk_size
                [chunk.function_name or "" for chunk in chunks],         # function_name
                [chunk.class_name or "" for chunk in chunks],            # class_name
                [chunk.url for chunk in chunks]                          # url
            ]
            
            # Insert data
            result = collection.insert(insert_data)
            collection.flush()
            
            logger.info(f"Inserted {len(chunks)} chunks into {chunk_type.value} collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert chunks for {chunk_type.value}: {e}")
            return False
    
    def load_collections(self):
        """Load all collections into memory."""
        for chunk_type, collection in self.collections.items():
            try:
                collection.load()
                count = collection.num_entities
                logger.info(f"Loaded {chunk_type.value} collection: {count} entities")
            except Exception as e:
                logger.error(f"Failed to load {chunk_type.value} collection: {e}")

def run_chunked_experiment(repos: List[str], max_files_per_repo: int = 10):
    """Run the full chunked experiment."""
    
    logger.info("🚀 Starting comprehensive chunking experiment...")
    
    # Initialize components
    fetcher = GitHubDataFetcher(GITHUB_TOKEN)
    chunker = CodeChunker()
    encoder = ChunkedVectorEncoder()
    milvus_manager = ChunkedMilvusManager("rag_chunks")
    
    # Connect to Milvus
    if not milvus_manager.connect(MILVUS_URI, MILVUS_TOKEN):
        return False
    
    # Define chunk strategies to test
    strategies = [
        ChunkType.FUNCTION,      # Most useful for code search
        ChunkType.FIXED_SIZE,    # Most granular
        ChunkType.SEMANTIC_BLOCK # Balanced approach
    ]
    
    # Create collections for each strategy
    for strategy in strategies:
        milvus_manager.create_collection_for_chunk_type(strategy)
    
    # Fetch all files first
    all_files = []
    for repo in repos:
        files = fetcher.get_repo_files(repo, max_files_per_repo)
        all_files.extend(files)
    
    logger.info(f"📁 Total files fetched: {len(all_files)}")
    
    # Process each chunking strategy
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n🔧 Processing strategy: {strategy.value}")
        
        # Generate chunks for all files
        all_chunks = []
        for file_data in all_files:
            chunks = chunker.chunk_code(
                file_content=file_data["content"],
                file_path=file_data["file_path"],
                repo_name=file_data["repo_name"],
                chunk_type=strategy,
                file_sha=file_data["sha"],
                url=file_data["url"]
            )
            all_chunks.extend(chunks)
        
        logger.info(f"📊 Generated {len(all_chunks)} chunks")
        
        if all_chunks:
            # Create vectors
            vectors = encoder.fit_transform_chunks(all_chunks, strategy)
            
            # Insert into Milvus
            success = milvus_manager.insert_chunks(all_chunks, vectors, strategy)
            
            results[strategy] = {
                "chunks": len(all_chunks),
                "success": success,
                "avg_chunk_size": sum(chunk.chunk_size for chunk in all_chunks) / len(all_chunks)
            }
    
    # Load all collections
    milvus_manager.load_collections()
    
    # Print results summary
    logger.info("\n📈 EXPERIMENT RESULTS:")
    logger.info("=" * 50)
    
    for strategy, result in results.items():
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {strategy.value}:")
        logger.info(f"   Chunks: {result['chunks']}")
        logger.info(f"   Avg size: {result['avg_chunk_size']:.0f} chars")
        logger.info(f"   Collection: rag_chunks_{strategy.value}")
    
    logger.info("\n🎯 RECOMMENDATIONS:")
    logger.info("• FUNCTION chunks: Best for code-specific queries")
    logger.info("• FIXED_SIZE chunks: Best for general text search")
    logger.info("• SEMANTIC_BLOCK chunks: Balanced for mixed queries")
    
    return True

def main():
    """Main function."""
    
    if not MILVUS_TOKEN:
        print("❌ MILVUS_TOKEN environment variable not set!")
        return
    
    # Test repositories (focused on backend for consistency)
    test_repos = [
        "fastapi/fastapi",
        "django/django"
    ]
    
    print("🎯 Chunked Loading Experiment")
    print("=" * 40)
    print(f"Repositories: {test_repos}")
    print(f"Strategies: function, fixed_size, semantic_block")
    print()
    
    confirm = input("Proceed with experiment? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    success = run_chunked_experiment(test_repos, max_files_per_repo=15)
    
    if success:
        print("\n🎉 Experiment completed successfully!")
        print("You now have 3 collections to compare:")
        print("• rag_chunks_function")
        print("• rag_chunks_fixed_size") 
        print("• rag_chunks_semantic_block")
    else:
        print("\n❌ Experiment failed. Check logs for details.")

if __name__ == "__main__":
    main() 