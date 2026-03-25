#!/usr/bin/env python3
"""
Simple runner script for the GitHub to Milvus loader
===================================================

This script provides an easy way to run the data loader with different configurations.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path so we can import load_data
sys.path.append(str(Path(__file__).parent))

try:
    from load_data import load_github_data_to_milvus, logger
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install -r requirements_milvus.txt")
    sys.exit(1)

def setup_environment():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    else:
        logger.info("No .env file found, using system environment variables")

def main():
    """Main runner function."""
    
    # Setup environment
    setup_environment()
    
    # Configuration
    CLUSTER_NAME = "vibecoding_072025"
    COLLECTION_NAME = "rag_01"
    
    # Get credentials from environment
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    
    if not MILVUS_TOKEN:
        print("❌ MILVUS_TOKEN environment variable not set!")
        print("\nTo set up your credentials:")
        print("1. Create a .env file in this directory with:")
        print("   MILVUS_TOKEN=your_milvus_token_here")
        print("   GITHUB_TOKEN=your_github_token_here  # Optional but recommended")
        print("\n2. Or export them in your shell:")
        print("   export MILVUS_TOKEN='your_token'")
        print("   export GITHUB_TOKEN='your_token'")
        return
    
    # Milvus connection details - using serverless endpoint
    MILVUS_URI = "https://in03-28c44765e826237.serverless.gcp-us-west1.cloud.zilliz.com"
    
    # Repository configurations
    repo_configs = {
        "ai_ml": [
            "huggingface/transformers",
            "pytorch/pytorch", 
            "tensorflow/tensorflow",
            "openai/openai-python",
            "microsoft/DeepSpeed"
        ],
        "web_dev": [
            "vercel/next.js",
            "facebook/react",
            "vuejs/vue",
            "angular/angular",
            "sveltejs/svelte"
        ],
        "backend": [
            "fastapi/fastapi",
            "django/django",
            "rails/rails",
            "spring-projects/spring-boot",
            "gin-gonic/gin"
        ],
        "devtools": [
            "microsoft/vscode",
            "neovim/neovim",
            "jetbrains/intellij-community",
            "atom/atom",
            "code-server/code-server"
        ]
    }
    
    print("🚀 GitHub to Milvus Sparse Vector Loader")
    print("=" * 50)
    print(f"Cluster: {CLUSTER_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"GitHub Token: {'✅ Set' if GITHUB_TOKEN else '⚠️  Not set (rate limited)'}")
    print()
    
    # Let user choose category or use all
    print("Available repository categories:")
    for i, (category, repos) in enumerate(repo_configs.items(), 1):
        print(f"{i}. {category}: {len(repos)} repositories")
    print(f"{len(repo_configs) + 1}. All categories")
    print()
    
    try:
        choice = input("Choose category (1-5) or press Enter for AI/ML: ").strip()
        
        if not choice:
            choice = "1"  # Default to AI/ML
            
        choice_num = int(choice)
        
        if choice_num == len(repo_configs) + 1:
            # All categories
            repos_to_load = []
            for repos in repo_configs.values():
                repos_to_load.extend(repos)
        elif 1 <= choice_num <= len(repo_configs):
            # Specific category
            category_name = list(repo_configs.keys())[choice_num - 1]
            repos_to_load = repo_configs[category_name]
            print(f"Selected category: {category_name}")
        else:
            print("Invalid choice, using AI/ML repositories")
            repos_to_load = repo_configs["ai_ml"]
            
    except (ValueError, KeyboardInterrupt):
        print("Using default AI/ML repositories")
        repos_to_load = repo_configs["ai_ml"]
    
    print(f"\nRepositories to index: {repos_to_load}")
    print(f"Total repositories: {len(repos_to_load)}")
    
    # Confirm before proceeding
    confirm = input("\nProceed with data loading? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Run the loader
    success = load_github_data_to_milvus(
        repos=repos_to_load,
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        max_files_per_repo=25  # Reasonable limit to avoid rate limiting
    )
    
    if success:
        print("\n🎉 Data loading completed successfully!")
        print(f"✅ Collection '{COLLECTION_NAME}' is ready for use")
        print("\nNext steps:")
        print("1. You can now search the collection using sparse vectors")
        print("2. Use the collection for RAG applications")
        print("3. Check the Milvus console for collection statistics")
    else:
        print("\n❌ Data loading failed. Check the logs for details.")

if __name__ == "__main__":
    main() 