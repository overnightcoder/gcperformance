#!/usr/bin/env python3
"""
Interactive setup script for configuring cloud Milvus connections.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with cloud Milvus configuration."""
    
    print("🌐 Cloud Milvus Setup for DenggCert Chat & RAG System")
    print("=" * 60)
    
    # Get OpenAI API key
    openai_key = input("Enter your OpenAI API key: ").strip()
    if not openai_key:
        print("❌ OpenAI API key is required!")
        return False
    
    # Get Milvus provider
    print("\nSelect your Milvus provider:")
    print("1. Milvus Cloud (Zilliz)")
    print("2. AWS")
    print("3. Google Cloud Platform")
    print("4. Azure")
    print("5. Self-hosted")
    print("6. Other")
    
    provider = input("Enter your choice (1-6): ").strip()
    
    # Get Milvus connection details
    print("\nEnter your Milvus connection details:")
    host = input("Milvus host/endpoint: ").strip()
    if not host:
        print("❌ Host is required!")
        return False
    
    port_input = input("Port (default: 443 for cloud, 19530 for self-hosted): ").strip()
    port = int(port_input) if port_input else (443 if provider == "1" else 19530)
    
    # Get authentication details
    username = input("Username (leave empty if no authentication): ").strip()
    password = input("Password (leave empty if no authentication): ").strip()
    
    # SSL configuration
    ssl_input = input("Use SSL? (y/n, default: y): ").strip().lower()
    use_ssl = ssl_input != "n"
    
    # Collection name
    collection = input("Collection name (default: denggcert_documents): ").strip()
    if not collection:
        collection = "denggcert_documents"
    
    # Create .env content
    env_content = f"""# OpenAI Configuration
OPENAI_API_KEY={openai_key}

# Milvus Configuration
MILVUS_HOST={host}
MILVUS_PORT={port}
MILVUS_COLLECTION_NAME={collection}
MILVUS_USE_SSL={str(use_ssl).lower()}
"""
    
    if username:
        env_content += f"MILVUS_USERNAME={username}\n"
    
    if password:
        env_content += f"MILVUS_PASSWORD={password}\n"
    
    # Write .env file
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Configuration saved to {env_file}")
    print("\n📋 Configuration Summary:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   SSL: {use_ssl}")
    print(f"   Collection: {collection}")
    if username:
        print(f"   Username: {username}")
        print(f"   Password: {'*' * len(password)}")
    
    return True

def test_connection():
    """Test the Milvus connection."""
    print("\n🧪 Testing connection...")
    
    try:
        # Import and test
        from core.config import settings
        from core.database import init_milvus_connection, close_milvus_connection
        
        import asyncio
        asyncio.run(init_milvus_connection())
        print("✅ Connection successful!")
        asyncio.run(close_milvus_connection())
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Welcome to Cloud Milvus Setup!")
    
    # Check if .env already exists
    if Path(".env").exists():
        overwrite = input("A .env file already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Setup cancelled.")
            return
    
    # Create configuration
    if not create_env_file():
        print("Setup failed. Please try again.")
        return
    
    # Test connection
    print("\n" + "=" * 60)
    test_connection()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Start the application: python3 main.py")
    print("2. Open http://localhost:8000 in your browser")
    print("3. Add documents to your vector database")
    print("4. Start chatting with RAG capabilities!")
    
    print("\n📚 For more information, see CLOUD_MILVUS_SETUP.md")

if __name__ == "__main__":
    main() 