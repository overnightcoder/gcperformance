#!/usr/bin/env python3
"""
Simple test script to verify the DenggCert Chat & RAG System setup.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

async def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from core.config import settings
        print("✅ Core config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core config: {e}")
        return False
    
    try:
        from routers import chat_routes, vector_routes
        print("✅ Routers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import routers: {e}")
        return False
    
    try:
        from main import app
        print("✅ Main app imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main app: {e}")
        return False
    
    return True

async def test_milvus_connection():
    """Test Milvus connection."""
    print("\n🔍 Testing Milvus connection...")
    
    try:
        from core.database import init_milvus_connection, close_milvus_connection
        await init_milvus_connection()
        print("✅ Milvus connection successful")
        await close_milvus_connection()
        return True
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        print("💡 Make sure Milvus is running: docker-compose up -d")
        return False

def test_static_files():
    """Test if static files exist."""
    print("\n🔍 Testing static files...")
    
    required_files = [
        "templates/index.html",
        "static/css/style.css",
        "static/js/app.js"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

async def main():
    """Run all tests."""
    print("🚀 Testing DenggCert Chat & RAG System Setup\n")
    
    # Test imports
    imports_ok = await test_imports()
    
    # Test static files
    files_ok = test_static_files()
    
    # Test Milvus (optional - won't fail the test)
    milvus_ok = await test_milvus_connection()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Static Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"   Milvus Connection: {'✅ PASS' if milvus_ok else '⚠️  SKIP (Milvus not running)'}")
    
    if imports_ok and files_ok:
        print("\n🎉 Setup looks good! You can start the application with:")
        print("   python main.py")
        print("   or")
        print("   ./start.sh")
        return True
    else:
        print("\n❌ Setup has issues. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 