#!/bin/bash

echo "🚀 Starting DenggCert Chat & RAG System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Check if Milvus is running
echo "🔍 Checking Milvus connection..."
if curl -s http://localhost:19530/health > /dev/null 2>&1; then
    echo "✅ Milvus is running"
else
    echo "⚠️  Milvus is not running. Starting with Docker Compose..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d milvus-standalone
        echo "⏳ Waiting for Milvus to start..."
        sleep 10
    else
        echo "❌ Docker Compose not found. Please start Milvus manually:"
        echo "   docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone"
        echo "   Or use: docker-compose up -d"
    fi
fi

# Start the application
echo "🌟 Starting FastAPI application..."
echo "📱 Open your browser and go to: http://localhost:8000"
echo "🔑 Don't forget to enter your OpenAI API key!"
echo ""
echo "Press Ctrl+C to stop the application"

python3 main.py 