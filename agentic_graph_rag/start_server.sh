#!/bin/bash

# Synthea Chatbot Server Startup Script

echo "╔══════════════════════════════════════════════════╗"
echo "║     Synthea Chatbot - Starting Server           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: uv venv && source .venv/bin/activate && uv pip install -r requirements.txt"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY before continuing"
    echo "Press Enter when ready..."
    read
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Test Neo4j connection
echo "🔄 Testing Neo4j connection..."
python test_connection.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Neo4j connection test failed!"
    echo "Please ensure Neo4j is running and configured correctly."
    exit 1
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "🚀 Starting server..."
echo ""

# Start the server
cd backend && python server.py
