#!/bin/bash
# Tonkatsu-OS Local Development Startup Script

set -e

echo "🔬 Starting Tonkatsu-OS Development Environment"
echo "=============================================="

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first:"
    echo "   https://nodejs.org/"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend && npm install && cd ..

echo ""
echo "🚀 Starting both backend and frontend..."
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=============================================="

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🔧 Starting backend server..."
poetry run python scripts/start_backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "🌐 Starting frontend server..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 3

echo ""
echo "✅ Both servers are running!"
echo "   Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "   Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "👀 Watch this terminal for logs from both servers"
echo "🔗 Open http://localhost:3000 in your browser to use the app"

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID