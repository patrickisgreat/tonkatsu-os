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

# Setup serial port permissions for spectrometer
echo "🔌 Checking serial port permissions..."
SERIAL_PORT="/dev/ttyUSB0"
if [ -e "$SERIAL_PORT" ]; then
    # Check if another process is using the port
    PORT_PID=$(lsof -t "$SERIAL_PORT" 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "⚠️  Serial port $SERIAL_PORT is in use by PID $PORT_PID"
        echo "   Killing stale process..."
        kill $PORT_PID 2>/dev/null || true
        sleep 1
        echo "✅ Stale process killed"
    fi

    if [ ! -r "$SERIAL_PORT" ] || [ ! -w "$SERIAL_PORT" ]; then
        echo "⚠️  Serial port $SERIAL_PORT needs permissions"
        echo "   Running: sudo chmod 666 $SERIAL_PORT"
        sudo chmod 666 "$SERIAL_PORT"
        if [ $? -eq 0 ]; then
            echo "✅ Serial port permissions set"
        else
            echo "⚠️  Could not set permissions. You may need to run manually:"
            echo "   sudo chmod 666 $SERIAL_PORT"
            echo "   Or add yourself to dialout group: sudo usermod -aG dialout \$USER"
        fi
    else
        echo "✅ Serial port $SERIAL_PORT is accessible"
    fi
else
    echo "ℹ️  Serial port $SERIAL_PORT not found (spectrometer not connected?)"
fi

# Also check ttyACM0 as alternative
ALT_PORT="/dev/ttyACM0"
if [ -e "$ALT_PORT" ]; then
    if [ ! -r "$ALT_PORT" ] || [ ! -w "$ALT_PORT" ]; then
        echo "⚠️  Serial port $ALT_PORT needs permissions"
        sudo chmod 666 "$ALT_PORT" 2>/dev/null || true
    fi
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