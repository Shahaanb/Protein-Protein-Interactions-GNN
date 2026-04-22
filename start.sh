#!/bin/bash
# Proteome-X Local Startup Script
# Run this script if your Docker daemon is not active.

echo "==================================="
echo "Starting Proteome-X Local Setup"
echo "==================================="

# 1. Start backend
echo "-> Setting up Python Backend API..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Start FastAPI server in the background
uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd ..

# 2. Start frontend
echo "-> Setting up React Frontend..."
cd frontend
npm install
# Start Vite development server
npm run dev &
FRONTEND_PID=$!
cd ..

echo "==================================="
echo "Proteome-X is now Initializing!"
echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:5173"
echo "==================================="
echo "Press Ctrl+C to stop all services."

# Wait for user interrupt to kill sub-processes
trap "kill $BACKEND_PID $FRONTEND_PID" SIGINT SIGTERM
wait
