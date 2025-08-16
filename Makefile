# Tonkatsu-OS Development Makefile

.PHONY: help setup dev dev-backend dev-frontend install-deps clean test lint format

# Default target
help:
	@echo "🔬 Tonkatsu-OS Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup           - Set up development environment"
	@echo "  install-deps    - Install all dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev             - Start both backend and frontend"
	@echo "  dev-backend     - Start FastAPI backend server"
	@echo "  dev-frontend    - Start React/Next.js frontend"
	@echo ""
	@echo "Quality Commands:"
	@echo "  test            - Run all tests"
	@echo "  lint            - Run linting"
	@echo "  format          - Format code"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean           - Clean build artifacts"
	@echo "  reset-db        - Reset database"

# Setup development environment
setup:
	@python3 scripts/setup_dev.py

# Install dependencies
install-deps:
	@echo "📦 Installing Python dependencies..."
	poetry install
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install

# Start both services
dev:
	@echo "🚀 Starting Tonkatsu-OS (Backend + Frontend)"
	@echo "This will start both services in parallel..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "Press Ctrl+C to stop both services"
	@trap 'kill $$backend_pid $$frontend_pid 2>/dev/null' EXIT; \
	poetry run python scripts/start_backend.py & backend_pid=$$!; \
	cd frontend && npm run dev & frontend_pid=$$!; \
	wait

# Start backend only
dev-backend:
	@echo "🔧 Starting Tonkatsu-OS Backend Server..."
	poetry run python scripts/start_backend.py

# Start frontend only  
dev-frontend:
	@echo "🌐 Starting Tonkatsu-OS Frontend..."
	python3 scripts/start_frontend.py

# Run tests
test:
	@echo "🧪 Running tests..."
	poetry run pytest tests/ -v --cov=tonkatsu_os

# Run linting
lint:
	@echo "🔍 Running linting..."
	poetry run flake8 src/tonkatsu_os/
	poetry run mypy src/tonkatsu_os/
	cd frontend && npm run lint

# Format code
format:
	@echo "✨ Formatting code..."
	poetry run black src/tonkatsu_os/
	poetry run isort src/tonkatsu_os/
	cd frontend && npm run format

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
	rm -rf frontend/.next/
	rm -rf dist/ build/

# Reset database
reset-db:
	@echo "🗄️ Resetting database..."
	rm -f raman_spectra.db
	@echo "Database reset. It will be recreated on next startup."

# Production build
build:
	@echo "🏗️ Building for production..."
	cd frontend && npm run build
	poetry build

# Install pre-commit hooks
install-hooks:
	@echo "🪝 Installing pre-commit hooks..."
	poetry run pre-commit install

# Update dependencies
update-deps:
	@echo "⬆️ Updating dependencies..."
	poetry update
	cd frontend && npm update

# Run security audit
audit:
	@echo "🔒 Running security audit..."
	poetry run safety check
	cd frontend && npm audit

# Generate API documentation
docs:
	@echo "📚 Generating API documentation..."
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "Start the backend server and visit the URL above"

# Run development server with hot reload
dev-hot:
	@echo "🔥 Starting with hot reload..."
	poetry run uvicorn tonkatsu_os.api.main:app --reload --host 0.0.0.0 --port 8000

# Database backup
backup-db:
	@echo "💾 Backing up database..."
	@if [ -f raman_spectra.db ]; then \
		cp raman_spectra.db "backup_$$(date +%Y%m%d_%H%M%S).db"; \
		echo "Database backed up successfully"; \
	else \
		echo "No database file found"; \
	fi

# Show project status
status:
	@echo "📊 Tonkatsu-OS Project Status"
	@echo "============================="
	@echo ""
	@echo "📂 Project Structure:"
	@find src/tonkatsu_os -name "*.py" | wc -l | xargs echo "  Python files:"
	@find frontend/src -name "*.tsx" -o -name "*.ts" | wc -l | xargs echo "  TypeScript files:"
	@echo ""
	@echo "📊 Database:"
	@if [ -f raman_spectra.db ]; then \
		echo "  Database exists (size: $$(du -h raman_spectra.db | cut -f1))"; \
	else \
		echo "  No database found"; \
	fi
	@echo ""
	@echo "🔧 Dependencies:"
	@poetry show --only=main | wc -l | xargs echo "  Python packages:"
	@if [ -f frontend/package.json ]; then \
		cat frontend/package.json | grep -c '"' | xargs echo "  NPM packages: ~"; \
	fi