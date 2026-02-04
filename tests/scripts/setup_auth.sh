#!/bin/bash

# JWT Authentication System - Quick Setup Script
# This script automates the setup of the authentication system

set -e  # Exit on error

echo "=========================================="
echo "  JWT Authentication System Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION found${NC}"
echo ""

# Check if PostgreSQL is installed
echo "Checking PostgreSQL installation..."
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}Warning: PostgreSQL not found${NC}"
    echo "Install with: sudo apt-get install postgresql"
else
    PG_VERSION=$(psql --version)
    echo -e "${GREEN}✓ $PG_VERSION found${NC}"
fi
echo ""

# Check if Redis is installed
echo "Checking Redis installation..."
if ! command -v redis-cli &> /dev/null; then
    echo -e "${YELLOW}Warning: Redis not found${NC}"
    echo "Install with: sudo apt-get install redis-server"
else
    REDIS_VERSION=$(redis-cli --version)
    echo -e "${GREEN}✓ $REDIS_VERSION found${NC}"

    # Test Redis connection
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${YELLOW}Warning: Redis is installed but not running${NC}"
        echo "Start with: sudo systemctl start redis"
    fi
fi
echo ""

# Create virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
if [ -f "requirements-auth.txt" ]; then
    pip install --upgrade pip > /dev/null
    pip install -r requirements-auth.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements-auth.txt not found${NC}"
    exit 1
fi
echo ""

# Create .env file if it doesn't exist
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env file..."

    # Generate a secure JWT secret
    JWT_SECRET=$(openssl rand -hex 32)

    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/options_db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# JWT Authentication
JWT_SECRET=$JWT_SECRET
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Password Requirements
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_DIGIT=true
PASSWORD_REQUIRE_SPECIAL=false
BCRYPT_ROUNDS=12

# Rate Limiting (per hour)
RATE_LIMIT_FREE=100
RATE_LIMIT_PRO=10000
RATE_LIMIT_ENTERPRISE=0

# Application Configuration
ENVIRONMENT=dev
DEBUG=true
LOG_LEVEL=INFO

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# API Configuration
API_PREFIX=/api/v1
EOF

    echo -e "${GREEN}✓ .env file created with secure JWT secret${NC}"
    echo -e "${YELLOW}⚠ Please update DATABASE_URL if needed${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi
echo ""

# Database setup
echo "Database setup..."
if command -v psql &> /dev/null; then
    echo "Would you like to create the database? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Creating database 'options_db'..."

        # Check if database exists
        if psql -lqt | cut -d \| -f 1 | grep -qw options_db; then
            echo -e "${GREEN}✓ Database 'options_db' already exists${NC}"
        else
            createdb options_db 2>/dev/null && echo -e "${GREEN}✓ Database 'options_db' created${NC}" || echo -e "${YELLOW}Warning: Could not create database (may need superuser access)${NC}"
        fi

        # Apply schema
        if [ -f "src/database/schema.sql" ]; then
            echo "Applying database schema..."
            psql options_db < src/database/schema.sql && echo -e "${GREEN}✓ Database schema applied${NC}" || echo -e "${YELLOW}Warning: Could not apply schema${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Skipping database setup (PostgreSQL not installed)${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review and update .env file if needed:"
echo "   nano .env"
echo ""
echo "2. Start the FastAPI server:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "3. Test the authentication system:"
echo "   python test_auth.py"
echo ""
echo "4. View API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "For detailed documentation, see:"
echo "   - AUTH_SETUP.md"
echo ""
echo "=========================================="
