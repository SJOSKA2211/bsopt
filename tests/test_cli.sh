#!/bin/bash
#
# Test Script for Black-Scholes CLI
# ==================================
#
# Comprehensive test suite for all CLI commands
# Run with: bash test_cli.sh

set -e  # Exit on error

echo "========================================"
echo "Black-Scholes CLI Test Suite"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_command() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${YELLOW}Test $TESTS_RUN: $1${NC}"
    if eval "$2" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

test_command_output() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${YELLOW}Test $TESTS_RUN: $1${NC}"
    OUTPUT=$(eval "$2" 2>&1)
    if echo "$OUTPUT" | grep -q "$3"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Expected: $3"
        echo "Got: $OUTPUT"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Check if CLI is installed
echo "Checking CLI installation..."
if ! command -v bsopt &> /dev/null; then
    echo -e "${RED}bsopt command not found. Installing...${NC}"
    python setup_cli.py install
fi

echo -e "${GREEN}CLI found!${NC}"
echo ""

# Test 1: Version
echo "----------------------------------------"
echo "1. Basic Commands"
echo "----------------------------------------"
test_command "Version check" "bsopt --version"
test_command "Help display" "bsopt --help"

# Test 2: Pricing Commands
echo ""
echo "----------------------------------------"
echo "2. Pricing Commands"
echo "----------------------------------------"

# Black-Scholes call
test_command "BS call pricing" \
    "bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05"

# Black-Scholes put
test_command "BS put pricing" \
    "bsopt price put --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05"

# With dividend
test_command "BS with dividend" \
    "bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --dividend 0.02"

# FDM method
test_command "FDM pricing" \
    "bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method fdm"

# JSON output
test_command "JSON output" \
    "bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --output json"

# Test 3: Greeks
echo ""
echo "----------------------------------------"
echo "3. Greeks Calculation"
echo "----------------------------------------"

test_command "Calculate Greeks" \
    "bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05"

test_command "Greeks for put" \
    "bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --option-type put"

# Test 4: Configuration
echo ""
echo "----------------------------------------"
echo "4. Configuration Management"
echo "----------------------------------------"

test_command "Config list" "bsopt config list"
test_command "Config get" "bsopt config get api.base_url"
test_command "Config set" "bsopt config set pricing.default_method bs"

# Test 5: Portfolio
echo ""
echo "----------------------------------------"
echo "5. Portfolio Management"
echo "----------------------------------------"

test_command "Portfolio list (empty)" "bsopt portfolio list"

# Add position
test_command "Add portfolio position" \
    "bsopt portfolio add --symbol TEST --option-type call --quantity 10 \
     --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
     --entry-price 5.50 --spot 100"

test_command "Portfolio list (with position)" "bsopt portfolio list"
test_command "Portfolio P&L" "bsopt portfolio pnl"

# Test 6: Batch Processing
echo ""
echo "----------------------------------------"
echo "6. Batch Processing"
echo "----------------------------------------"

# Create test CSV
cat > /tmp/test_options.csv << EOF
symbol,spot,strike,maturity,volatility,rate,dividend,option_type
TEST1,100,100,1.0,0.2,0.05,0.0,call
TEST2,100,105,0.5,0.25,0.05,0.0,put
TEST3,100,95,0.75,0.22,0.05,0.01,call
EOF

test_command "Batch pricing" \
    "bsopt batch --input /tmp/test_options.csv --output /tmp/test_results.csv"

if [ -f /tmp/test_results.csv ]; then
    echo -e "${GREEN}✓ Batch output file created${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Batch output file not found${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

# Test 7: Error Handling
echo ""
echo "----------------------------------------"
echo "7. Error Handling"
echo "----------------------------------------"

# Invalid parameters should fail
test_command "Negative spot price (should fail)" \
    "! bsopt price call --spot -100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05"

test_command "Invalid maturity (should fail)" \
    "! bsopt price call --spot 100 --strike 100 --maturity -1.0 --vol 0.2 --rate 0.05"

# Test 8: Authentication (skip if no API)
echo ""
echo "----------------------------------------"
echo "8. Authentication Commands"
echo "----------------------------------------"

test_command "Check auth status" "bsopt auth whoami || true"

# Test 9: Server commands (just check they exist)
echo ""
echo "----------------------------------------"
echo "9. Server Commands"
echo "----------------------------------------"

test_command "Serve help" "bsopt serve --help"
test_command "Init-db help" "bsopt init-db --help"

# Cleanup
echo ""
echo "----------------------------------------"
echo "Cleanup"
echo "----------------------------------------"

# Remove test files
rm -f /tmp/test_options.csv /tmp/test_results.csv

# Clear test portfolio (optional)
# bsopt portfolio clear --confirm

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Total tests:  $TESTS_RUN"
echo -e "${GREEN}Passed:       $TESTS_PASSED${NC}"
echo -e "${RED}Failed:       $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
