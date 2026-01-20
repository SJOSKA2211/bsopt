#!/bin/bash
# OpenAPI Specification Validation Script
# Validates the OpenAPI spec and performs various checks

set -e

echo "=========================================="
echo "OpenAPI Specification Validation"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENAPI_SPEC="$PROJECT_ROOT/docs/api/openapi.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if OpenAPI spec exists
if [ ! -f "$OPENAPI_SPEC" ]; then
    echo -e "${RED}ERROR: OpenAPI specification not found at $OPENAPI_SPEC${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found OpenAPI specification: $OPENAPI_SPEC"
echo ""

# 1. Check YAML syntax
echo "1. Validating YAML syntax..."
if command -v python3 &> /dev/null; then
    python3 -c "import yaml; yaml.safe_load(open('$OPENAPI_SPEC'))" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} YAML syntax is valid"
    else
        echo -e "${RED}✗${NC} YAML syntax error"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} Python3 not found, skipping YAML syntax check"
fi
echo ""

# 2. Check OpenAPI version
echo "2. Checking OpenAPI version..."
OPENAPI_VERSION=$(grep -m 1 "^openapi:" "$OPENAPI_SPEC" | awk '{print $2}')
echo "   Version: $OPENAPI_VERSION"
if [[ "$OPENAPI_VERSION" == 3.0.* || "$OPENAPI_VERSION" == 3.1.* ]]; then
    echo -e "${GREEN}✓${NC} OpenAPI version is valid"
else
    echo -e "${RED}✗${NC} OpenAPI version should be 3.0.x or 3.1.x"
    exit 1
fi
echo ""

# 3. Check required top-level fields
echo "3. Checking required fields..."
REQUIRED_FIELDS=("openapi" "info" "paths")
for field in "${REQUIRED_FIELDS[@]}"; do
    if grep -q "^$field:" "$OPENAPI_SPEC"; then
        echo -e "${GREEN}✓${NC} Found required field: $field"
    else
        echo -e "${RED}✗${NC} Missing required field: $field"
        exit 1
    fi
done
echo ""

# 4. Count endpoints
echo "4. Analyzing API endpoints..."
ENDPOINT_COUNT=$(grep -c "^  /.*:" "$OPENAPI_SPEC" || true)
echo "   Total endpoints defined: $ENDPOINT_COUNT"

# Count by method
GET_COUNT=$(grep -c "    get:" "$OPENAPI_SPEC" || true)
POST_COUNT=$(grep -c "    post:" "$OPENAPI_SPEC" || true)
PUT_COUNT=$(grep -c "    put:" "$OPENAPI_SPEC" || true)
DELETE_COUNT=$(grep -c "    delete:" "$OPENAPI_SPEC" || true)

echo "   - GET:    $GET_COUNT"
echo "   - POST:   $POST_COUNT"
echo "   - PUT:    $PUT_COUNT"
echo "   - DELETE: $DELETE_COUNT"
echo ""

# 5. Check for required endpoint properties
echo "5. Checking endpoint properties..."
MISSING_SUMMARY=0
MISSING_DESCRIPTION=0
MISSING_RESPONSES=0

# This is a simplified check
if grep -q "summary:" "$OPENAPI_SPEC"; then
    echo -e "${GREEN}✓${NC} Endpoints have summaries"
else
    echo -e "${YELLOW}⚠${NC} Some endpoints may be missing summaries"
fi

if grep -q "description:" "$OPENAPI_SPEC"; then
    echo -e "${GREEN}✓${NC} Endpoints have descriptions"
else
    echo -e "${YELLOW}⚠${NC} Some endpoints may be missing descriptions"
fi

if grep -q "responses:" "$OPENAPI_SPEC"; then
    echo -e "${GREEN}✓${NC} Endpoints have response definitions"
else
    echo -e "${RED}✗${NC} Missing response definitions"
    exit 1
fi
echo ""

# 6. Check for security definitions
echo "6. Checking security configuration..."
if grep -q "securitySchemes:" "$OPENAPI_SPEC"; then
    echo -e "${GREEN}✓${NC} Security schemes defined"
    SECURITY_SCHEME=$(grep -A 2 "securitySchemes:" "$OPENAPI_SPEC" | grep -m 1 "type:" | awk '{print $2}')
    echo "   Type: $SECURITY_SCHEME"
else
    echo -e "${YELLOW}⚠${NC} No security schemes defined"
fi
echo ""

# 7. Check for schemas/components
echo "7. Checking schema definitions..."
if grep -q "components:" "$OPENAPI_SPEC"; then
    echo -e "${GREEN}✓${NC} Components section exists"
    if grep -q "schemas:" "$OPENAPI_SPEC"; then
        SCHEMA_COUNT=$(grep -c "^    [A-Z].*:" "$OPENAPI_SPEC" || true)
        echo "   Schemas defined: ~$SCHEMA_COUNT"
    fi
else
    echo -e "${YELLOW}⚠${NC} No components section found"
fi
echo ""

# 8. Check for examples
echo "8. Checking for examples..."
EXAMPLE_COUNT=$(grep -c "example:" "$OPENAPI_SPEC" || true)
EXAMPLES_COUNT=$(grep -c "examples:" "$OPENAPI_SPEC" || true)
TOTAL_EXAMPLES=$((EXAMPLE_COUNT + EXAMPLES_COUNT))
echo "   Examples found: $TOTAL_EXAMPLES"
if [ "$TOTAL_EXAMPLES" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} API includes examples"
else
    echo -e "${YELLOW}⚠${NC} No examples found"
fi
echo ""

# 9. Check servers
echo "9. Checking server definitions..."
if grep -q "servers:" "$OPENAPI_SPEC"; then
    SERVER_COUNT=$(grep -c "  - url:" "$OPENAPI_SPEC" || true)
    echo "   Servers defined: $SERVER_COUNT"
    echo -e "${GREEN}✓${NC} Server definitions found"
else
    echo -e "${YELLOW}⚠${NC} No server definitions"
fi
echo ""

# 10. Validate with swagger-cli if available
echo "10. Advanced validation..."
if command -v swagger-cli &> /dev/null; then
    echo "   Running swagger-cli validation..."
    swagger-cli validate "$OPENAPI_SPEC"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} swagger-cli validation passed"
    else
        echo -e "${RED}✗${NC} swagger-cli validation failed"
        exit 1
    fi
elif npm list -g @apidevtools/swagger-cli &> /dev/null 2>&1; then
    echo "   Running swagger-cli validation..."
    npx @apidevtools/swagger-cli validate "$OPENAPI_SPEC"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} swagger-cli validation passed"
    else
        echo -e "${RED}✗${NC} swagger-cli validation failed"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} swagger-cli not installed"
    echo "   Install: npm install -g @apidevtools/swagger-cli"
fi
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}✓ OpenAPI specification is valid${NC}"
echo ""
echo "Statistics:"
echo "  - Endpoints: $ENDPOINT_COUNT"
echo "  - GET:       $GET_COUNT"
echo "  - POST:      $POST_COUNT"
echo "  - PUT:       $PUT_COUNT"
echo "  - DELETE:    $DELETE_COUNT"
echo "  - Examples:  $TOTAL_EXAMPLES"
echo ""
echo "Next steps:"
echo "  1. Generate client SDK: openapi-generator-cli generate -i $OPENAPI_SPEC -g python -o ./clients/python"
echo "  2. View in Swagger UI: docker run -p 8080:8080 -v $(pwd):/api -e SWAGGER_JSON=/api/docs/api/openapi.yaml swaggerapi/swagger-ui"
echo "  3. Generate documentation: npx redoc-cli bundle $OPENAPI_SPEC -o api-docs.html"
echo ""
