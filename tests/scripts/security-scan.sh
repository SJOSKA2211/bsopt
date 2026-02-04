#!/bin/bash
# =====================================================================
# DOCKER SECURITY SCANNING SCRIPT
# =====================================================================
# Purpose: Run Trivy security scans on all Docker images
# Features: Vulnerability scanning, compliance checks, report generation
# Usage: ./scripts/security-scan.sh
# =====================================================================

set -euo pipefail

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/security-reports"
VERSION="${VERSION:-latest}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =====================================================================
# Functions
# =====================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "====================================================================="
    echo "$1"
    echo "====================================================================="
    echo ""
}

check_trivy() {
    if ! command -v trivy &> /dev/null; then
        log_error "Trivy is not installed"
        log_info "Install Trivy:"
        log_info "  macOS: brew install trivy"
        log_info "  Linux: wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -"
        log_info "  Docker: docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy"
        exit 1
    fi
    log_success "Trivy is installed: $(trivy --version | head -n1)"
}

scan_image() {
    local image_name=$1
    local image_tag="${image_name}:${VERSION}"

    print_header "Scanning ${image_name}"

    # Check if image exists
    if ! docker image inspect "$image_tag" &> /dev/null; then
        log_warning "Image $image_tag not found. Skipping scan."
        return
    fi

    # JSON report (detailed)
    log_info "Generating detailed JSON report..."
    trivy image \
        --format json \
        --output "$REPORT_DIR/${image_name}-detailed.json" \
        "$image_tag"

    # Table report (high/critical only)
    log_info "Generating summary report (HIGH/CRITICAL)..."
    trivy image \
        --severity HIGH,CRITICAL \
        --format table \
        --output "$REPORT_DIR/${image_name}-summary.txt" \
        "$image_tag"

    # SARIF report (for GitHub Security)
    log_info "Generating SARIF report..."
    trivy image \
        --format sarif \
        --output "$REPORT_DIR/${image_name}.sarif" \
        "$image_tag"

    # Display summary in console
    echo ""
    log_info "Vulnerability Summary for ${image_name}:"
    trivy image \
        --severity HIGH,CRITICAL \
        --format table \
        "$image_tag"

    # Count vulnerabilities
    local critical=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$REPORT_DIR/${image_name}-detailed.json")
    local high=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$REPORT_DIR/${image_name}-detailed.json")
    local medium=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "MEDIUM")] | length' "$REPORT_DIR/${image_name}-detailed.json")
    local low=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "LOW")] | length' "$REPORT_DIR/${image_name}-detailed.json")

    echo ""
    log_info "Vulnerability Count:"
    echo "  CRITICAL: ${critical:-0}"
    echo "  HIGH:     ${high:-0}"
    echo "  MEDIUM:   ${medium:-0}"
    echo "  LOW:      ${low:-0}"

    # Save metrics
    echo "${image_name},${critical:-0},${high:-0},${medium:-0},${low:-0}" >> "$REPORT_DIR/vulnerability-metrics.csv"

    # Fail if critical vulnerabilities found
    if [ "${critical:-0}" -gt 0 ]; then
        log_error "${image_name} has ${critical} CRITICAL vulnerabilities!"
        return 1
    fi

    log_success "${image_name} scan completed"
    return 0
}

generate_summary_report() {
    print_header "Generating Summary Report"

    cat > "$REPORT_DIR/SECURITY_SUMMARY.md" << EOF
# Docker Security Scan Summary

**Scan Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Version:** ${VERSION}
**Scanner:** Trivy $(trivy --version | head -n1)

## Vulnerability Summary

| Image | Critical | High | Medium | Low |
|-------|----------|------|--------|-----|
EOF

    # Read metrics and generate table
    tail -n +2 "$REPORT_DIR/vulnerability-metrics.csv" | while IFS=',' read -r image critical high medium low; do
        echo "| $image | $critical | $high | $medium | $low |" >> "$REPORT_DIR/SECURITY_SUMMARY.md"
    done

    cat >> "$REPORT_DIR/SECURITY_SUMMARY.md" << EOF

## Production Readiness Criteria

✅ = Pass | ⚠️ = Warning | ❌ = Fail

| Image | Critical CVEs | High CVEs | Status |
|-------|---------------|-----------|--------|
EOF

    # Generate status table
    tail -n +2 "$REPORT_DIR/vulnerability-metrics.csv" | while IFS=',' read -r image critical high medium low; do
        if [ "$critical" -eq 0 ] && [ "$high" -eq 0 ]; then
            status="✅ PASS"
        elif [ "$critical" -eq 0 ]; then
            status="⚠️ WARNING"
        else
            status="❌ FAIL"
        fi
        echo "| $image | $critical | $high | $status |" >> "$REPORT_DIR/SECURITY_SUMMARY.md"
    done

    cat >> "$REPORT_DIR/SECURITY_SUMMARY.md" << EOF

## Recommendations

1. **Critical Vulnerabilities:** Update base images and dependencies immediately
2. **High Vulnerabilities:** Schedule update in next sprint
3. **Medium/Low Vulnerabilities:** Track and update during regular maintenance

## Detailed Reports

- JSON Reports: \`security-reports/*-detailed.json\`
- Summary Reports: \`security-reports/*-summary.txt\`
- SARIF Reports: \`security-reports/*.sarif\` (for GitHub Security)

## Next Steps

1. Review detailed reports for each image
2. Update vulnerable dependencies
3. Rebuild images with updated dependencies
4. Re-scan to verify fixes
5. Integrate security scanning into CI/CD pipeline

---

**Note:** This report is generated automatically. Always verify findings and apply security updates promptly.
EOF

    log_success "Summary report generated: $REPORT_DIR/SECURITY_SUMMARY.md"
}

# =====================================================================
# Main Script
# =====================================================================

print_header "Docker Security Scanning Script"

# Check prerequisites
check_trivy

# Create report directory
mkdir -p "$REPORT_DIR"
log_info "Report directory: $REPORT_DIR"

# Initialize metrics
echo "image,critical,high,medium,low" > "$REPORT_DIR/vulnerability-metrics.csv"

# Update Trivy DB
print_header "Updating Trivy Database"
trivy image --download-db-only
log_success "Trivy database updated"

# Scan all images
SCAN_FAILED=0

scan_image "bsopt-api" || SCAN_FAILED=1
scan_image "bsopt-worker" || SCAN_FAILED=1
scan_image "bsopt-frontend" || SCAN_FAILED=1
scan_image "bsopt-jupyter" || true  # Optional, don't fail build

# Generate summary report
generate_summary_report

# Display summary
print_header "Scan Results"
cat "$REPORT_DIR/SECURITY_SUMMARY.md"

# Exit with appropriate code
if [ $SCAN_FAILED -eq 1 ]; then
    log_error "Security scan failed due to critical vulnerabilities!"
    log_info "Review reports in: $REPORT_DIR"
    exit 1
else
    log_success "All security scans passed!"
    exit 0
fi
