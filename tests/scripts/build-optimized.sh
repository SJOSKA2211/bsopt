#!/bin/bash
# =====================================================================
# BUILDKIT-ENABLED DOCKER BUILD SCRIPT
# =====================================================================
# Purpose: Build all optimized Docker images with BuildKit features
# Features: Parallel builds, cache optimization, build metrics
# Usage: ./scripts/build-optimized.sh [OPTIONS]
# =====================================================================

set -euo pipefail

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-}"
PLATFORM="${PLATFORM:-linux/amd64}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_buildkit() {
    if ! docker buildx version &> /dev/null; then
        log_error "Docker BuildKit is not available. Please install Docker 19.03+ or enable BuildKit."
        exit 1
    fi
    log_success "Docker BuildKit is available"
}

build_image() {
    local name=$1
    local dockerfile=$2
    local context=$3
    local image_name=$4

    print_header "Building $name"

    local start_time=$(date +%s)

    # Build command with BuildKit features
    DOCKER_BUILDKIT=1 docker build \
        --file "$dockerfile" \
        --tag "${image_name}:${VERSION}" \
        --tag "${image_name}:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --platform "$PLATFORM" \
        --progress=plain \
        "$context" 2>&1 | tee "/tmp/build-${name}.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "Built $name in ${duration}s"

    # Get image size
    local size=$(docker images "${image_name}:${VERSION}" --format "{{.Size}}")
    log_info "Image size: $size"

    # Save metrics
    echo "$name,$duration,$size" >> "$PROJECT_ROOT/build-metrics.csv"
}

build_with_cache_export() {
    local name=$1
    local dockerfile=$2
    local context=$3
    local image_name=$4

    print_header "Building $name (with cache export)"

    local start_time=$(date +%s)

    # Build with cache export to registry or local
    docker buildx build \
        --file "$dockerfile" \
        --tag "${image_name}:${VERSION}" \
        --tag "${image_name}:latest" \
        --platform "$PLATFORM" \
        --cache-from=type=local,src=/tmp/.buildx-cache \
        --cache-to=type=local,dest=/tmp/.buildx-cache-new,mode=max \
        --load \
        --progress=plain \
        "$context" 2>&1 | tee "/tmp/build-${name}.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "Built $name in ${duration}s"

    # Rotate cache
    rm -rf /tmp/.buildx-cache
    mv /tmp/.buildx-cache-new /tmp/.buildx-cache || true

    # Get image size
    local size=$(docker images "${image_name}:${VERSION}" --format "{{.Size}}")
    log_info "Image size: $size"

    echo "$name,$duration,$size" >> "$PROJECT_ROOT/build-metrics.csv"
}

tag_and_push() {
    local image_name=$1

    if [ -n "$REGISTRY" ]; then
        log_info "Tagging and pushing ${image_name} to ${REGISTRY}"
        docker tag "${image_name}:${VERSION}" "${REGISTRY}/${image_name}:${VERSION}"
        docker tag "${image_name}:${VERSION}" "${REGISTRY}/${image_name}:latest"
        docker push "${REGISTRY}/${image_name}:${VERSION}"
        docker push "${REGISTRY}/${image_name}:latest"
        log_success "Pushed ${image_name} to registry"
    fi
}

# =====================================================================
# Main Script
# =====================================================================

print_header "Docker BuildKit Optimized Build Script"

log_info "Project Root: $PROJECT_ROOT"
log_info "Version: $VERSION"
log_info "Platform: $PLATFORM"
[ -n "$REGISTRY" ] && log_info "Registry: $REGISTRY"

# Check prerequisites
check_buildkit

# Initialize metrics file
echo "image,build_time_seconds,size" > "$PROJECT_ROOT/build-metrics.csv"

# Change to project root
cd "$PROJECT_ROOT"

# =====================================================================
# Build Images
# =====================================================================

# Build API (FastAPI)
build_with_cache_export \
    "api" \
    "Dockerfile.api.optimized" \
    "." \
    "bsopt-api"

# Build Worker (Celery)
build_with_cache_export \
    "worker" \
    "Dockerfile.worker" \
    "." \
    "bsopt-worker"

# Build Frontend (React)
build_with_cache_export \
    "frontend" \
    "frontend/Dockerfile" \
    "frontend" \
    "bsopt-frontend"

# Build Jupyter (Optional, for development)
if [ "${BUILD_JUPYTER:-false}" = "true" ]; then
    build_with_cache_export \
        "jupyter" \
        "Dockerfile.jupyter.optimized" \
        "." \
        "bsopt-jupyter"
fi

# =====================================================================
# Post-Build Actions
# =====================================================================

print_header "Build Summary"

# Display metrics
log_info "Build Metrics:"
column -t -s',' "$PROJECT_ROOT/build-metrics.csv"

# Push to registry if configured
if [ -n "$REGISTRY" ]; then
    print_header "Pushing Images to Registry"
    tag_and_push "bsopt-api"
    tag_and_push "bsopt-worker"
    tag_and_push "bsopt-frontend"
    [ "${BUILD_JUPYTER:-false}" = "true" ] && tag_and_push "bsopt-jupyter"
fi

# List built images
print_header "Built Images"
docker images | grep -E "bsopt-(api|worker|frontend|jupyter)"

# Security scan (optional, requires Trivy)
if command -v trivy &> /dev/null; then
    print_header "Security Scanning (Trivy)"
    log_info "Running Trivy security scans..."

    mkdir -p "$PROJECT_ROOT/security-reports"

    for image in bsopt-api bsopt-worker bsopt-frontend; do
        log_info "Scanning ${image}:${VERSION}"
        trivy image \
            --severity HIGH,CRITICAL \
            --format json \
            --output "$PROJECT_ROOT/security-reports/${image}-scan.json" \
            "${image}:${VERSION}"

        # Display summary
        trivy image \
            --severity HIGH,CRITICAL \
            "${image}:${VERSION}"
    done

    log_success "Security scans completed. Reports saved to security-reports/"
else
    log_warning "Trivy not found. Skipping security scans."
    log_info "Install Trivy: https://github.com/aquasecurity/trivy"
fi

# =====================================================================
# Cleanup
# =====================================================================

print_header "Cleanup"

# Remove dangling images
log_info "Removing dangling images..."
docker image prune -f

log_success "Build script completed successfully!"

# Display next steps
print_header "Next Steps"
echo "1. Review build metrics: cat build-metrics.csv"
echo "2. Review security reports: ls -la security-reports/"
echo "3. Test images locally: docker-compose -f docker-compose.production.yml up"
echo "4. Deploy to production: docker stack deploy or kubectl apply"

exit 0
