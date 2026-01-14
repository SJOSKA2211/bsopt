#!/bin/bash
# =====================================================================
# VERSION MANAGEMENT SCRIPT
# =====================================================================
# Purpose: Manage semantic versioning across the project
# Usage: ./scripts/version.sh [COMMAND] [OPTIONS]
# Commands:
#   get                  - Display current version
#   bump [major|minor|patch] - Increment version number
#   set VERSION          - Set specific version
#   tag                  - Create git tag for current version
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION_FILE="$PROJECT_ROOT/VERSION.txt"
VERSION_JSON="$PROJECT_ROOT/.version.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current version
get_version() {
    if [ -f "$VERSION_FILE" ]; then
        cat "$VERSION_FILE"
    else
        echo "0.0.0"
    fi
}

# Parse version into components
parse_version() {
    local version=$1
    IFS='.' read -ra PARTS <<< "$version"
    MAJOR="${PARTS[0]}"
    MINOR="${PARTS[1]}"
    PATCH="${PARTS[2]}"
}

# Bump version
bump_version() {
    local bump_type=$1
    local current_version=$(get_version)

    parse_version "$current_version"

    case $bump_type in
        major)
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        minor)
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        patch)
            PATCH=$((PATCH + 1))
            ;;
        *)
            log_error "Invalid bump type: $bump_type (use: major, minor, or patch)"
            exit 1
            ;;
    esac

    local new_version="${MAJOR}.${MINOR}.${PATCH}"
    echo "$new_version"
}

# Set version in all files
set_version() {
    local version=$1

    # Validate semantic version format
    if ! [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: $version (expected: X.Y.Z)"
        exit 1
    fi

    log_info "Setting version to $version..."

    # Update VERSION.txt
    echo "$version" > "$VERSION_FILE"
    log_success "Updated VERSION.txt"

    # Update pyproject.toml
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        sed -i "s/^version = .*/version = \"$version\"/" "$PROJECT_ROOT/pyproject.toml"
        log_success "Updated pyproject.toml"
    fi

    # Update frontend package.json
    if [ -f "$PROJECT_ROOT/frontend/package.json" ]; then
        sed -i "s/\"version\": \".*\"/\"version\": \"$version\"/" "$PROJECT_ROOT/frontend/package.json"
        log_success "Updated frontend/package.json"
    fi

    # Update .version.json
    if [ -f "$VERSION_JSON" ]; then
        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        local git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

        jq --arg version "$version" \
           --arg timestamp "$timestamp" \
           --arg commit "$git_commit" \
           --arg branch "$git_branch" \
           '.version = $version | .build.timestamp = $timestamp | .build.git_commit = $commit | .build.git_branch = $branch' \
           "$VERSION_JSON" > "$VERSION_JSON.tmp" && mv "$VERSION_JSON.tmp" "$VERSION_JSON"

        log_success "Updated .version.json"
    fi

    # Update src/api/main.py version string
    if [ -f "$PROJECT_ROOT/src/api/main.py" ]; then
        sed -i "s/version=\".*\"/version=\"$version\"/" "$PROJECT_ROOT/src/api/main.py"
        log_success "Updated src/api/main.py"
    fi

    log_success "Version set to $version across all files"
}

# Create git tag
create_tag() {
    local version=$(get_version)
    local tag="v$version"

    log_info "Creating git tag: $tag"

    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_error "Tag $tag already exists"
        exit 1
    fi

    git tag -a "$tag" -m "Release version $version"
    log_success "Created tag: $tag"
    log_info "Push tag with: git push origin $tag"
}

# Display version info
show_version_info() {
    local version=$(get_version)

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Black-Scholes Option Pricing Platform - Version Information"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Current Version: $version"
    echo ""

    if [ -f "$VERSION_JSON" ]; then
        echo "Build Information:"
        jq -r '.build | "  Timestamp:   \(.timestamp)\n  Git Commit:  \(.git_commit)\n  Git Branch:  \(.git_branch)"' "$VERSION_JSON"
        echo ""

        echo "Component Versions:"
        jq -r '.components | to_entries[] | "  \(.key | ascii_upcase): \(.value)"' "$VERSION_JSON"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

# Main command router
case "${1:-get}" in
    get)
        get_version
        ;;

    info)
        show_version_info
        ;;

    bump)
        if [ $# -lt 2 ]; then
            log_error "Usage: $0 bump [major|minor|patch]"
            exit 1
        fi
        new_version=$(bump_version "$2")
        set_version "$new_version"
        log_success "Version bumped to $new_version"
        ;;

    set)
        if [ $# -lt 2 ]; then
            log_error "Usage: $0 set VERSION"
            exit 1
        fi
        set_version "$2"
        ;;

    tag)
        create_tag
        ;;

    help|--help|-h)
        echo "Version Management Script"
        echo ""
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  get                       Display current version"
        echo "  info                      Display detailed version information"
        echo "  bump [major|minor|patch]  Increment version number"
        echo "  set VERSION               Set specific version (X.Y.Z format)"
        echo "  tag                       Create git tag for current version"
        echo "  help                      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 get                    # Show current version"
        echo "  $0 bump patch             # 2.1.0 -> 2.1.1"
        echo "  $0 bump minor             # 2.1.0 -> 2.2.0"
        echo "  $0 bump major             # 2.1.0 -> 3.0.0"
        echo "  $0 set 2.5.0              # Set to specific version"
        echo "  $0 tag                    # Create git tag v2.1.0"
        ;;

    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
