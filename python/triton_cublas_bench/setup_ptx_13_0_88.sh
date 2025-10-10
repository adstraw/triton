#!/bin/bash
#
# Script to download and setup PTX 13.0.88 for Blackwell FP4 support
#
# This version of PTX is required for:
# - MXFP4 block-scaled matmul
# - NVFP4 block-scaled matmul
#
# Usage:
#   ./setup_ptx_13_0_88.sh [install_directory]
#
# If install_directory is not specified, installs to current directory
#

set -e

# Parse arguments
INSTALL_DIR="${1:-.}"
# Convert to absolute path
INSTALL_DIR="$(mkdir -p "${INSTALL_DIR}" && cd "${INSTALL_DIR}" && pwd)"
PTX_VERSION="13.0.88"
ARCHIVE_NAME="cuda_nvcc-linux-x86_64-${PTX_VERSION}-archive"
ARCHIVE_FILE="${ARCHIVE_NAME}.tar.xz"
DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/${ARCHIVE_FILE}"

echo "=========================================="
echo "PTX ${PTX_VERSION} Setup Script"
echo "=========================================="
echo "Install directory: ${INSTALL_DIR}"
echo "Download URL: ${DOWNLOAD_URL}"
echo ""

# Create install directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Check if already downloaded
if [ -f "${ARCHIVE_FILE}" ]; then
    echo "✓ Archive already downloaded: ${ARCHIVE_FILE}"
else
    echo "⬇️  Downloading PTX ${PTX_VERSION}..."
    wget -q --show-progress "${DOWNLOAD_URL}" || {
        echo "❌ Download failed!"
        echo "Trying with curl..."
        curl -L -o "${ARCHIVE_FILE}" "${DOWNLOAD_URL}"
    }
    echo "✓ Download complete"
fi

# Check if already extracted
if [ -d "${ARCHIVE_NAME}" ]; then
    echo "✓ Archive already extracted: ${ARCHIVE_NAME}"
else
    echo "📦 Extracting archive..."
    tar -xf "${ARCHIVE_FILE}"
    echo "✓ Extraction complete"
fi

# Verify ptxas binary exists
PTXAS_PATH="${INSTALL_DIR}/${ARCHIVE_NAME}/bin/ptxas"
if [ -f "${PTXAS_PATH}" ]; then
    echo "✓ PTX binary found: ${PTXAS_PATH}"
    
    # Make it executable
    chmod +x "${PTXAS_PATH}"
    
    # Check version
    echo ""
    echo "PTX Version Info:"
    "${PTXAS_PATH}" --version || true
    
    echo ""
    echo "=========================================="
    echo "✅ Setup complete!"
    echo "=========================================="
    echo ""
    echo "To use this PTX version, set the environment variable:"
    echo ""
    echo "  export TRITON_PTXAS_PATH=\"${PTXAS_PATH}\""
    echo ""
    echo "Or use the provided benchmark scripts which set it automatically."
    echo ""
    
    # Write the path to a file for easy sourcing
    echo "export TRITON_PTXAS_PATH=\"${PTXAS_PATH}\"" > ptx_env.sh
    echo "✓ Environment file created: ptx_env.sh"
    echo "  (You can source this: source ptx_env.sh)"
    
else
    echo "❌ ERROR: PTX binary not found at ${PTXAS_PATH}"
    exit 1
fi

