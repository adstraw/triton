# Triton Dev Container

Dev container for Triton development with PyTorch 2.7, CUDA, and all build dependencies pre-installed.

## Setup

After the container starts, build Triton:

```bash
pip install -e . --no-build-isolation
```

The first build takes a while. Build artifacts persist on the host, so rebuilding the container is fast.

## Development

### Incremental builds

Use ninja directly for fastest iteration:

```bash
ninja -C build/cmake.linux-x86_64-cpython-3.12
```

### Clean rebuild

Remove build artifacts and return to Setup:

```bash
rm -rf build python/triton/_C
```

## Manual Container

If dev container integration fails, use the provided shell script:

```bash
.devcontainer/triton-dev.sh
```
