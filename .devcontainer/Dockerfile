FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        build-essential \
        clang \
        lld \
        mold \
        ccache \
        pkg-config \
        python3-dev \
        libssl-dev \
        zlib1g-dev \
        ca-certificates \
        curl \
        git \
        libclang-18-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

COPY python/requirements.txt /tmp/requirements.txt
COPY python/test-requirements.txt /tmp/test-requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/test-requirements.txt \
    && rm /tmp/requirements.txt /tmp/test-requirements.txt

ENV TRITON_HOME=/triton \
    CPATH=/usr/include/x86_64-linux-gnu \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    PYTHONPATH=/triton/python/triton_kernels:/triton/python:${PYTHONPATH} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=1 \
    TRITON_APPEND_CMAKE_ARGS="-DCMAKE_LINKER=mold -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold" \
    PATH=${PATH}:/triton/python/build/cmake.linux-x86_64-cpython-3.12/bin:/triton/.triton/llvm/llvm-ubuntu-x64/bin \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /triton

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
