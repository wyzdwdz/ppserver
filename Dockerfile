FROM nvcr.io/nvidia/tensorrt:24.06-py3

WORKDIR /workspace
RUN apt update && apt install -y curl zip unzip tar pkg-config
RUN git clone --recurse-submodules https://github.com/wyzdwdz/ppserver.git && mkdir -p ./ppserver/build
WORKDIR /workspace/ppserver/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
RUN make . && make install
WORKDIR /workspace
RUN rm -rf ./ppserver /var/cache/apt/archives /var/lib/apt/lists/*
RUN apt clean