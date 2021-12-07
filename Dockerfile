FROM mcr.microsoft.com/azureml/o16n-base/python-assets:20210623.40134510 AS inferencing-assets

# Tag: cuda:10.0-cudnn7-devel-ubuntu18.04
# Env: CUDA_VERSION=10.0.130
# Env: CUDA_PKG_VERSION=10-0=10.0.130-1
# Env: NCCL_VERSION=2.4.8
# Env: CUDNN_VERSION=7.6.3.30
# Env: NVIDIA_VISIBLE_DEVICES=all
# Env: NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Env: NVIDIA_REQUIRE_CUDA=cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411
# Label: com.nvidia.cuda.version=10.0.130
# Label: com.nvidia.cudnn.version=7.6.3.30
# Label: com.nvidia.volumes.needed=nvidia_driver
# Ubuntu 18.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04


USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL


# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    redis \
    iproute2 && \
    # rdma-core dependencies
    apt-get install -y \
    udev \
    libudev-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    gcc \
    ninja-build \
    pkg-config \
    valgrind \
    cython3 \
    python3-docutils \
    pandoc \
    python3-dev && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    libbz2-1.0 \
    systemd \
    git \
    wget \
    cpio \
    pciutils \
    libnuma-dev \
    ibutils \
    ibverbs-utils \ 
    rdmacm-utils \
    infiniband-diags \
    perftest \
    librdmacm-dev \
    libibverbs-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libssl1.1 \
    libglib2.0-0 \
    dh-make \
    libnettle6 \
    libx11-dev \
    nginx \
    libgl1-mesa-dev \
    libglib2.0-dev \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install lib for video
# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
# RUN apt update && apt-get install -y libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libswresample-dev
# RUN apt-get install -y ffmpeg
# RUN export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Conda Environment
ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Open-MPI-UCX installation
RUN mkdir /tmp/ucx && \
    cd /tmp/ucx && \
        wget -q https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-1.9.0.tar.gz && \
        tar zxf ucx-1.9.0.tar.gz && \
	cd ucx-1.9.0 && \
        ./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && \
        make -j $(nproc --all) && \
        make install && \
        rm -rf /tmp/ucx


# Open-MPI installation
ENV OPENMPI_VERSION 4.1.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-ucx=/usr/local/ --enable-mca-no-build=btl-uct --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi	
	
# Msodbcsql17 installation
RUN apt-get update && \
    apt-get install -y curl && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17

#Cmake Installation
RUN apt-get update && \
    apt-get install -y cmake

# rdma-core v30.0 for Mlnx_ofed_5_1_2 as user space driver
RUN mkdir /tmp/rdma-core && \
    cd /tmp/rdma-core && \
    git clone --branch v30.0 https://github.com/linux-rdma/rdma-core && \
    cd /tmp/rdma-core/rdma-core && \
    debian/rules binary && \
    dpkg -i ../*.deb && \
    rm -rf /tmp/rdma-core

#Install v3 version of nccl-rdma-sharp-plugins
RUN cd /tmp && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    apt install -y dh-make zlib1g-dev && \
    git clone -b v2.0.0 https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda --without-ucx && \
    make && \
    make install
    
RUN conda install -c r -y conda python=3.7
RUN conda install -y pyyaml scipy ipython scikit-learn matplotlib pandas setuptools Cython h5py graphviz libgcc cmake cffi typing cython pip=20.1.1
RUN conda clean -ya
RUN pip install boto3 addict tqdm regex pyyaml opencv-python opencv-contrib-python nltk spacy future tensorboard filelock tokenizers sentencepiece yapf attrs azureml-core==1.30.0 pillow lmdb imageio scikit-image mmcv-full==1.3.0
# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install pytorch
RUN conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# ENV MKL_SERVICE_FORCE_INTEL=1
#Install Faiss
#RUN conda install faiss-gpu -c pytorch # For CUDA10.1
#RUN pip uninstall -y pillow && CC="cc -mavx2" pip install --force-reinstall pillow-simd && \
    #pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

# Install horovod
# RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.16.1

#Install apex
# RUN pip uninstall -y apex || :
# RUN cd /tmp && \
#     SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
# RUN cd /tmp/apex/ && \
#     python setup.py install --cuda_ext --cpp_ext && \
#     rm -rf /tmp/apex*