# FROM mcr.microsoft.com/azureml/base-gpu:latest 
FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

# # Install basic dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         --allow-change-held-packages \
#         build-essential \
#         autotools-dev \
#         rsync \
#         curl \
#         cmake \
#         wget \
#         vim \
#         tmux \
#         htop \
#         git \
#         unzip \
#         libnccl2 \
#         libnccl-dev \
#         ca-certificates \
#         libjpeg-dev \
#         htop \ 
#         sudo \
#         g++ \
#         gcc \
#         apt-utils \
#         libosmesa6-dev \
#         net-tools

# RUN export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
# # Set timezone
# RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# # very important!!!!!!!
# RUN ln -s /opt/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "export PATH=/opt/miniconda/bin:$PATH" >> ~/.bashrc
# # very important!!!!!!!
# ENV PATH /opt/miniconda/bin:$PATH
# RUN conda update -n base conda

# Install general libraries
RUN conda install -y python=3.6 numpy scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install -y mkl-include cmake cffi typing cython
RUN conda install -y -c mingfeima mkldnn
RUN pip install boto3 addict tqdm regex pyyaml opencv-python torchsummary azureml_core azureml-sdk albumentations pretrainedmodels efficientnet_pytorch scikit-image==0.15  yacs git+https://github.com/qiuzhongwei-USTB/ResNeSt.git tensorboard pydicom
RUN pip install --upgrade pipi

# Install pytorch
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN conda install -y -c conda-forge pillow=6.2.1

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"


# RUN echo "aml_rsna_docker dockerfile finished !"
