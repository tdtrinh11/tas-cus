#!/bin/bash
## This gist contains instructions about cuda v11.2 and cudnn 8.1 installation in Ubuntu 18.04 for PyTorch

#############################################################################################
##### forked by : https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73 ########
#############################################################################################


### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### If you have previous installation remove it first. 
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*


### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# system update
sudo apt-get update
sudo apt-get upgrade


# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update

 # installing CUDA-11.2
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-2 cuda-drivers


# setup your paths
echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v8.1

CUDNN_TAR_FILE="cudnn-11.2-linux-ppc64le-v8.1.1.33.tgz"
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-ppc64le-v8.1.1.33.tgz
tar -xzvf ${CUDNN_TAR_FILE}


# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.2/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64/
sudo chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

# install PyTorch (an open source machine learning framework)
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html