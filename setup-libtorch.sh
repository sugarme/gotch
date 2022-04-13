#!/bin/bash

LIBTORCH_VERSION="${LIBTORCH_VER:-1.11.0}"
CUDA_VERSION="${CUDA_VER:-11.3}"

if [ "${CUDA_VERSION}" == "cpu" ]; then
  CU_VERSION="cpu"
else
  CU_VERSION="cu${CUDA_VERSION//./}"
fi

# Libtorch paths
LIBTORCH_PATH="/usr/local/lib"
GOTCH_LIBTORCH="$LIBRARY_PATH/libtorch"

LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
if [[ "${CUDA_VERSION}" == "cpu" ]]; then
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
else
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
fi

# Update current shell environment variables for newly installed Libtorch
export GOTCH_LIBTORCH=$GOTCH_LIBTORCH
export LIBRARY_PATH=$LIBRARY_PATH
export CPATH=$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Install Libtorch
#=================
LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${CU_VERSION}.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/${CU_VERSION}/${LIBTORCH_ZIP}"
echo $LIBTORCH_URL
sudo wget  -q --show-progress --progress=bar:force:noscroll  -O "/tmp/$LIBTORCH_ZIP" "$LIBTORCH_URL"
sudo rm -rf $GOTCH_LIBTORCH # delete old libtorch if exisitng
sudo unzip "/tmp/$LIBTORCH_ZIP" -d $LIBTORCH_PATH
sudo rm "/tmp/$LIBTORCH_ZIP"

sudo ldconfig

# refresh environment for all next opening shells.
exec "$BASH"
