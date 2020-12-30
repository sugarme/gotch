#!/bin/bash

GOTCH_VERSION="${GOTCH_VER:-v0.3.5}"
LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"
CUDA_VERSION="${CUDA_VER:-10.1}"

if [ $CUDA_VERSION == "cpu" ]
then
  CU_VERSION="cpu"
else
  CU_VERSION="cu${CUDA_VERSION//./}"
fi

# Libtorch paths
GOTCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"
LIBTORCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION/libtch/libtorch"
LIBRARY_PATH="$LIBTORCH/lib"
# CPATH="$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"
CPATH="$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"

if [ $CUDA_VERSION == "cpu" ]
then
  # LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib"
  LD_LIBRARY_PATH="$LIBTORCH/lib"
else
  # LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
  LD_LIBRARY_PATH="$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
fi

# Update current shell environment variables for newly installed Libtorch
export LIBRARY_PATH=$LIBRARY_PATH
export CPATH=$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Install gotch
#==============
cwd=$(pwd)
mkdir /tmp/gotch-test
cd /tmp/gotch-test
go mod init "github.com/sugarme/gotch-test"
go get "github.com/sugarme/gotch@$GOTCH_VERSION" 
rm -rf /tmp/gotch-test
cd $cwd

if [ $CUDA_VERSION == "cpu" ]
then
  # prepare C lib for CPU version
  sudo rm $GOTCH/libtch/lib.go
  sudo cp $GOTCH/libtch/lib.go.cpu $GOTCH/libtch/lib.go
  sudo mv $GOTCH/libtch/dummy_cuda_dependency.cpp $GOTCH/libtch/dummy_cuda_dependency.cpp.gpu
  sudo mv $GOTCH/libtch/fake_cuda_dependency.cpp.cpu $GOTCH/libtch/fake_cuda_dependency.cpp
fi

# Install Libtorch
#=================
LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${CU_VERSION}.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/${CU_VERSION}/${LIBTORCH_ZIP}"
wget -O "/tmp/$LIBTORCH_ZIP" "$LIBTORCH_URL"
# delete old libtorch if existing
sudo rm -rf $LIBTORCH
sudo unzip "/tmp/$LIBTORCH_ZIP" -d $GOTCH/libtch
rm "/tmp/$LIBTORCH_ZIP"

# Update .bashrc
#===============
FILE="$HOME/.bashrc"
LN_GOTCH_LIBTORCH="export GOTCH_LIBTORCH=$LIBTORCH"
LN_LIBRARY_PATH="export LIBRARY_PATH=$LIBRARY_PATH"
LN_CPATH="export CPATH=$CPATH"
LN_LD_LIBRARY_PATH="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# replace line if matching pattern otherwise, insert a new line to the bottom.
# -qF quiet, plain text
grep -qF 'export GOTCH_LIBTORCH' "$FILE" && sed -i 's|^export GOTCH_LIBTORCH.*|'"$LN_GOTCH_LIBTORCH"'|g' "$FILE" || echo "$LN_GOTCH_LIBTORCH" >> "$FILE"
grep -qF 'export LIBRARY_PATH' "$FILE" && sed -i 's|^export LIBRARY_PATH.*|'"$LN_LIBRARY_PATH"'|g' "$FILE" || echo "$LN_LIBRARY_PATH" >> "$FILE"
grep -qF 'export CPATH' "$FILE" && sed -i 's|^export CPATH.*|'"$LN_CPATH"'|g' "$FILE" || echo "$LN_CPATH" >> "$FILE"
grep -qF 'export LD_LIBRARY_PATH' "$FILE" && sed -i 's|^export LD_LIBRARY_PATH.*|'"$LN_LD_LIBRARY_PATH"'|g' "$FILE" || echo "$LN_LD_LIBRARY_PATH" >> "$FILE"

# refresh environment for all next opening shells.
exec "$BASH"


