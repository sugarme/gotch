#!/bin/bash

LIBTORCH_VERSION="${LIBTORCH_VER:-1.7.0}"
CUDA_VERSION="${CUDA_VER:-10.1}"

if [ $CUDA_VERSION=="cpu" ]
then
  CU_VERSION="cpu"
else
  CU_VERSION="cu${CUDA_VERSION//./}"
fi

# Libtorch paths
GOTCH_LIBTORCH="/usr/local/lib/libtorch"
LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"

if [ $CUDA_VERSION=="cpu" ]
then
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
wget  -q --show-progress --progress=bar:force:noscroll  -O "/tmp/$LIBTORCH_ZIP" "$LIBTORCH_URL"
sudo rm -rf $GOTCH_LIBTORCH # delete old libtorch if exisitng
sudo unzip "/tmp/$LIBTORCH_ZIP" -d $GOTCH_LIBTORCH
rm "/tmp/$LIBTORCH_ZIP"

sudo ldconfig

# Update .bashrc
#===============
FILE="$HOME/.bashrc"
LN_GOTCH_LIBTORCH="export GOTCH_LIBTORCH=$GOTCH_LIBTORCH"
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


