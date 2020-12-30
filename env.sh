#!/bin/bash

# Catch info from Go exported environment variables
GOTCH_VERSION="${GOTCH_VER:-NotSpecified}"
LIBTORCH="${GOTCH_LIBTORCH:-NotSpecified}" # Libtorch root path
CUDA_VERSION="${GOTCH_CUDA_VERSION:-NotSpecified}" # e.g 10.1; cpu

GOTCH="$GOPATH/pkg/mod/github.com/sugarme/gotch@$GOTCH_VERSION"
LIBRARY_PATH="$LIBTORCH/lib"
# CPATH="$CPATH:$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"
CPATH="$LIBTORCH/lib:$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include"

if [ $CUDA_VERSION == "cpu" ]
then
  # LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib"
  LD_LIBRARY_PATH="$LIBTORCH/lib"

  # prepare C lib for CPU version
  sudo rm $GOTCH/libtch/lib.go
  sudo cp $GOTCH/libtch/lib.go.cpu $GOTCH/libtch/lib.go
  sudo mv $GOTCH/libtch/dummy_cuda_dependency.cpp $GOTCH/libtch/dummy_cuda_dependency.cpp.gpu
  sudo mv $GOTCH/libtch/fake_cuda_dependency.cpp.cpu $GOTCH/libtch/fake_cuda_dependency.cpp
else
  # LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
  LD_LIBRARY_PATH="$LIBTORCH/lib:/usr/lib64-nvidia:/usr/local/cuda-${CUDA_VERSION}/lib64"
fi

#update .bashrc
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

