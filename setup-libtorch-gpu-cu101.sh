#!/bin/bash

sudo rm -rf /opt/libtorch
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
unzip /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip -d /opt
sudo echo LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/lib64-nvidia:/usr/local/cuda-10.1/lib64 >> $HOME/.bashrc
