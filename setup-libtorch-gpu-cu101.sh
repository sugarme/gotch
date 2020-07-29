#!/bin/bash

sudo rm -rf /opt/libtorch
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip
unzip /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip -d /opt
sudo echo LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/lib64-nvidia:/usr/local/cuda-10.1/lib64 >> $HOME/.bashrc
