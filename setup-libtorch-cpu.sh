#!/bin/bash

sudo rm -rf /opt/libtorch
wget -O /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
unzip /tmp/libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip -d /opt
sudo echo LIBTORCH=/opt/libtorch >> $HOME/.bashrc
sudo echo LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH >> $HOME/.bashrc
