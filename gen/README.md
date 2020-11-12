# Generate APIs

## Get `Declaration.yaml` file

```bash
# master branch can be replaced with release version. E.g., v1.7.0
git clone -b master --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir pytorch-build
cd pytorch-build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
cmake --build . --target install
```

The `Declaration.yaml` file is artifact of building Libtorch from source. After running step 4 (can take a while - couple of hours)

```
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
```

.yaml file can be found in either: `pytorch-install/share/ATEN/Declarations.yaml` or `pytorch-build/aten/src/ATen/Declarations.yaml`


Ref. 
1. https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst
2. https://discuss.pytorch.org/t/compile-libtorch-c-api-from-source/81624
3. https://github.com/pytorch/pytorch/issues/12562


## Generate APIs

run from root folder (`gotch`): 

```
dune exec gen/gen.exe
```
