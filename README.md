# Compigra
Compigra produces ASM target at various CGRA architectures.

## Build the project
#### 1. Clone the project
```bash
$ git clone --recurse-submodules https://github.com/yuxwang99/compigra.git
# or through ssh
$ git clone --recurse-submodules git@github.com:yuxwang99/compigra.git
```
#### 2. Refer to [Polygeist](https://github.com/llvm/Polygeist) to build dependencies including LLVM, MLIR, and Polygeist.
#### 3. Build compigra
```bash
$ mkdir build && and cd build
$ cmake ../
$ ninja
```

## Use Compigra
```bash
$ benchmark=`benchmark_name`
$ ./run_Compigra.sh $benchmark
# or for example, directly called the benchmark name
$ ./run_Compigra.sh GSM
```
