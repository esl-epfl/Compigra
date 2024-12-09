# Compigra
Compigra is an open-source compilation tool to produce assembly code from C/C++ code targets at CGRA based on MLIR.
Compigra provides customized dialects to abstract operations on IR level in front end.
In the back end, an ILP model is supported to compute mapping and output assembly code.

## Prerequisites
Please refer to [LLVM-Getting Started](https://llvm.org/docs/GettingStarted.html#requirements) to see the requirements on the hardware and software.

## Build the project
#### 1.(1) [Recommended] Clone the full project and use all features in Compigra.

You can full clone the project with `--recurse-submodules`. Please refer to [Polygeist](https://github.com/llvm/Polygeist) to build dependencies including LLVM, MLIR, and Polygeist(Rough disk space ~130G).
```bash
$ git clone --recurse-submodules https://github.com/yuxwang99/compigra.git
```
#### 1.(2) Use compilation front-end in Clang and MLIR. 
Polygeist is not compulsory for the back end assembly code generation. You can also refer to [LLVM](https://llvm.org/docs/GettingStarted.html) to build LLVM, MLIR, and clang for front end support(Rough disk space ~30G). It is **highly recommended** to use LLVM version in commit hash ID(26eb428) same with the Polygeist version to ensure MLIR consistency.
```bash
$ git clone https://github.com/yuxwang99/compigra.git
```
#### 2. Install gurobi as ILP solver
We use [gurobi](https://www.gurobi.com/) as our solver for the ILP model. Gurobi is compulsory for back end assembly code generation, but not for building the project. 
You can still use the front end optimization passes to generate hardware compatible IR even without gurobi enabled.
You can apply [gurobi acedemic license](https://www.gurobi.com/academia/academic-program-and-licenses/) and configure it if you work in acedemia.

#### 3. Build compigra
If you use 1.(1) to clone the project, run the following command to build the project and specify GUROBI_INSTALL_DIR if gurobi is installed in your workstation. 
```bash
$ mkdir build && and cd build
$ cmake -DGUROBI_INSTALL_DIR=/path/to/gurobi ../
$ ninja
```
Otherwise, set *MLIR_DIR*(Line50) and *LLVM_BUILD_MAIN_SRC_DIR*(Line58) in the CMakeLists.txt to point to your customized MLIR location.

## Use Compigra
All the passes including front end optimization and back end assembly generation is integrated in our executable **compigra-opt**, for more helpful information run:
```bash
./bin/compigra-opt -h
```
<span style="font-size:1.5em;">[[ARC25 submission](./ARC25.md)]</span> 

You could refer [ARC25](./ARC25.md) for detailed explanations on the passes we use in the end-to-end compilation flow for the ARC25 submission.

## Simulate your result
### Software simulation
Use [CGRA-simulator](https://github.com/esl-epfl/ESL-CGRA-simulator) for software simulation to get CGRA computation results using assembly.
### RTL simulation
See [X-Heep](https://github.com/esl-epfl/x-heep) and [OpenEdgeCGRA](https://github.com/esl-epfl/OpenEdgeCGRA/tree/main) for more help.