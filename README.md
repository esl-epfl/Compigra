# Compigra
Compigra is an open-source compilation tool to produce assembly code from C/C++ code targets at CGRA based on MLIR.
Compigra provides customized dialects to abstract operations on IR level in front end.
In the back end, an ILP model is supported to compute mapping and output assembly code.

## Prerequisites
Please refer to [LLVM-Getting Started](https://llvm.org/docs/GettingStarted.html#requirements) to see the requirements on the hardware and software.

## Build the project
#### 1. Clone the full project and use all features in Compigra.

You can full clone the project with `--recurse-submodules`. Please refer to [Polygeist](https://github.com/llvm/Polygeist) to build dependencies including LLVM, MLIR, and Polygeist(Rough disk space ~130G).
<!-- ```bash
$ git clone --recurse-submodules https://github.com/yuxwang99/compigra.git
``` -->

#### 2. Install gurobi as ILP solver
We use [gurobi](https://www.gurobi.com/) as our solver for the ILP model. Gurobi is compulsory for back end assembly code generation, but not for building the project. 
You can still use the front end optimization passes to generate hardware compatible IR even without gurobi enabled.
You can apply [gurobi acedemic license](https://www.gurobi.com/academia/academic-program-and-licenses/) and configure it if you work in acedemia.

#### 3. Build compigra
Run the following command to build the project and specify GUROBI_INSTALL_DIR if gurobi is installed in your workstation. 
```bash
$ mkdir build && and cd build
$ cmake -DGUROBI_INSTALL_DIR=/path/to/gurobi ../
$ ninja
```

## Use Compigra
All the passes including front end optimization and back end assembly generation is integrated in our executable **compigra-opt**, for more helpful information run:
```bash
./bin/compigra-opt -h
```

For detailed instructions on compiling your C code using Compigra, refer to the [`runner_compile.sh`](./runner_compile.sh) script. This script integrates front-end parsing, user-defined middle-end optimizations, back-end mapping, and optimizations using modulo scheduling.


## Simulate your result
### Software simulation
Use [CGRA-simulator](https://github.com/esl-epfl/ESL-CGRA-simulator) for software simulation to get CGRA computation results using assembly.
### RTL simulation
See [X-Heep](https://github.com/esl-epfl/x-heep) and [OpenEdgeCGRA](https://github.com/esl-epfl/OpenEdgeCGRA/tree/main) for more help.