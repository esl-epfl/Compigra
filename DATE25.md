# DATE25 submission
We aggregate all the compilation passes in *run_compigra.sh*.
Before running the one-line script to generate everything, please use **ABSOLUTE PATH** to POLYGEIST_PATH(in the submodule if you fully clone the project), CLANG14, and MS_PLUGIN(optional) in the start of *run_compigra.sh* to specify executables.

You might need to independently build **clang14** even other versions of clang is supported by your Polygeist infrastructure. We will simplify the dependency issues in the future. 

We use a lite version of [modulo scheduler](https://github.com/CristianTirelli/SAT-MapIt-Lite) that compile DFG in python back end. See the README.md of the repo and set MS_PLUGIN to Mapper/main.py to enable modulo scheduler.

Run compigra as follows: (4 specify the grid size of CGRA, and 0 represent the compilation routes without modulo scheduling, which internally supported inside compigra.)
```bash
$ BENCHMARK=`benchmark_name` 
$ ./run_Compigra.sh $BENCHMARK 4 0
# or for example, directly called the benchmark name
$ ./run_Compigra.sh GSM 4 0
```
If you enabled your modulo scheduler through MS_PLUGIN, the script uses the modulo scheduler result by default which produces more efficient assembly code in faster compilation time for smaller grid size CGRA:
```bash
$ BENCHMARK=`benchmark_name` 
$ ./run_Compigra.sh $BENCHMARK 4
# or for example, directly called the benchmark name
$ ./run_Compigra.sh GSM 4 
```
We specify the address in benchmarks/memory_config.json, where the first data is placed in C800. The address before is reserved space for system configuration. You could change the address setting by directly change the memory configuration file.

### IR generation
This automatically keeps the original C file and clears the $BENCHMARK in the benchmark folder and generates
- IR/llvm.ll -through clang14 using $(BENCHMARK).c.
- IR/llvm.mlir -through mlir-translate processing IR/llvm.ll.
- IR/cgra.mlir -Conversion pass that convert all operations into CGRA ISA using IR/llvm.mlir.
- IR/hardware.mlir -Transformation pass that generates hardware-compatible IR using IR/cgra.mlir.
- IR/sat.mlir - IR reconstruction with modulo scheduler result.

### Assembly generation
The assembly is placed under 4x4 folders generated with modulo scheduler disabled. 
- 4x4/NoOptim_out_grid.sat presents the assembly in 4x4 format 
- 4x4/NoOptim_out.sat presents in sequential 16 instructions.

The assembly is placed under 4x4 folders generated with modulo scheduler enabled. 
- 4x4/out_grid.sat 
- 4x4/out.sat
- 4x4/out_raw.sat This is raw printout of results of modulo scheduler.

Please refer to the benchmarks folder where all outputs are updated. 

