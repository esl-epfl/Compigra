#!/bin/bash
# MLIR frontend
CLANG14="/home/yuxuan/Projects/24S/SAT-MapIt/llvm-project/build/bin/clang"
POLYGEIST_PATH="/home/yuxuan/Projects/24S/Compigra/Polygeist"
MLIR_OPT="$POLYGEIST_PATH/llvm-project/build/bin/mlir-opt"
MLIR_TRANSLATE="$POLYGEIST_PATH/llvm-project/build/bin/mlir-translate"
COMPIGRA_OPT="$POLYGEIST_PATH/../build/bin/compigra-opt"
BENCH_BASE="/home/yuxuan/Projects/24S/Compigra/benchmarks"
# backend script to generate bitstream
ASM_GEN="/home/yuxuan/Projects/24S/SAT-MapIt/Mapper/main.py"
CONVERTER="/home/yuxuan/Projects/24S/cgra/ESL-CGRA-simulator/src/sat_to_csv.py"
# Bitstream generator
EXPORTER="/home/yuxuan/Projects/24S/cgra/ESL-CGRA-simulator/src/exporter.py"
BIN_GEN_DIR="/home/yuxuan/Projects/24S/HEEPsilon/sw/applications/kernel_test/utils/"


compile() {
    local bench_name="$1"
    local bench_path="$BENCH_BASE/$bench_name"
    local bench_c="$bench_path/$bench_name.c"

    # Check if the benchmark file exists
    if [ ! -f "$bench_c" ]; then
        echo "Error:$bench_c not found."
        exit 1
    fi

    # remove the old IR files
    rm -f "$bench_path/IR/"*

    local f_scf="$bench_path/IR/scf.mlir"
    local f_cf="$bench_path/IR/cf.mlir"
    local f_cf_opt="$bench_path/IR/cf_opt.mlir"
    local f_std="$bench_path/IR/std.mlir"
    local f_llvm="$bench_path/IR/llvm.mlir"

    # Compile the benchmark using cgeist
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    $POLYGEIST_PATH/build/bin/cgeist "$bench_c" -I "$include" \
         -function="$bench_name" -S --O3 --memref-fullrank  --canonicalizeiters=10 > "$f_scf"
    
    # lower down to cf 
    $MLIR_OPT --scf-for-loop-canonicalization --canonicalize --convert-scf-to-cf   "$f_scf" > "$f_cf"

    # # rewrite the cf
    $COMPIGRA_OPT --allow-unregistered-dialect --fix-index-width \
        --reduce-branches "$f_cf" > "$f_cf_opt"

    # # lower down to standard dialect
    # # $MLIR_OPT --convert-complex-to-standard "$f_cf_opt" > "$f_std"
    # $MLIR_OPT --test-lower-to-llvm "$f_cf" > "$f_llvm"

    # Check if the compilation was successful
    if [ $? -eq 0 ]; then
        echo "Compilation successful."
    else
        echo "Compilation failed."
    fi
    return 0
}

# compile use C->LLVM->MLIR(LLVM)->MLIR(CGRA)
compile_sat() {
    local bench_name="$1"
    local config="$2x$2"
    local bench_path="$BENCH_BASE/$bench_name"
    local bench_c="$bench_path/$bench_name.c"
    local bench_ll="$bench_path/$bench_name.ll"
 

    # Check if the benchmark file exists
    if [ ! -f "$bench_c" ]; then
        echo "Error:$bench_c not found."
        exit 1
    fi

    # remove the old IR files
    rm -rf "$bench_path"/IR/*

    # Create the directory to store the DAG text files
    if [ ! -d "$BENCH_BASE/$benchmark/IR/SatMapDAG" ]; then
        mkdir "$BENCH_BASE/$benchmark/IR/SatMapDAG"
    fi
    # remove the old DAG text files
    rm -rf "$bench_path/IR/SatMapDAG"/$bench_name*

    # using clang to compile 32-bit system
    local f_ll="$bench_path/IR/llvm.ll"
    local f_llvm="$bench_path/IR/llvm.mlir"
    local f_cgra="$bench_path/IR/cgra.mlir"
    local f_dag="$bench_path/IR/SatMapDAG/$bench_name"
    local f_hardware="$bench_path/IR/hardware.mlir"
    local f_sat="$bench_path/IR/sat.mlir"
    
    # Get llvm IR from clang
    $CLANG14 -S -c -emit-llvm -m32 -O3 \
      -fno-unroll-loops -fno-vectorize -fno-slp-vectorize \
      "$bench_c" -o "$f_ll"
    # Translate llvm to mlir llvm dialect
    $MLIR_TRANSLATE --import-llvm  "$f_ll" > "$f_llvm"

    # convert llvm to cgra operation
    $COMPIGRA_OPT --allow-unregistered-dialect \
      --convert-llvm-to-cgra="func-name=$bench_name mem-json=${BENCH_BASE}/../build/bin/memory_config.json" \
     "$f_llvm" > "$f_cgra"

    # Check whether conversion success
    if [ $? -ne 0 ]; then
        echo "FAILED ON CONVERTING LLVM TO CGRA DIALECT."
        return 1
    fi

    #  convert cgra operations to fit into hardware ISA
    $COMPIGRA_OPT --allow-unregistered-dialect \
      --fit-openedge="out-dag=$f_dag" "$f_cgra" > "$f_hardware"

    # Check whether hardware transformation success
    if [ $? -ne 0 ]; then
        echo "FAILED ON HARDWARE TRANSFORMATION."
        return 1
    fi

    # folder to place the DAG text files
    local sat_text="$bench_path/IR/SatMapDAG/"
    if [ ! -d $sat_text ]; then
        mkdir -p $sat_text
    fi

    # if not existed $path/$config, mkdir
    if [ ! -d $bench_path/$config ]; then
        mkdir -p $bench_path/$config
    fi

    # Run SAT-MapIt to schedule the loop block
    python3 $ASM_GEN --path $sat_text --bench $bench_name --unit $2 > $bench_path/$config/"out_raw.sat"

    # Check whether loop block scheduling success
    if [ $? -ne 0 ]; then
        echo "FAILED ON LOOP BLOCK SCHEDULING."
        return 1
    fi

    # Generate the scheduled assembly code
    $COMPIGRA_OPT --allow-unregistered-dialect \
        --gen-openedge-asm="func-name=$bench_name map-result=$bench_path/$config/out_raw.sat" \
        "$f_hardware" > $f_sat

    # Check if the scheduling was successful
    if [ $? -eq 0 ]; then
        echo "Compilation successful."
    else
        echo "Kernel schedule failed."
    fi
    return 0
}

# Generate binary code for OpenEdge
asm2bin(){
    local env=${pwd}
    local path=$1
    local bench=$2
    local config=$3x$3
    local out="out.sat"

    # Export instructions
    python3 $EXPORTER --infile $path/$config/"instructions.csv" --outfile $path/$config/$out


    cd $BIN_GEN_DIR
    python3 $BIN_GEN_DIR/inst_encoder.py $path $config
    cd $env
}


benchmark=$1
config=$2

# Check if the second parameter is provided
if [ -z "$2" ]; then
    config=4
else
    config=$2
fi

task=$3
# use absolute path to avoid any confusion
# $1 /home/yuxuan/Projects/24S/Compigra/benchmarks/BitCount/IR/
# $2 bitcount BitCount
BENCH_BASE="/home/yuxuan/Projects/24S/Compigra/benchmarks/"
ASM_BASE="/home/yuxuan/Projects/24S/Compigra/benchmarks/${benchmark}/IR/SatMapDAG/"

# generate binary code
if [ "$task" == "bin" ]; then
    asm2bin $ASM_BASE $benchmark $config
    exit 0
else
    if [ ! -d "$BENCH_BASE/$benchmark/IR" ]; then
        mkdir "$BENCH_BASE/$benchmark/IR"
    fi

    compile_sat $benchmark $config
fi 



