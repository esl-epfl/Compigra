#!/bin/bash
POLYGEIST_PATH="/home/yuxuan/Projects/24S/Compigra/Polygeist"
MLIR_OPT="$POLYGEIST_PATH/llvm-project/build/bin/mlir-opt"
BENCH_BASE="/home/yuxuan/Projects/24S/Compigra/benchmarks"

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <benchmark_name>"
    exit 1
fi

# Store the benchmark name provided as the first argument

compile() {
    local bench_name="$1"
    local bench_path="$BENCH_BASE/$bench_name"
    local bench_c="$bench_path/$bench_name.c"

    # Check if the benchmark file exists
    if [ ! -f "$bench_c" ]; then
        echo "Error:$bench_c not found."
        exit 1
    fi

    local f_scf="$bench_path/IR/scf.mlir"
    local f_cf="$bench_path/IR/cf.mlir"
    local f_llvm="$bench_path/IR/llvm.mlir"

    # Compile the benchmark using cgeist
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    $POLYGEIST_PATH/build/bin/cgeist "$bench_c" -I "$include" \
        -function="$bench_name" -S -O3 > "$f_scf"
    
    $MLIR_OPT --convert-scf-to-cf "$f_scf" > "$f_cf"
    $MLIR_OPT --test-lower-to-llvm "$f_cf" > "$f_llvm"

    # Check if the compilation was successful
    if [ $? -eq 0 ]; then
        echo "Compilation successful."
    else
        echo "Compilation failed."
    fi
    return 0
}


benchmark="$1"
if [ ! -d "$BENCH_BASE/$benchmark/IR" ]; then
    mkdir "$BENCH_BASE/$benchmark/IR"
fi

compile $benchmark
