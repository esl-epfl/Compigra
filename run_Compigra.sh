#!/bin/bash
# MLIR frontend
POLYGEIST_PATH="/home/yuxuan/Projects/24S/Compigra/Polygeist" # optional: set the path to the Polygeist repository
CLANG14="/home/yuxuan/Projects/24S/SAT-MapIt/llvm-project/build/bin/clang"
MLIR_OPT="$POLYGEIST_PATH/llvm-project/build/bin/mlir-opt"
MLIR_TRANSLATE="$POLYGEIST_PATH/llvm-project/build/bin/mlir-translate"
COMPIGRA_OPT="$POLYGEIST_PATH/../build/bin/compigra-opt"
BENCH_BASE="$POLYGEIST_PATH/../benchmarks"
# Plugin for modulo scheduling
MS_PLUGIN="/home/yuxuan/Projects/24S/SAT-MapIt/Mapper/main.py"

# compile use C->LLVM->MLIR(LLVM)->MLIR(CGRA)
compile_sat() {
    local bench_name="$1"
    local config="$2x$2"
    local mode=$3
    local bench_path="$BENCH_BASE/$bench_name"
    local bench_c="$bench_path/$bench_name.c"
    local bench_ll="$bench_path/$bench_name.ll"

    echo $bench_path    
 

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
      "$bench_c" -o "$f_ll" 2> /dev/null

    # Check whether conversion success
    if [ $? -ne 0 ]; then
        echo "FAILED ON LLVM IR GENERATION."
        return 1
    else
        echo "Generate LLVM IR."
    fi

    # Translate llvm to mlir llvm dialect
    $MLIR_TRANSLATE --import-llvm  "$f_ll" > "$f_llvm" 2> /dev/null
       
    # Check whether conversion success
    if [ $? -ne 0 ]; then
        echo "FAILED ON LLVM IR GENERATION."
        return 1
    else
        echo "Translate to MLIR with LLVM Dialect."
    fi

    # convert llvm to cgra operation
    $COMPIGRA_OPT --allow-unregistered-dialect \
      --convert-llvm-to-cgra="func-name=$bench_name mem-json=${BENCH_BASE}/memory_config.json" \
     "$f_llvm" > "$f_cgra" 2> /dev/null

    # Check whether conversion success
    if [ $? -ne 0 ]; then
        echo "FAILED ON CONVERTING LLVM TO CGRA DIALECT."
        return 1
    else
        echo "Conversion success."
    fi

    #  convert cgra operations to fit into hardware ISA
    $COMPIGRA_OPT --allow-unregistered-dialect \
      --fit-openedge="out-dag=$f_dag" "$f_cgra" > "$f_hardware" 2> /dev/null

    # Check whether hardware transformation success
    if [ $? -ne 0 ]; then
        echo "FAILED ON HARDWARE TRANSFORMATION."
        return 1
    else
        echo "Hardware transformation success."
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

    # Generate the scheduled assembly code
    if [ "$mode" -eq 0 ]; then
        echo "!Warning: Not using SAT-MapIt results."
        $COMPIGRA_OPT --allow-unregistered-dialect \
            --gen-openedge-asm="func-name=$bench_name \
            grid=$2" "$f_hardware" > $f_sat #2> /dev/null

        if [ $? -ne 0 ]; then
            echo "Kernel scheduling failed."
            return 1
        else
            echo "Kernel schedule success."
        fi

        # mv out.sat and out_grid.sat to $bench_path/$config
        mv out.sat $bench_path/$config/NoOptim_out.sat
        mv out_grid.sat $bench_path/$config/NoOptim_out_grid.sat
    else
        # Run SAT-MapIt to schedule the loop block
        python3 $MS_PLUGIN --path $sat_text --bench $bench_name --unit $2 --seed 13\
            > $bench_path/$config/"out_raw.sat" 2> /dev/null

        # Check whether loop block scheduling success
        if [ $? -ne 0 ]; then
            echo "FAILED ON LOOP BLOCK SCHEDULING."
            return 1
        else
            echo "Loop block scheduling success."
        fi

        $COMPIGRA_OPT --allow-unregistered-dialect \
            --gen-openedge-asm="func-name=$bench_name \
            map-result=$bench_path/$config/out_raw.sat grid=$2" \
            "$f_hardware" > $f_sat 2> /dev/null
        if [ $? -ne 0 ]; then
            echo "Kernel scheduling failed."
            return 1
        else
            echo "Kernel schedule success."
        fi
    fi

    echo "COMPILATION SUCCESS."
    return 0
}


benchmark=$1
config=$2
task=$3

# Check if the second parameter is provided
if [ -z "$2" ]; then
    config=4
else
    config=$2
fi

# Check if the third parameter is provided
if [ -z "$3" ]; then
    task=1
else
    if [ "$3" -eq 0 ]; then
        task=0
    else
        echo "Error: Invalid task value. \
        Task must be 0, specifying not use sat-mapit results"
        exit 1
    fi
fi


if [ ! -d "$BENCH_BASE/$benchmark/IR" ]; then
    mkdir "$BENCH_BASE/$benchmark/IR"
fi

compile_sat $benchmark $config $task




