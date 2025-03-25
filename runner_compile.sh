#!/bin/bash
# MLIR frontend
POLYGEIST_PATH="$COMPIGRA_PROJECT/Polygeist" 
MLIR_OPT="$POLYGEIST_PATH/llvm-project/build/bin/mlir-opt"
COMPIGRA_OPT="$POLYGEIST_PATH/../build/bin/compigra-opt"
BENCH_BASE="$POLYGEIST_PATH/../benchmarks"
MS_PLUGIN="YOUR MS PLUGIN PATH"

front_end() {
    local bench_name="$1"
    shift
    local bench_path="$BENCH_BASE/$bench_name"
    local bench_c="$bench_path/$bench_name.c"

    # Check if the benchmark file exists
    if [ ! -f "$bench_c" ]; then
        echo "Error: $bench_c not found."
        exit 1
    fi

    # Remove the old IR files
    rm -rf "$bench_path/IR_opt"
    mkdir -p "$bench_path/IR_opt"
    local f_affine="$bench_path/IR_opt/affine.mlir"
    local f_scf="$bench_path/IR_opt/scf.mlir"
    local f_cf="$bench_path/IR_opt/cf.mlir"

    # C -> Affine
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    $POLYGEIST_PATH/build/bin/cgeist "$bench_c" -I "$include" \
        -function="$bench_name" -S --O3 --memref-fullrank --raise-scf-to-affine \
        > "$f_affine"
    
    # Lower down to cf 
    $POLYGEIST_PATH/build/bin/polygeist-opt \
        --affine-loop-fusion --affine-super-vectorize \
        --lower-affine  "$f_affine" > "$f_scf"

    $MLIR_OPT --allow-unregistered-dialect \
        --scf-for-loop-canonicalization --convert-scf-to-cf --canonicalize "$f_scf" > "$f_cf"

}

optimizer(){
    return 0
}

mapper() {
    local bench_name="$1"
    local row_size="$2" 
    local col_size="$3"
    local use_scheduler="$4"
    shift 4
    local params=("${@}")
    local bench_path="$BENCH_BASE/$bench_name"

    local f_cf="$bench_path/IR_opt/cf.mlir"
    local f_cgra="$bench_path/IR_opt/cgra.mlir"
    local f_cgra_i32="$bench_path/IR_opt/cgra_i32.mlir"
    local f_asm="$bench_path/IR_opt/asm.mlir"

    rm -f "$bench_path/IR_opt/"f_cgra
    rm -f "$bench_path/IR_opt/"f_cgra_i32
    rm -f "$bench_path/IR_opt/"f_asm

    # Convert cf to cgra
    if [ ${#params[@]} -eq 0 ]; then
        $COMPIGRA_OPT --allow-unregistered-dialect --convert-cf-to-cgra \
            "$f_cf" > "$f_cgra"
    else
        start_addr=$(( $(IFS=+; echo "${params[*]}") + 128 ))
        $COMPIGRA_OPT --allow-unregistered-dialect --convert-cf-to-cgra="start-addr=$start_addr" \
            "$f_cf" > "$f_cgra"
    fi
    if [ $? -eq 0 ]; then
        echo "Convert to CGRA success."
    else
        echo "Convert to CGRA failed."
        exit 1
    fi

    # Fix the index width
    $COMPIGRA_OPT --allow-unregistered-dialect --fix-index-width --fit-openedge \
        "$f_cgra" > "$f_cgra_i32"
        
    if [ $? -eq 0 ]; then
        echo "Bitwidth To I32 success."
    else
        echo "Fix bit width failed."
        exit 1
    fi

    # create modulo schedule folder
    mkdir -p "$bench_path/IR_opt/satmapit"
    local msOpt=' '
    if [ "$use_scheduler" -eq 1 ]; then
        msOpt="python3 $MS_PLUGIN"
    fi
    if [ ${#params[@]} -eq 0 ]; then
        if [ "$use_scheduler" -eq 1 ]; then
            $COMPIGRA_OPT --allow-unregistered-dialect \
                --gen-temporal-cgra-asm="row=$row_size col=$col_size  ms-opt='$msOpt'
                asm-out=$bench_path/out_$row_size" \
                "$f_cgra_i32" > "$f_asm"
        else 
            $COMPIGRA_OPT --allow-unregistered-dialect \
                --gen-temporal-cgra-asm="row=$row_size col=$col_size ms-opt='$msOpt'
                asm-out=$bench_path/out_$row_size" \
                "$f_cgra_i32" > "$f_asm"
        fi
    else
        start_addr=$(IFS=,; echo "${params[*]}")
        $COMPIGRA_OPT --allow-unregistered-dialect \
            --gen-temporal-cgra-asm="row=$row_size col=$col_size mem=$start_addr  ms-opt='$msOpt'
            asm-out=$bench_path/out_$row_size" \
            "$f_cgra_i32" > "$f_asm"
    fi

    # Check if the compilation was successful
    if [ $? -eq 0 ]; then
        echo "Compilation successful."
    else
        echo "Compilation failed."
    fi
    return 0
}

compile() {
    local execute_frontend="$1"
    shift
    local start_time=$(date +%s%3N)  # Start time in milliseconds

    if [ "$execute_frontend" == "yes" ]; then
        front_end "$@"
    fi
    mapper "$@"

    local end_time=$(date +%s%3N)  # End time in milliseconds
    local elapsed_time=$((end_time - start_time))  # Convert to milliseconds

    echo "Execution time: $((elapsed_time / 1000)).$((elapsed_time % 1000)) seconds"
}

benchmark=$1
front_end=$2
row_size=$3
col_size=$4
shift 4
# The remaining arguments are our integer parameters
params=("$@")

if [ ! -d "$BENCH_BASE/$benchmark/IR_opt" ]; then
    mkdir "$BENCH_BASE/$benchmark/IR_opt"
fi

compile $front_end $benchmark $row_size $col_size ${params[@]}
