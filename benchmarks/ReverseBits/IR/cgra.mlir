module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @ReverseBits(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(4 : i32) {arg1 = "int", stage = "init"} : i32
    %1 = lwi %0 {stage = "init"} : i32
    %2 = llvm.mlir.constant(0 : i32) {arg0 = "int", stage = "init"} : i32
    %3 = lwi %2 {stage = "init"} : i32
    %4 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %5 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %6 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %7 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %8 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %9 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %10 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %11 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %12 = beq [%1, %7] %22 {stage = "init"} : i32
    %13 = merge %18, %6 {stage = "loop"} : i32
    %14 = merge %20, %5 {stage = "loop"} : i32
    %15 = merge %19, %3 {stage = "loop"} : i32
    %16 = llvm.shl %13, %10 {stage = "loop"} : i32
    %17 = llvm.and %15, %9  {stage = "loop"} : i32
    %18 = llvm.or %16, %17  {stage = "loop"} : i32
    %19 = llvm.lshr %15, %8  {stage = "loop"} : i32
    %20 = llvm.add %14, %11 {stage = "loop"} : i32
    %21 = beq [%20, %1] %13 {stage = "loop"} : i32
    %22 = merge %18, %4 {stage = "fini"} : i32
    llvm.return {stage = "fini"} %22 : i32
  }
  cgra.func @main(...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = [], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

