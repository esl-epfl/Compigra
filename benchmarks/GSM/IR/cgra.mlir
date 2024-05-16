module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @GSM(%arg0: i32 {llvm.noundef}, %arg1: i32, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: none {cgra.None}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2", "in3"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(512 : i32) {arg2 = "base", stage = "init"} : i32
    %1 = llvm.mlir.constant(0 : i32) {arg0 = "int", stage = "init"} : i32
    %2 = lwi %1 {stage = "init"} : i32
    %3 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %4 = llvm.mlir.constant(-32768 : i32) {stage = "init"} : i32
    %5 = llvm.mlir.constant(32767 : i32) {stage = "init"} : i32
    %6 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %7 = llvm.mlir.constant(40 : i32) {stage = "init"} : i32
    %8 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %9 = merge %21, %3 {stage = "loop"} : i32
    %10 = merge %20, %2 {stage = "loop"} : i32
    %11 = llvm.mul %8, %9 {stage = "loop"} : i32
    %12 = llvm.add %0, %11 {stage = "loop"} : i32
    %13 = lwi %12 {stage = "loop"} : i32
    %14 = llvm.sub %13, %3 {stage = "loop"} : i32
    %15 = llvm.sub %13, %4 {stage = "loop"} : i32
    %16 = llvm.sub %3, %13 {stage = "loop"} : i32
    %17 = bsfa %15 [%5, %16]  {stage = "loop"} : i32
    %18 = bsfa %14 [%17, %13]  {stage = "loop"} : i32
    %19 = llvm.sub %10, %18 {stage = "loop"} : i32
    %20 = bsfa %19 [%18, %10]  {stage = "loop"} : i32
    %21 = llvm.add %9, %6 {stage = "loop"} : i32
    %22 = beq [%21, %7] %9 {stage = "loop"} : i32
    llvm.return {stage = "fini"} %20 : i32
  }
  cgra.func @main(%arg0: none {cgra.None}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

