module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @SHA2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(512 : i32) {arg0 = "base", stage = "init"} : i32
    %1 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %2 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %3 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %4 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %5 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %6 = llvm.mlir.constant(0 : i32) {stage = "init"} : i32
    %7 = llvm.mlir.constant(5 : i32) {stage = "init"} : i32
    %8 = llvm.mlir.constant(27 : i32) {stage = "init"} : i32
    %9 = llvm.mlir.constant(-1 : i32) {stage = "init"} : i32
    %10 = llvm.mlir.constant(1518500249 : i32) {stage = "init"} : i32
    %11 = llvm.mlir.constant(30 : i32) {stage = "init"} : i32
    %12 = llvm.mlir.constant(2 : i32) {stage = "init"} : i32
    %13 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %14 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %15 = llvm.mlir.constant(20 : i32) {stage = "init"} : i32
    %16 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %17 = merge %40, %5 {stage = "loop"} : i32
    %18 = merge %19, %4 {stage = "loop"} : i32
    %19 = merge %20, %3 {stage = "loop"} : i32
    %20 = merge %39, %2 {stage = "loop"} : i32
    %21 = merge %22, %1 {stage = "loop"} : i32
    %22 = merge %36, %6 {stage = "loop"} : i32
    %23 = llvm.shl %22, %7 {stage = "loop"} : i32
    %24 = llvm.ashr %22, %8  {stage = "loop"} : i32
    %25 = llvm.or %23, %24  {stage = "loop"} : i32
    %26 = llvm.and %20, %21  {stage = "loop"} : i32
    %27 = llvm.xor %21, %9  {stage = "loop"} : i32
    %28 = llvm.and %19, %27  {stage = "loop"} : i32
    %29 = llvm.mul %16, %17 {stage = "loop"} : i32
    %30 = llvm.add %0, %29 {stage = "loop"} : i32
    %31 = lwi %30 {stage = "loop"} : i32
    %32 = llvm.add %25, %10 {stage = "loop"} : i32
    %33 = llvm.add %32, %26 {stage = "loop"} : i32
    %34 = llvm.add %33, %28 {stage = "loop"} : i32
    %35 = llvm.add %34, %18 {stage = "loop"} : i32
    %36 = llvm.add %35, %31 {stage = "loop"} : i32
    %37 = llvm.shl %21, %11 {stage = "loop"} : i32
    %38 = llvm.ashr %21, %12  {stage = "loop"} : i32
    %39 = llvm.or %37, %38  {stage = "loop"} : i32
    %40 = llvm.add %17, %14 {stage = "loop"} : i32
    %41 = beq [%40, %15] %17 {stage = "loop"} : i32
    %42 = llvm.shl %36, %13 {stage = "fini"} : i32
    %43 = llvm.add %42, %22 {stage = "fini"} : i32
    %44 = llvm.add %43, %39 {stage = "fini"} : i32
    %45 = llvm.add %44, %20 {stage = "fini"} : i32
    %46 = llvm.add %45, %19 {stage = "fini"} : i32
    llvm.return {stage = "fini"} %46 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

