Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @isqrt(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(2048 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(12 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(12 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(2 : i32) : i32
    %15 = llvm.mlir.constant(2052 : i32) : i32
    %16 = llvm.add %0, %1 : i32
    %17 = llvm.shl %16, %2 : i32
    %18 = llvm.add %15, %17 {constant = 51204 : i32} : i32
    %19 = llvm.add %4, %5 : i32
    %20 = llvm.shl %19, %6 : i32
    %21 = llvm.add %3, %20 {constant = 51200 : i32} : i32
    %22 = llvm.add %8, %9 : i32
    %23 = llvm.shl %22, %10 : i32
    %24 = llvm.add %7, %23 {constant = 16384 : i32} : i32
    %25 = lwi %18 : i32->i32
    %26 = lwi %21 : i32->i32
    %27 = llvm.add %12, %14 {constant = 2 : i32} : i32
    llvm.br ^bb1(%25, %24 : i32, i32)
  ^bb1(%28: i32, %29: i32):  // 2 preds: ^bb0, ^bb1
    %30 = llvm.or %28, %29  : i32
    %31 = llvm.mul %30, %30 : i32
    %32 = llvm.sub %26, %31 : i32
    %33 = bsfa %32 : i32 [%28, %30]  : i32
    %34 = llvm.lshr %29, %11  : i32
    cond_br<ge> [%29 : i32, %27 : i32], ^bb1(%33, %34 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %35 = llvm.mlir.constant(2052 : i32) : i32
    %36 = llvm.mlir.constant(12 : i32) : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.add %36, %37 : i32
    %39 = llvm.mlir.constant(12 : i32) : i32
    %40 = llvm.shl %38, %39 : i32
    %41 = llvm.add %35, %40 {constant = 51204 : i32} : i32
    swi %33, %41 : i32, i32
    llvm.return %13 : i32
  }
}

