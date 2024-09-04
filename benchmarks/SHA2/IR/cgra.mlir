module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @SHA2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(51200 : i32) {hostValue = "arg0"} : i32
    %2 = llvm.mlir.constant(51264 : i32) {returnValue = "arg1"} : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(5 : i32) : i32
    %11 = llvm.mlir.constant(27 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(1518500249 : i32) : i32
    %14 = llvm.mlir.constant(30 : i32) : i32
    %15 = llvm.mlir.constant(2 : i32) : i32
    %16 = llvm.mlir.constant(1 : i32) : i32
    %17 = llvm.mlir.constant(1 : i32) : i32
    %18 = llvm.mlir.constant(20 : i32) : i32
    llvm.br ^bb1(%9, %8, %7, %6, %5, %4 : i32, i32, i32, i32, i32, i32)
  ^bb1(%19: i32, %20: i32, %21: i32, %22: i32, %23: i32, %24: i32):  // 2 preds: ^bb0, ^bb1
    %25 = llvm.shl %24, %10 : i32
    %26 = llvm.ashr %24, %11  : i32
    %27 = llvm.or %25, %26  : i32
    %28 = llvm.and %22, %23  : i32
    %29 = llvm.xor %23, %12  : i32
    %30 = llvm.and %21, %29  : i32
    %31 = llvm.mul %19, %0 : i32
    %32 = llvm.add %1, %31 : i32
    %33 = lwi %32 : i32->i32
    %34 = llvm.add %27, %13 : i32
    %35 = llvm.add %34, %28 : i32
    %36 = llvm.add %35, %30 : i32
    %37 = llvm.add %36, %20 : i32
    %38 = llvm.add %37, %33 : i32
    %39 = llvm.shl %23, %14 : i32
    %40 = llvm.ashr %23, %15  : i32
    %41 = llvm.or %39, %40  : i32
    %42 = llvm.add %19, %16 : i32
    cond_br<ne> [%42 : i32, %18 : i32], ^bb1(%42, %21, %22, %41, %24, %38 : i32, i32, i32, i32, i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %43 = llvm.shl %38, %17 : i32
    %44 = llvm.add %43, %24 : i32
    %45 = llvm.add %44, %41 : i32
    %46 = llvm.add %45, %22 : i32
    %47 = llvm.add %46, %21 : i32
    swi %47, %2 : i32, i32
    llvm.return %3 : i32
  }
}

