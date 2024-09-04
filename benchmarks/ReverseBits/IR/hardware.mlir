module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @ReverseBits(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(2048 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(12 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(2052 : i32) : i32
    %21 = llvm.add %0, %1 : i32
    %22 = llvm.shl %21, %2 : i32
    %23 = llvm.add %20, %22 {constant = 51204 : i32} : i32
    %24 = llvm.add %4, %5 : i32
    %25 = llvm.shl %24, %6 : i32
    %26 = llvm.add %3, %25 {constant = 51200 : i32} : i32
    %27 = lwi %23 : i32->i32
    %28 = llvm.add %7, %16 {constant = 0 : i32} : i32
    %29 = llvm.add %8, %17 {constant = 0 : i32} : i32
    %30 = llvm.add %10, %18 {constant = 0 : i32} : i32
    %31 = llvm.add %11, %19 {constant = 0 : i32} : i32
    cond_br<eq> [%27 : i32, %29 : i32], ^bb5(%28 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    %32 = lwi %26 : i32->i32
    llvm.br ^bb3(%31, %32, %30 : i32, i32, i32)
  ^bb3(%33: i32, %34: i32, %35: i32):  // 2 preds: ^bb2, ^bb3
    %36 = llvm.shl %35, %15 : i32
    %37 = llvm.and %34, %14  : i32
    %38 = llvm.or %37, %36  : i32
    %39 = llvm.lshr %34, %13  : i32
    %40 = llvm.add %33, %12 : i32
    cond_br<ne> [%40 : i32, %27 : i32], ^bb3(%40, %39, %38 : i32, i32, i32), ^bb4
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%38 : i32)
  ^bb5(%41: i32):  // 2 preds: ^bb0, ^bb4
    %42 = llvm.mlir.constant(2056 : i32) : i32
    %43 = llvm.mlir.constant(12 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.add %43, %44 : i32
    %46 = llvm.mlir.constant(12 : i32) : i32
    %47 = llvm.shl %45, %46 : i32
    %48 = llvm.add %42, %47 {constant = 51208 : i32} : i32
    swi %41, %48 : i32, i32
    llvm.return %9 : i32
  }
}

