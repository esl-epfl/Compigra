module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @StringSearch(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2", "in3"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = [], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(51200 : i32) {hostValue = "arg0"} : i32
    %1 = llvm.mlir.constant(51204 : i32) {hostValue = "arg1"} : i32
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.constant(51208 : i32) {hostValue = "arg2"} : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %5 = llvm.mlir.constant(51216 : i32) {hostValue = "arg3"} : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(51208 : i32) {hostValue = "arg2"} : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(51216 : i32) {hostValue = "arg3"} : i32
    %10 = llvm.mlir.constant(51204 : i32) {returnValue = "arg1"} : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = lwi %0 : i32->i32
    %17 = lwi %1 : i32->i32
    %18 = llvm.add %16, %12 : i32
    cond_br<ge> [%13 : i32, %16 : i32], ^bb5(%17 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    %19 = llvm.mul %18, %2 : i32
    %20 = llvm.add %3, %19 : i32
    %21 = lwi %20 : i32->i8
    %22 = llvm.sext %21 : i8 to i32
    %23 = llvm.mul %22, %4 : i32
    %24 = llvm.add %5, %23 : i32
    %25 = lwi %24 : i32->i8
    llvm.br ^bb3(%15, %17 : i32, i32)
  ^bb3(%26: i32, %27: i32):  // 2 preds: ^bb2, ^bb3
    %28 = llvm.mul %26, %6 : i32
    %29 = llvm.add %7, %28 : i32
    %30 = lwi %29 : i32->i8
    %31 = llvm.sext %30 : i8 to i32
    %32 = llvm.mul %31, %8 : i32
    %33 = llvm.add %9, %32 : i32
    %34 = lwi %33 : i32->i8
    %35 = llvm.sub %34, %25 : i8
    %36 = llvm.xor %26, %11  : i32
    %37 = llvm.add %16, %36 : i32
    %38 = bzfa %35 : i8 [%37, %27]  : i32
    %39 = llvm.add %26, %14 : i32
    cond_br<ne> [%39 : i32, %18 : i32], ^bb3(%39, %38 : i32, i32), ^bb4
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%38 : i32)
  ^bb5(%40: i32):  // 2 preds: ^bb0, ^bb4
    swi %40, %10 : i32, i32
    llvm.return
  }
}

