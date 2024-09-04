module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @ReverseBits(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(51204 : i32) {hostValue = "arg1"} : i32
    %1 = llvm.mlir.constant(51200 : i32) {hostValue = "arg0"} : i32
    %2 = llvm.mlir.constant(51208 : i32) {returnValue = "arg2"} : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = lwi %0 : i32->i32
    cond_br<eq> [%12 : i32, %4 : i32], ^bb5(%3 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    %13 = lwi %1 : i32->i32
    llvm.br ^bb3(%7, %13, %6 : i32, i32, i32)
  ^bb3(%14: i32, %15: i32, %16: i32):  // 2 preds: ^bb2, ^bb3
    %17 = llvm.shl %16, %11 : i32
    %18 = llvm.and %15, %10  : i32
    %19 = llvm.or %18, %17  : i32
    %20 = llvm.lshr %15, %9  : i32
    %21 = llvm.add %14, %8 : i32
    cond_br<ne> [%21 : i32, %12 : i32], ^bb3(%21, %20, %19 : i32, i32, i32), ^bb4
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%19 : i32)
  ^bb5(%22: i32):  // 2 preds: ^bb0, ^bb4
    swi %22, %2 : i32, i32
    llvm.return %5 : i32
  }
}

