Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @isqrt(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(4096 : i32) {hostValue = "arg0"} : i32
    %1 = llvm.mlir.constant(4100 : i32) {returnValue = "arg1"} : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(12 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.constant(4100 : i32) {hostValue = "arg1"} : i32
    %11 = llvm.add %3, %4 : i32
    %12 = llvm.shl %11, %5 : i32
    %13 = llvm.add %2, %12 {constant = 16384 : i32} : i32
    %14 = lwi %10 : i32
    %15 = lwi %0 : i32
    %16 = llvm.add %7, %9 {constant = 2 : i32} : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1(%14, %13 : i32, i32)
  ^bb1(%19: i32, %20: i32):  // 2 preds: ^bb0, ^bb1
    %21 = llvm.add %19, %18 : i32
    %22 = llvm.add %20, %17 : i32
    %23 = llvm.or %21, %22  : i32
    %24 = llvm.mul %23, %23 : i32
    %25 = llvm.sub %15, %24 : i32
    %26 = bsfa %25 [%21, %23]  : i32
    %27 = llvm.lshr %22, %6  : i32
    bge [%22 : i32, %16 : i32], ^bb1(%26, %27 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    swi %26, %1 : i32, i32
    llvm.return %8 : i32
  }
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}) -> i32 attributes {passthrough = ["nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.call @time(%0) : (!llvm.ptr) -> i32
    llvm.call @srand(%2) : (i32) -> ()
    %3 = llvm.call @rand() : () -> i32
    llvm.return %1 : i32
  }
  llvm.func local_unnamed_addr @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @time(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @rand() -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

