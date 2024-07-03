module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @GSM(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(51200 : i32) {hostValue = "arg0"} : i32
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.constant(51208 : i32) {hostValue = "arg2"} : i32
    %3 = llvm.mlir.constant(51200 : i32) {returnValue = "arg0"} : i32
    %4 = llvm.mlir.constant(51204 : i32) {returnValue = "arg1"} : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(-32768 : i32) : i32
    %10 = llvm.mlir.constant(32767 : i32) : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(40 : i32) : i32
    %13 = lwi %0 : i32
    llvm.br ^bb1(%8, %13 : i32, i32)
  ^bb1(%14: i32, %15: i32):  // 2 preds: ^bb0, ^bb1
    %16 = llvm.mul %14, %1 : i32
    %17 = llvm.add %2, %16 : i32
    %18 = lwi %17 : i32
    %19 = llvm.sub %18, %5 : i32
    %20 = llvm.sub %18, %9 : i32
    %21 = llvm.sub %6, %18 : i32
    %22 = bzfa %20 [%10, %21]  : i32
    %23 = bsfa %19 [%22, %18]  : i32
    %24 = llvm.sub %15, %23 : i32
    %25 = bsfa %24 [%23, %15]  : i32
    %26 = llvm.add %14, %11 : i32
    bne [%26 : i32, %12 : i32], ^bb1(%26, %25 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    swi %25, %3 : i32, i32
    swi %23, %4 : i32, i32
    llvm.return %7 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    llvm.unreachable
  }
}

