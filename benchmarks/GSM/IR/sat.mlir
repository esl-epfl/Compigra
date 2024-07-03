Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @GSM(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(8 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.constant(4096 : i32) {returnValue = "arg0"} : i32
    %6 = llvm.mlir.constant(4100 : i32) {returnValue = "arg1"} : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(4088 : i32) : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(12 : i32) : i32
    %15 = llvm.mlir.constant(-1 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(24 : i32) : i32
    %18 = llvm.mlir.constant(4095 : i32) : i32
    %19 = llvm.mlir.constant(7 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(12 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(40 : i32) : i32
    %26 = llvm.mlir.constant(4096 : i32) {hostValue = "arg0"} : i32
    %27 = llvm.add %2, %3 : i32
    %28 = llvm.shl %27, %4 : i32
    %29 = llvm.add %1, %28 {constant = 4104 : i32} : i32
    %30 = llvm.add %12, %13 : i32
    %31 = llvm.shl %30, %14 : i32
    %32 = llvm.add %11, %31 : i32
    %33 = llvm.add %15, %16 : i32
    %34 = llvm.shl %33, %17 : i32
    %35 = llvm.add %31, %34 {constant = -32768 : i32} : i32
    %36 = llvm.add %19, %20 : i32
    %37 = llvm.shl %36, %21 : i32
    %38 = llvm.add %18, %37 {constant = 32767 : i32} : i32
    %39 = lwi %26 : i32
    %40 = llvm.add %9, %24 {constant = 0 : i32} : i32
    %41 = llvm.add %23, %25 {constant = 40 : i32} : i32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1(%40, %39 : i32, i32)
  ^bb1(%44: i32, %45: i32):  // 2 preds: ^bb0, ^bb1
    %46 = llvm.add %44, %43 : i32
    %47 = llvm.add %45, %42 : i32
    %48 = llvm.mul %46, %0 : i32
    %49 = llvm.add %29, %48 : i32
    %50 = lwi %49 : i32
    %51 = llvm.sub %50, %10 : i32
    %52 = llvm.sub %50, %35 : i32
    %53 = llvm.sub %7, %50 : i32
    %54 = bzfa %52 [%38, %53]  : i32
    %55 = bsfa %51 [%54, %50]  : i32
    %56 = llvm.sub %47, %55 : i32
    %57 = bsfa %56 [%55, %47]  : i32
    %58 = llvm.add %46, %22 : i32
    bne [%58 : i32, %41 : i32], ^bb1(%58, %57 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    swi %57, %5 : i32, i32
    swi %55, %6 : i32, i32
    llvm.return %8 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    llvm.unreachable
  }
}

