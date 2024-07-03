module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @SHA1(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = [], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(2048 : i32) : i32
    %1 = llvm.mlir.constant(12 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(12 : i32) : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(4 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(-3 : i32) : i32
    %10 = llvm.mlir.constant(-8 : i32) : i32
    %11 = llvm.mlir.constant(-14 : i32) : i32
    %12 = llvm.mlir.constant(-16 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(16 : i32) : i32
    %16 = llvm.mlir.constant(80 : i32) : i32
    %17 = llvm.mlir.constant(4 : i32) : i32
    %18 = llvm.add %1, %2 : i32
    %19 = llvm.shl %18, %3 : i32
    %20 = llvm.add %0, %19 {constant = 51200 : i32} : i32
    %21 = llvm.add %8, %15 {constant = 16 : i32} : i32
    %22 = llvm.add %14, %16 {constant = 80 : i32} : i32
    llvm.br ^bb1(%21 : i32)
  ^bb1(%23: i32):  // 2 preds: ^bb0, ^bb1
    %24 = llvm.add %23, %9 : i32
    %25 = llvm.mul %24, %17 : i32
    %26 = llvm.add %20, %25 : i32
    %27 = lwi %26 : i32
    %28 = llvm.add %23, %10 : i32
    %29 = llvm.mul %28, %4 : i32
    %30 = llvm.add %20, %29 : i32
    %31 = lwi %30 : i32
    %32 = llvm.xor %31, %27  : i32
    %33 = llvm.add %23, %11 : i32
    %34 = llvm.mul %33, %5 : i32
    %35 = llvm.add %20, %34 : i32
    %36 = lwi %35 : i32
    %37 = llvm.xor %32, %36  : i32
    %38 = llvm.add %23, %12 : i32
    %39 = llvm.mul %38, %6 : i32
    %40 = llvm.add %20, %39 : i32
    %41 = lwi %40 : i32
    %42 = llvm.xor %37, %41  : i32
    %43 = llvm.mul %23, %7 : i32
    %44 = llvm.add %20, %43 : i32
    swi %42, %44 : i32, i32
    %45 = llvm.add %23, %13 : i32
    bne [%45 : i32, %22 : i32], ^bb1(%45 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    llvm.return
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

