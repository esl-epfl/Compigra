module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @GSM(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(2056 : i32) : i32
    %5 = llvm.mlir.constant(12 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(12 : i32) : i32
    %8 = llvm.mlir.constant(2052 : i32) : i32
    %9 = llvm.mlir.constant(12 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(12 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(4088 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(12 : i32) : i32
    %20 = llvm.mlir.constant(-1 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(24 : i32) : i32
    %23 = llvm.mlir.constant(4095 : i32) : i32
    %24 = llvm.mlir.constant(7 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(12 : i32) : i32
    %27 = llvm.mlir.constant(1 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(40 : i32) : i32
    %31 = llvm.mlir.constant(2048 : i32) : i32
    %32 = llvm.add %0, %1 : i32
    %33 = llvm.shl %32, %2 : i32
    %34 = llvm.add %31, %33 {constant = 51200 : i32} : i32
    %35 = llvm.add %5, %6 : i32
    %36 = llvm.shl %35, %7 : i32
    %37 = llvm.add %4, %36 {constant = 51208 : i32} : i32
    %38 = llvm.add %9, %10 : i32
    %39 = llvm.shl %38, %11 : i32
    %40 = llvm.add %8, %39 {constant = 51204 : i32} : i32
    %41 = llvm.add %17, %18 : i32
    %42 = llvm.shl %41, %19 : i32
    %43 = llvm.add %16, %42 : i32
    %44 = llvm.add %20, %21 : i32
    %45 = llvm.shl %44, %22 : i32
    %46 = llvm.add %42, %45 {constant = -32768 : i32} : i32
    %47 = llvm.add %24, %25 : i32
    %48 = llvm.shl %47, %26 : i32
    %49 = llvm.add %23, %48 {constant = 32767 : i32} : i32
    %50 = lwi %34 : i32
    %51 = llvm.add %15, %29 {constant = 0 : i32} : i32
    %52 = llvm.add %28, %30 {constant = 40 : i32} : i32
    llvm.br ^bb1(%51, %50 : i32, i32)
  ^bb1(%53: i32, %54: i32):  // 2 preds: ^bb0, ^bb1
    %55 = llvm.mul %53, %3 : i32
    %56 = llvm.add %37, %55 : i32
    %57 = lwi %56 : i32
    %58 = llvm.sub %57, %12 : i32
    %59 = llvm.sub %57, %46 : i32
    %60 = llvm.sub %13, %57 : i32
    %61 = bzfa %59 [%49, %60]  : i32
    %62 = bsfa %58 [%61, %57]  : i32
    %63 = llvm.sub %54, %62 : i32
    %64 = bsfa %63 [%62, %54]  : i32
    %65 = llvm.add %53, %27 : i32
    bne [%65 : i32, %52 : i32], ^bb1(%65, %64 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    swi %64, %34 : i32, i32
    swi %62, %40 : i32, i32
    llvm.return %14 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    llvm.unreachable
  }
}

