module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @SHA2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(2048 : i32) : i32
    %1 = llvm.mlir.constant(12 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(12 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(27 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(2457 : i32) : i32
    %15 = llvm.mlir.constant(2087 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(12 : i32) : i32
    %18 = llvm.mlir.constant(90 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(24 : i32) : i32
    %21 = llvm.mlir.constant(30 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.mlir.constant(1 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.mlir.constant(20 : i32) : i32
    %33 = llvm.mlir.constant(4 : i32) : i32
    %34 = llvm.add %1, %2 : i32
    %35 = llvm.shl %34, %3 : i32
    %36 = llvm.add %0, %35 {constant = 51200 : i32} : i32
    %37 = llvm.add %15, %16 : i32
    %38 = llvm.shl %37, %17 : i32
    %39 = llvm.add %14, %38 : i32
    %40 = llvm.add %18, %19 : i32
    %41 = llvm.shl %40, %20 : i32
    %42 = llvm.add %39, %41 {constant = 1518500249 : i32} : i32
    %43 = llvm.add %5, %26 {constant = 0 : i32} : i32
    %44 = llvm.add %6, %27 {constant = 0 : i32} : i32
    %45 = llvm.add %7, %28 {constant = 0 : i32} : i32
    %46 = llvm.add %8, %29 {constant = 0 : i32} : i32
    %47 = llvm.add %9, %30 {constant = 0 : i32} : i32
    %48 = llvm.add %10, %31 {constant = 0 : i32} : i32
    %49 = llvm.add %25, %32 {constant = 20 : i32} : i32
    llvm.br ^bb1(%48, %47, %46, %45, %44, %43 : i32, i32, i32, i32, i32, i32)
  ^bb1(%50: i32, %51: i32, %52: i32, %53: i32, %54: i32, %55: i32):  // 2 preds: ^bb0, ^bb1
    %56 = llvm.shl %55, %11 : i32
    %57 = llvm.ashr %55, %12  : i32
    %58 = llvm.or %56, %57  : i32
    %59 = llvm.and %53, %54  : i32
    %60 = llvm.xor %54, %13  : i32
    %61 = llvm.and %52, %60  : i32
    %62 = llvm.mul %50, %33 : i32
    %63 = llvm.add %36, %62 : i32
    %64 = lwi %63 : i32->i32
    %65 = llvm.add %58, %42 : i32
    %66 = llvm.add %65, %59 : i32
    %67 = llvm.add %66, %61 : i32
    %68 = llvm.add %67, %51 : i32
    %69 = llvm.add %68, %64 : i32
    %70 = llvm.shl %54, %21 : i32
    %71 = llvm.ashr %54, %22  : i32
    %72 = llvm.or %70, %71  : i32
    %73 = llvm.add %50, %23 : i32
    cond_br<ne> [%73 : i32, %49 : i32], ^bb1(%73, %52, %53, %72, %55, %69 : i32, i32, i32, i32, i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %74 = llvm.shl %69, %24 : i32
    %75 = llvm.add %74, %55 : i32
    %76 = llvm.add %75, %72 : i32
    %77 = llvm.add %76, %53 : i32
    %78 = llvm.add %77, %52 : i32
    %79 = llvm.mlir.constant(2112 : i32) : i32
    %80 = llvm.mlir.constant(12 : i32) : i32
    %81 = llvm.mlir.constant(0 : i32) : i32
    %82 = llvm.add %80, %81 : i32
    %83 = llvm.mlir.constant(12 : i32) : i32
    %84 = llvm.shl %82, %83 : i32
    %85 = llvm.add %79, %84 {constant = 51264 : i32} : i32
    swi %78, %85 : i32, i32
    llvm.return %4 : i32
  }
}

