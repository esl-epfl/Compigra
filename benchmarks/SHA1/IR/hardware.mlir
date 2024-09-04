module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @SHA1(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = [], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(2048 : i32) : i32
    %1 = llvm.mlir.constant(12 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(12 : i32) : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %5 = llvm.mlir.constant(2048 : i32) : i32
    %6 = llvm.mlir.constant(12 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(12 : i32) : i32
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.mlir.constant(2048 : i32) : i32
    %11 = llvm.mlir.constant(12 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(12 : i32) : i32
    %14 = llvm.mlir.constant(4 : i32) : i32
    %15 = llvm.mlir.constant(2048 : i32) : i32
    %16 = llvm.mlir.constant(12 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(12 : i32) : i32
    %19 = llvm.mlir.constant(4 : i32) : i32
    %20 = llvm.mlir.constant(2048 : i32) : i32
    %21 = llvm.mlir.constant(12 : i32) : i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    %23 = llvm.mlir.constant(12 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(-3 : i32) : i32
    %26 = llvm.mlir.constant(-8 : i32) : i32
    %27 = llvm.mlir.constant(-14 : i32) : i32
    %28 = llvm.mlir.constant(-16 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = llvm.mlir.constant(16 : i32) : i32
    %32 = llvm.mlir.constant(80 : i32) : i32
    %33 = llvm.mlir.constant(4 : i32) : i32
    %34 = llvm.add %1, %2 : i32
    %35 = llvm.shl %34, %3 : i32
    %36 = llvm.add %0, %35 {constant = 51200 : i32} : i32
    %37 = llvm.add %6, %7 : i32
    %38 = llvm.shl %37, %8 : i32
    %39 = llvm.add %5, %38 {constant = 51200 : i32} : i32
    %40 = llvm.add %11, %12 : i32
    %41 = llvm.shl %40, %13 : i32
    %42 = llvm.add %10, %41 {constant = 51200 : i32} : i32
    %43 = llvm.add %16, %17 : i32
    %44 = llvm.shl %43, %18 : i32
    %45 = llvm.add %15, %44 {constant = 51200 : i32} : i32
    %46 = llvm.add %21, %22 : i32
    %47 = llvm.shl %46, %23 : i32
    %48 = llvm.add %20, %47 {constant = 51200 : i32} : i32
    %49 = llvm.add %24, %31 {constant = 16 : i32} : i32
    %50 = llvm.add %30, %32 {constant = 80 : i32} : i32
    llvm.br ^bb1(%49 : i32)
  ^bb1(%51: i32):  // 2 preds: ^bb0, ^bb1
    %52 = llvm.add %51, %25 : i32
    %53 = llvm.mul %52, %33 : i32
    %54 = llvm.add %36, %53 : i32
    %55 = lwi %54 : i32->i32
    %56 = llvm.add %51, %26 : i32
    %57 = llvm.mul %56, %4 : i32
    %58 = llvm.add %39, %57 : i32
    %59 = lwi %58 : i32->i32
    %60 = llvm.xor %59, %55  : i32
    %61 = llvm.add %51, %27 : i32
    %62 = llvm.mul %61, %9 : i32
    %63 = llvm.add %42, %62 : i32
    %64 = lwi %63 : i32->i32
    %65 = llvm.xor %60, %64  : i32
    %66 = llvm.add %51, %28 : i32
    %67 = llvm.mul %66, %14 : i32
    %68 = llvm.add %45, %67 : i32
    %69 = lwi %68 : i32->i32
    %70 = llvm.xor %65, %69  : i32
    %71 = llvm.mul %51, %19 : i32
    %72 = llvm.add %48, %71 : i32
    swi %70, %72 : i32, i32
    %73 = llvm.add %51, %29 : i32
    cond_br<ne> [%73 : i32, %50 : i32], ^bb1(%73 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    llvm.return
  }
}

