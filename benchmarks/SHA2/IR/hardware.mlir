module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @SHA2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
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
    %10 = llvm.mlir.constant(5 : i32) : i32
    %11 = llvm.mlir.constant(27 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(2457 : i32) : i32
    %14 = llvm.mlir.constant(2087 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(12 : i32) : i32
    %17 = llvm.mlir.constant(90 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(24 : i32) : i32
    %20 = llvm.mlir.constant(30 : i32) : i32
    %21 = llvm.mlir.constant(2 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = llvm.mlir.constant(20 : i32) : i32
    %32 = llvm.mlir.constant(4 : i32) : i32
    %33 = llvm.add %1, %2 : i32
    %34 = llvm.shl %33, %3 : i32
    %35 = llvm.add %0, %34 {constant = 51200 : i32} : i32
    %36 = llvm.add %14, %15 : i32
    %37 = llvm.shl %36, %16 : i32
    %38 = llvm.add %13, %37 : i32
    %39 = llvm.add %17, %18 : i32
    %40 = llvm.shl %39, %19 : i32
    %41 = llvm.add %37, %40 {constant = 1518500249 : i32} : i32
    %42 = llvm.add %4, %25 {constant = 0 : i32} : i32
    %43 = llvm.add %5, %26 {constant = 0 : i32} : i32
    %44 = llvm.add %6, %27 {constant = 0 : i32} : i32
    %45 = llvm.add %7, %28 {constant = 0 : i32} : i32
    %46 = llvm.add %8, %29 {constant = 0 : i32} : i32
    %47 = llvm.add %9, %30 {constant = 0 : i32} : i32
    %48 = llvm.add %24, %31 {constant = 20 : i32} : i32
    llvm.br ^bb1(%47, %46, %45, %44, %43, %42 : i32, i32, i32, i32, i32, i32)
  ^bb1(%49: i32, %50: i32, %51: i32, %52: i32, %53: i32, %54: i32):  // 2 preds: ^bb0, ^bb1
    %55 = llvm.shl %54, %10 : i32
    %56 = llvm.ashr %54, %11  : i32
    %57 = llvm.or %55, %56  : i32
    %58 = llvm.and %52, %53  : i32
    %59 = llvm.xor %53, %12  : i32
    %60 = llvm.and %51, %59  : i32
    %61 = llvm.mul %49, %32 : i32
    %62 = llvm.add %35, %61 : i32
    %63 = lwi %62 : i32
    %64 = llvm.add %57, %41 : i32
    %65 = llvm.add %64, %58 : i32
    %66 = llvm.add %65, %60 : i32
    %67 = llvm.add %66, %50 : i32
    %68 = llvm.add %67, %63 : i32
    %69 = llvm.shl %53, %20 : i32
    %70 = llvm.ashr %53, %21  : i32
    %71 = llvm.or %69, %70  : i32
    %72 = llvm.add %49, %22 : i32
    bne [%72 : i32, %48 : i32], ^bb1(%72, %51, %52, %71, %54, %68 : i32, i32, i32, i32, i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %73 = llvm.shl %68, %23 : i32
    %74 = llvm.add %73, %54 : i32
    %75 = llvm.add %74, %71 : i32
    %76 = llvm.add %75, %52 : i32
    %77 = llvm.add %76, %51 : i32
    llvm.return %77 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

