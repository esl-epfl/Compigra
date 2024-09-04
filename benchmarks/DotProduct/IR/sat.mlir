Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @DotProduct(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1", "in2", "in3"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(2048 : i32) : i32
    %5 = llvm.mlir.constant(12 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(12 : i32) : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(2112 : i32) : i32
    %10 = llvm.mlir.constant(12 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(12 : i32) : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(1 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(2176 : i32) : i32
    %23 = llvm.add %0, %1 : i32
    %24 = llvm.shl %23, %2 : i32
    %25 = llvm.add %22, %24 {constant = 51328 : i32} : i32
    %26 = llvm.add %5, %6 : i32
    %27 = llvm.shl %26, %7 : i32
    %28 = llvm.add %4, %27 {constant = 51200 : i32} : i32
    %29 = llvm.add %10, %11 : i32
    %30 = llvm.shl %29, %12 : i32
    %31 = llvm.add %9, %30 {constant = 51264 : i32} : i32
    %32 = lwi %25 : i32->i32
    %33 = llvm.add %13, %18 {constant = 0 : i32} : i32
    %34 = llvm.add %14, %19 {constant = 0 : i32} : i32
    %35 = llvm.add %15, %20 {constant = 0 : i32} : i32
    %36 = llvm.add %16, %21 {constant = 0 : i32} : i32
    cond_br<ge> [%36 : i32, %32 : i32], ^bb5(%35 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    %39 = llvm.add %34, %38 : i32
    %40 = llvm.mul %39, %3 : i32
    %41 = llvm.add %28, %40 : i32
    %42 = lwi %41 : i32->i32
    %43 = llvm.mul %39, %8 : i32
    %44 = llvm.add %31, %43 : i32
    %45 = lwi %44 : i32->i32
    %46 = llvm.add %39, %17 : i32
    %47 = llvm.add %46, %38 : i32
    %48 = llvm.mul %47, %3 : i32
    %49 = llvm.mul %47, %8 : i32
    %50 = llvm.add %47, %17 : i32
    cond_br<eq> [%46 : i32, %32 : i32], ^bb4(%33, %45, %42 : i32, i32, i32), ^bb3(%33, %45, %42, %48, %49, %50, %50 : i32, i32, i32, i32, i32, i32, i32)
  ^bb3(%51: i32, %52: i32, %53: i32, %54: i32, %55: i32, %56: i32, %57: i32):  // 2 preds: ^bb2, ^bb3
    %58 = llvm.add %51, %37 : i32
    %59 = llvm.mul %52, %53 : i32
    %60 = llvm.add %59, %58 : i32
    %61 = llvm.add %28, %54 : i32
    %62 = lwi %61 : i32->i32
    %63 = llvm.add %31, %55 : i32
    %64 = lwi %63 : i32->i32
    %65 = llvm.add %57, %38 : i32
    %66 = llvm.mul %65, %3 : i32
    %67 = llvm.mul %65, %8 : i32
    %68 = llvm.add %65, %17 : i32
    cond_br<ne> [%56 : i32, %32 : i32], ^bb3(%60, %64, %62, %66, %67, %68, %68 : i32, i32, i32, i32, i32, i32, i32), ^bb4(%60, %45, %42 : i32, i32, i32)
  ^bb4(%69: i32, %70: i32, %71: i32):  // 2 preds: ^bb2, ^bb3
    %72 = llvm.add %69, %37 : i32
    %73 = llvm.mul %70, %71 : i32
    %74 = llvm.add %73, %72 : i32
    llvm.br ^bb5(%74 : i32)
  ^bb5(%75: i32):  // 2 preds: ^bb0, ^bb4
    %76 = llvm.mlir.constant(2180 : i32) : i32
    %77 = llvm.mlir.constant(12 : i32) : i32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.add %77, %78 : i32
    %80 = llvm.mlir.constant(12 : i32) : i32
    %81 = llvm.shl %79, %80 : i32
    %82 = llvm.add %76, %81 {constant = 51332 : i32} : i32
    swi %75, %82 : i32, i32
    llvm.return %75 : i32
  }
}

