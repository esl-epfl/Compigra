Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  cgra.func @GSM(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> i32 attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0", "in1"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.constant(2052 : i32) : i32
    %5 = llvm.mlir.constant(12 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(12 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(4088 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(12 : i32) : i32
    %16 = llvm.mlir.constant(-1 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(24 : i32) : i32
    %19 = llvm.mlir.constant(4095 : i32) : i32
    %20 = llvm.mlir.constant(7 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(12 : i32) : i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(40 : i32) : i32
    %27 = llvm.mlir.constant(2048 : i32) : i32
    %28 = llvm.add %0, %1 : i32
    %29 = llvm.shl %28, %2 : i32
    %30 = llvm.add %27, %29 {constant = 51200 : i32} : i32
    %31 = llvm.add %5, %6 : i32
    %32 = llvm.shl %31, %7 : i32
    %33 = llvm.add %4, %32 {constant = 51204 : i32} : i32
    %34 = llvm.add %13, %14 : i32
    %35 = llvm.shl %34, %15 : i32
    %36 = llvm.add %12, %35 : i32
    %37 = llvm.add %16, %17 : i32
    %38 = llvm.shl %37, %18 : i32
    %39 = llvm.add %36, %38 {constant = -32768 : i32} : i32
    %40 = llvm.add %20, %21 : i32
    %41 = llvm.shl %40, %22 : i32
    %42 = llvm.add %19, %41 {constant = 32767 : i32} : i32
    %43 = lwi %30 : i32->i32
    %44 = llvm.add %11, %25 {constant = 0 : i32} : i32
    %45 = llvm.add %24, %26 {constant = 40 : i32} : i32
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %48 = llvm.add %44, %47 : i32
    %49 = llvm.mul %48, %3 : i32
    %50 = llvm.add %33, %49 : i32
    %51 = lwi %50 : i32->i32
    %52 = llvm.sub %51, %39 : i32
    %53 = llvm.sub %9, %51 : i32
    %54 = llvm.add %48, %23 : i32
    cond_br<eq> [%54 : i32, %45 : i32], ^bb3(%43, %51, %52, %53, %51 : i32, i32, i32, i32, i32), ^bb2(%43, %51, %52, %53, %51, %54 : i32, i32, i32, i32, i32, i32)
  ^bb2(%55: i32, %56: i32, %57: i32, %58: i32, %59: i32, %60: i32):  // 2 preds: ^bb1, ^bb2
    %61 = llvm.add %55, %46 : i32
    %62 = llvm.sub %56, %8 : i32
    %63 = bzfa %57 : i32 [%42, %58]  : i32
    %64 = bsfa %62 : i32 [%63, %59]  : i32
    %65 = llvm.sub %61, %64 : i32
    %66 = bsfa %65 : i32 [%64, %61]  : i32
    %67 = llvm.add %60, %47 : i32
    %68 = llvm.mul %67, %3 : i32
    %69 = llvm.add %33, %68 : i32
    %70 = lwi %69 : i32->i32
    %71 = llvm.sub %70, %39 : i32
    %72 = llvm.sub %9, %70 : i32
    %73 = llvm.add %67, %23 : i32
    cond_br<ne> [%73 : i32, %45 : i32], ^bb2(%66, %70, %71, %72, %70, %73 : i32, i32, i32, i32, i32, i32), ^bb3(%66, %51, %52, %53, %51 : i32, i32, i32, i32, i32)
  ^bb3(%74: i32, %75: i32, %76: i32, %77: i32, %78: i32):  // 2 preds: ^bb1, ^bb2
    %79 = llvm.add %74, %46 : i32
    %80 = llvm.sub %75, %8 : i32
    %81 = bzfa %76 : i32 [%42, %77]  : i32
    %82 = bsfa %80 : i32 [%81, %78]  : i32
    %83 = llvm.sub %79, %82 : i32
    %84 = bsfa %83 : i32 [%82, %79]  : i32
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %85 = llvm.mlir.constant(2048 : i32) : i32
    %86 = llvm.mlir.constant(12 : i32) : i32
    %87 = llvm.mlir.constant(0 : i32) : i32
    %88 = llvm.add %86, %87 : i32
    %89 = llvm.mlir.constant(12 : i32) : i32
    %90 = llvm.shl %88, %89 : i32
    %91 = llvm.add %85, %90 {constant = 51200 : i32} : i32
    swi %84, %91 : i32, i32
    llvm.return %10 : i32
  }
}

