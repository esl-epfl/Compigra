Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
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
    %51 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %52 = llvm.add %49, %51 : i32
    %53 = llvm.add %52, %25 : i32
    %54 = llvm.mul %53, %33 : i32
    %55 = llvm.add %52, %26 : i32
    %56 = llvm.mul %55, %4 : i32
    %57 = llvm.add %52, %27 : i32
    %58 = llvm.add %52, %29 : i32
    cond_br<eq> [%58 : i32, %50 : i32], ^bb5, ^bb2
  ^bb2:  // pred: ^bb1
    %59 = llvm.add %36, %54 : i32
    %60 = lwi %59 : i32->i32
    %61 = llvm.add %39, %56 : i32
    %62 = lwi %61 : i32->i32
    %63 = llvm.xor %62, %60  : i32
    %64 = llvm.mul %57, %9 : i32
    %65 = llvm.add %42, %64 : i32
    %66 = lwi %65 : i32->i32
    %67 = llvm.add %52, %28 : i32
    %68 = llvm.mul %67, %14 : i32
    %69 = llvm.add %45, %68 : i32
    %70 = llvm.mul %52, %19 : i32
    %71 = llvm.add %48, %70 : i32
    %72 = llvm.add %58, %51 : i32
    %73 = llvm.add %72, %25 : i32
    %74 = llvm.mul %73, %33 : i32
    %75 = llvm.add %72, %26 : i32
    %76 = llvm.mul %75, %4 : i32
    %77 = llvm.add %72, %27 : i32
    %78 = llvm.add %72, %29 : i32
    cond_br<eq> [%78 : i32, %50 : i32], ^bb4(%63, %66, %69, %71, %74, %76, %77, %72, %72 : i32, i32, i32, i32, i32, i32, i32, i32, i32), ^bb3(%63, %66, %69, %71, %74, %76, %77, %72, %72, %78 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  ^bb3(%79: i32, %80: i32, %81: i32, %82: i32, %83: i32, %84: i32, %85: i32, %86: i32, %87: i32, %88: i32):  // 2 preds: ^bb2, ^bb3
    %89 = llvm.xor %79, %80  : i32
    %90 = lwi %81 : i32->i32
    %91 = llvm.xor %89, %90  : i32
    swi %91, %82 : i32, i32
    %92 = llvm.add %36, %83 : i32
    %93 = lwi %92 : i32->i32
    %94 = llvm.add %39, %84 : i32
    %95 = lwi %94 : i32->i32
    %96 = llvm.xor %95, %93  : i32
    %97 = llvm.mul %85, %9 : i32
    %98 = llvm.add %42, %97 : i32
    %99 = lwi %98 : i32->i32
    %100 = llvm.add %86, %28 : i32
    %101 = llvm.mul %100, %14 : i32
    %102 = llvm.add %45, %101 : i32
    %103 = llvm.mul %87, %19 : i32
    %104 = llvm.add %48, %103 : i32
    %105 = llvm.add %88, %51 : i32
    %106 = llvm.add %105, %25 : i32
    %107 = llvm.mul %106, %33 : i32
    %108 = llvm.add %105, %26 : i32
    %109 = llvm.mul %108, %4 : i32
    %110 = llvm.add %105, %27 : i32
    %111 = llvm.add %105, %29 : i32
    cond_br<ne> [%111 : i32, %50 : i32], ^bb3(%96, %99, %102, %104, %107, %109, %110, %105, %105, %111 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32), ^bb4(%63, %66, %69, %71, %74, %76, %77, %72, %72 : i32, i32, i32, i32, i32, i32, i32, i32, i32)
  ^bb4(%112: i32, %113: i32, %114: i32, %115: i32, %116: i32, %117: i32, %118: i32, %119: i32, %120: i32):  // 2 preds: ^bb2, ^bb3
    %121 = llvm.xor %112, %113  : i32
    %122 = lwi %114 : i32->i32
    %123 = llvm.xor %121, %122  : i32
    swi %123, %115 : i32, i32
    %124 = llvm.add %36, %116 : i32
    %125 = lwi %124 : i32->i32
    %126 = llvm.add %39, %117 : i32
    %127 = lwi %126 : i32->i32
    %128 = llvm.xor %127, %125  : i32
    %129 = llvm.mul %118, %9 : i32
    %130 = llvm.add %42, %129 : i32
    %131 = lwi %130 : i32->i32
    %132 = llvm.xor %128, %131  : i32
    %133 = llvm.add %119, %28 : i32
    %134 = llvm.mul %133, %14 : i32
    %135 = llvm.add %45, %134 : i32
    %136 = lwi %135 : i32->i32
    %137 = llvm.xor %132, %136  : i32
    %138 = llvm.mul %120, %19 : i32
    %139 = llvm.add %48, %138 : i32
    swi %137, %139 : i32, i32
    llvm.br ^bb6
  ^bb5:  // pred: ^bb1
    %140 = llvm.add %36, %54 : i32
    %141 = lwi %140 : i32->i32
    %142 = llvm.add %39, %56 : i32
    %143 = lwi %142 : i32->i32
    %144 = llvm.xor %143, %141  : i32
    %145 = llvm.mul %57, %9 : i32
    %146 = llvm.add %42, %145 : i32
    %147 = lwi %146 : i32->i32
    %148 = llvm.xor %144, %147  : i32
    %149 = llvm.add %52, %28 : i32
    %150 = llvm.mul %149, %14 : i32
    %151 = llvm.add %45, %150 : i32
    %152 = lwi %151 : i32->i32
    %153 = llvm.xor %148, %152  : i32
    %154 = llvm.mul %52, %19 : i32
    %155 = llvm.add %48, %154 : i32
    swi %153, %155 : i32, i32
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    llvm.return
  }
}

