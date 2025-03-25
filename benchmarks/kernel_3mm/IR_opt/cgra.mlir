module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>, %arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>, %arg6: memref<4x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4_i32_0 = arith.constant {DimProd = 1 : i32, arg = 6 : i32} 4 : i32
    %c512_i32 = arith.constant {BaseAddr = "arg6"} 512 : i32
    %c4_i32_1 = arith.constant {DimProd = 1 : i32, arg = 5 : i32} 4 : i32
    %c448_i32 = arith.constant {BaseAddr = "arg5"} 448 : i32
    %c4_i32_2 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 4 : i32
    %c384_i32 = arith.constant {BaseAddr = "arg4"} 384 : i32
    %c4_i32_3 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 4 : i32
    %c320_i32 = arith.constant {BaseAddr = "arg3"} 320 : i32
    %c4_i32_4 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 4 : i32
    %c256_i32 = arith.constant {BaseAddr = "arg2"} 256 : i32
    %c4_i32_5 = arith.constant {DimProd = 1 : i32, arg = 1 : i32} 4 : i32
    %c192_i32 = arith.constant {BaseAddr = "arg1"} 192 : i32
    %c4_i32_6 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 4 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb7
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb8(%c0 : index), ^bb2(%c0 : index)
  ^bb2(%3: index):  // 2 preds: ^bb1, ^bb6
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %6 = arith.index_cast %0 : index to i32
    %7 = arith.muli %6, %c4_i32_2 : i32
    %8 = arith.index_cast %3 : index to i32
    %9 = arith.addi %7, %8 : i32
    %10 = arith.muli %9, %c4_i32 : i32
    %11 = arith.addi %c384_i32, %10 : i32
    cgra.swi %c0_i32, %11 : i32, i32
    cf.br ^bb4(%c0 : index)
  ^bb4(%12: index):  // 2 preds: ^bb3, ^bb5
    %13 = arith.index_cast %12 : index to i32
    %14 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%13 : i32, %14 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %15 = arith.index_cast %0 : index to i32
    %16 = arith.muli %15, %c4_i32_6 : i32
    %17 = arith.index_cast %12 : index to i32
    %18 = arith.addi %16, %17 : i32
    %19 = arith.muli %18, %c4_i32 : i32
    %20 = arith.addi %c128_i32, %19 : i32
    %21 = cgra.lwi %20 : i32->i32
    %22 = arith.index_cast %12 : index to i32
    %23 = arith.muli %22, %c4_i32_5 : i32
    %24 = arith.index_cast %3 : index to i32
    %25 = arith.addi %23, %24 : i32
    %26 = arith.muli %25, %c4_i32 : i32
    %27 = arith.addi %c192_i32, %26 : i32
    %28 = cgra.lwi %27 : i32->i32
    %29 = arith.muli %21, %28 : i32
    %30 = arith.index_cast %0 : index to i32
    %31 = arith.muli %30, %c4_i32_2 : i32
    %32 = arith.index_cast %3 : index to i32
    %33 = arith.addi %31, %32 : i32
    %34 = arith.muli %33, %c4_i32 : i32
    %35 = arith.addi %c384_i32, %34 : i32
    %36 = cgra.lwi %35 : i32->i32
    %37 = arith.addi %36, %29 : i32
    %38 = arith.index_cast %0 : index to i32
    %39 = arith.muli %38, %c4_i32_2 : i32
    %40 = arith.index_cast %3 : index to i32
    %41 = arith.addi %39, %40 : i32
    %42 = arith.muli %41, %c4_i32 : i32
    %43 = arith.addi %c384_i32, %42 : i32
    cgra.swi %37, %43 : i32, i32
    %44 = arith.addi %12, %c1 : index
    cf.br ^bb4(%44 : index)
  ^bb6:  // pred: ^bb4
    %45 = arith.addi %3, %c1 : index
    cf.br ^bb2(%45 : index)
  ^bb7:  // pred: ^bb2
    %46 = arith.addi %0, %c1 : index
    cf.br ^bb1(%46 : index)
  ^bb8(%47: index):  // 2 preds: ^bb1, ^bb14
    %48 = arith.index_cast %47 : index to i32
    %49 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%48 : i32, %49 : i32], ^bb15(%c0 : index), ^bb9(%c0 : index)
  ^bb9(%50: index):  // 2 preds: ^bb8, ^bb13
    %51 = arith.index_cast %50 : index to i32
    %52 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%51 : i32, %52 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %53 = arith.index_cast %47 : index to i32
    %54 = arith.muli %53, %c4_i32_1 : i32
    %55 = arith.index_cast %50 : index to i32
    %56 = arith.addi %54, %55 : i32
    %57 = arith.muli %56, %c4_i32 : i32
    %58 = arith.addi %c448_i32, %57 : i32
    cgra.swi %c0_i32, %58 : i32, i32
    cf.br ^bb11(%c0 : index)
  ^bb11(%59: index):  // 2 preds: ^bb10, ^bb12
    %60 = arith.index_cast %59 : index to i32
    %61 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%60 : i32, %61 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %62 = arith.index_cast %47 : index to i32
    %63 = arith.muli %62, %c4_i32_4 : i32
    %64 = arith.index_cast %59 : index to i32
    %65 = arith.addi %63, %64 : i32
    %66 = arith.muli %65, %c4_i32 : i32
    %67 = arith.addi %c256_i32, %66 : i32
    %68 = cgra.lwi %67 : i32->i32
    %69 = arith.index_cast %59 : index to i32
    %70 = arith.muli %69, %c4_i32_3 : i32
    %71 = arith.index_cast %50 : index to i32
    %72 = arith.addi %70, %71 : i32
    %73 = arith.muli %72, %c4_i32 : i32
    %74 = arith.addi %c320_i32, %73 : i32
    %75 = cgra.lwi %74 : i32->i32
    %76 = arith.muli %68, %75 : i32
    %77 = arith.index_cast %47 : index to i32
    %78 = arith.muli %77, %c4_i32_1 : i32
    %79 = arith.index_cast %50 : index to i32
    %80 = arith.addi %78, %79 : i32
    %81 = arith.muli %80, %c4_i32 : i32
    %82 = arith.addi %c448_i32, %81 : i32
    %83 = cgra.lwi %82 : i32->i32
    %84 = arith.addi %83, %76 : i32
    %85 = arith.index_cast %47 : index to i32
    %86 = arith.muli %85, %c4_i32_1 : i32
    %87 = arith.index_cast %50 : index to i32
    %88 = arith.addi %86, %87 : i32
    %89 = arith.muli %88, %c4_i32 : i32
    %90 = arith.addi %c448_i32, %89 : i32
    cgra.swi %84, %90 : i32, i32
    %91 = arith.addi %59, %c1 : index
    cf.br ^bb11(%91 : index)
  ^bb13:  // pred: ^bb11
    %92 = arith.addi %50, %c1 : index
    cf.br ^bb9(%92 : index)
  ^bb14:  // pred: ^bb9
    %93 = arith.addi %47, %c1 : index
    cf.br ^bb8(%93 : index)
  ^bb15(%94: index):  // 2 preds: ^bb8, ^bb21
    %95 = arith.index_cast %94 : index to i32
    %96 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%95 : i32, %96 : i32], ^bb22, ^bb16(%c0 : index)
  ^bb16(%97: index):  // 2 preds: ^bb15, ^bb20
    %98 = arith.index_cast %97 : index to i32
    %99 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%98 : i32, %99 : i32], ^bb21, ^bb17
  ^bb17:  // pred: ^bb16
    %100 = arith.index_cast %94 : index to i32
    %101 = arith.muli %100, %c4_i32_0 : i32
    %102 = arith.index_cast %97 : index to i32
    %103 = arith.addi %101, %102 : i32
    %104 = arith.muli %103, %c4_i32 : i32
    %105 = arith.addi %c512_i32, %104 : i32
    cgra.swi %c0_i32, %105 : i32, i32
    cf.br ^bb18(%c0 : index)
  ^bb18(%106: index):  // 2 preds: ^bb17, ^bb19
    %107 = arith.index_cast %106 : index to i32
    %108 = arith.index_cast %c4 : index to i32
    cgra.cond_br<ge> [%107 : i32, %108 : i32], ^bb20, ^bb19
  ^bb19:  // pred: ^bb18
    %109 = arith.index_cast %94 : index to i32
    %110 = arith.muli %109, %c4_i32_2 : i32
    %111 = arith.index_cast %106 : index to i32
    %112 = arith.addi %110, %111 : i32
    %113 = arith.muli %112, %c4_i32 : i32
    %114 = arith.addi %c384_i32, %113 : i32
    %115 = cgra.lwi %114 : i32->i32
    %116 = arith.index_cast %106 : index to i32
    %117 = arith.muli %116, %c4_i32_1 : i32
    %118 = arith.index_cast %97 : index to i32
    %119 = arith.addi %117, %118 : i32
    %120 = arith.muli %119, %c4_i32 : i32
    %121 = arith.addi %c448_i32, %120 : i32
    %122 = cgra.lwi %121 : i32->i32
    %123 = arith.muli %115, %122 : i32
    %124 = arith.index_cast %94 : index to i32
    %125 = arith.muli %124, %c4_i32_0 : i32
    %126 = arith.index_cast %97 : index to i32
    %127 = arith.addi %125, %126 : i32
    %128 = arith.muli %127, %c4_i32 : i32
    %129 = arith.addi %c512_i32, %128 : i32
    %130 = cgra.lwi %129 : i32->i32
    %131 = arith.addi %130, %123 : i32
    %132 = arith.index_cast %94 : index to i32
    %133 = arith.muli %132, %c4_i32_0 : i32
    %134 = arith.index_cast %97 : index to i32
    %135 = arith.addi %133, %134 : i32
    %136 = arith.muli %135, %c4_i32 : i32
    %137 = arith.addi %c512_i32, %136 : i32
    cgra.swi %131, %137 : i32, i32
    %138 = arith.addi %106, %c1 : index
    cf.br ^bb18(%138 : index)
  ^bb20:  // pred: ^bb18
    %139 = arith.addi %97, %c1 : index
    cf.br ^bb16(%139 : index)
  ^bb21:  // pred: ^bb16
    %140 = arith.addi %94, %c1 : index
    cf.br ^bb15(%140 : index)
  ^bb22:  // pred: ^bb15
    return
  }
}

