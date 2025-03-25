module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<8x8xi32>, %arg3: memref<8x8xi32>, %arg4: memref<8x8xi32>, %arg5: memref<8x8xi32>, %arg6: memref<8x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant {DimProd = 1 : i32, arg = 6 : i32} 8 : i32
    %c1160_i32 = arith.constant {BaseAddr = "arg6"} 1160 : i32
    %c8_i32_0 = arith.constant {DimProd = 1 : i32, arg = 5 : i32} 8 : i32
    %c904_i32 = arith.constant {BaseAddr = "arg5"} 904 : i32
    %c8_i32_1 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 8 : i32
    %c648_i32 = arith.constant {BaseAddr = "arg4"} 648 : i32
    %c8_i32_2 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 8 : i32
    %c392_i32 = arith.constant {BaseAddr = "arg3"} 392 : i32
    %c8_i32_3 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 8 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb7
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%3 : i32, %4 : i32], ^bb8(%c0 : index), ^bb2(%c0 : index)
  ^bb2(%5: index):  // 2 preds: ^bb1, ^bb6
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%6 : i32, %7 : i32], ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %8 = arith.index_cast %2 : index to i32
    %9 = arith.muli %8, %c8_i32_3 : i32
    %10 = arith.index_cast %5 : index to i32
    %11 = arith.addi %9, %10 : i32
    %12 = arith.muli %11, %c4_i32 : i32
    %13 = arith.addi %c136_i32, %12 : i32
    cgra.swi %c0_i32, %13 : i32, i32
    cf.br ^bb4(%c0 : index)
  ^bb4(%14: index):  // 2 preds: ^bb3, ^bb5
    %15 = arith.index_cast %14 : index to i32
    %16 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%15 : i32, %16 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %17 = arith.index_cast %2 : index to i32
    %18 = arith.muli %17, %c8_i32_2 : i32
    %19 = arith.index_cast %14 : index to i32
    %20 = arith.addi %18, %19 : i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.addi %c392_i32, %21 : i32
    %23 = cgra.lwi %22 : i32->i32
    %24 = arith.muli %0, %23 : i32
    %25 = arith.index_cast %14 : index to i32
    %26 = arith.muli %25, %c8_i32_1 : i32
    %27 = arith.index_cast %5 : index to i32
    %28 = arith.addi %26, %27 : i32
    %29 = arith.muli %28, %c4_i32 : i32
    %30 = arith.addi %c648_i32, %29 : i32
    %31 = cgra.lwi %30 : i32->i32
    %32 = arith.muli %24, %31 : i32
    %33 = arith.index_cast %2 : index to i32
    %34 = arith.muli %33, %c8_i32_3 : i32
    %35 = arith.index_cast %5 : index to i32
    %36 = arith.addi %34, %35 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %c136_i32, %37 : i32
    %39 = cgra.lwi %38 : i32->i32
    %40 = arith.addi %39, %32 : i32
    %41 = arith.index_cast %2 : index to i32
    %42 = arith.muli %41, %c8_i32_3 : i32
    %43 = arith.index_cast %5 : index to i32
    %44 = arith.addi %42, %43 : i32
    %45 = arith.muli %44, %c4_i32 : i32
    %46 = arith.addi %c136_i32, %45 : i32
    cgra.swi %40, %46 : i32, i32
    %47 = arith.addi %14, %c1 : index
    cf.br ^bb4(%47 : index)
  ^bb6:  // pred: ^bb4
    %48 = arith.addi %5, %c1 : index
    cf.br ^bb2(%48 : index)
  ^bb7:  // pred: ^bb2
    %49 = arith.addi %2, %c1 : index
    cf.br ^bb1(%49 : index)
  ^bb8(%50: index):  // 2 preds: ^bb1, ^bb14
    %51 = arith.index_cast %50 : index to i32
    %52 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%51 : i32, %52 : i32], ^bb15, ^bb9(%c0 : index)
  ^bb9(%53: index):  // 2 preds: ^bb8, ^bb13
    %54 = arith.index_cast %53 : index to i32
    %55 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%54 : i32, %55 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %56 = arith.index_cast %50 : index to i32
    %57 = arith.muli %56, %c8_i32 : i32
    %58 = arith.index_cast %53 : index to i32
    %59 = arith.addi %57, %58 : i32
    %60 = arith.muli %59, %c4_i32 : i32
    %61 = arith.addi %c1160_i32, %60 : i32
    %62 = cgra.lwi %61 : i32->i32
    %63 = arith.muli %62, %1 : i32
    %64 = arith.index_cast %50 : index to i32
    %65 = arith.muli %64, %c8_i32 : i32
    %66 = arith.index_cast %53 : index to i32
    %67 = arith.addi %65, %66 : i32
    %68 = arith.muli %67, %c4_i32 : i32
    %69 = arith.addi %c1160_i32, %68 : i32
    cgra.swi %63, %69 : i32, i32
    cf.br ^bb11(%c0 : index)
  ^bb11(%70: index):  // 2 preds: ^bb10, ^bb12
    %71 = arith.index_cast %70 : index to i32
    %72 = arith.index_cast %c8 : index to i32
    cgra.cond_br<ge> [%71 : i32, %72 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %73 = arith.index_cast %50 : index to i32
    %74 = arith.muli %73, %c8_i32_3 : i32
    %75 = arith.index_cast %70 : index to i32
    %76 = arith.addi %74, %75 : i32
    %77 = arith.muli %76, %c4_i32 : i32
    %78 = arith.addi %c136_i32, %77 : i32
    %79 = cgra.lwi %78 : i32->i32
    %80 = arith.index_cast %70 : index to i32
    %81 = arith.muli %80, %c8_i32_0 : i32
    %82 = arith.index_cast %53 : index to i32
    %83 = arith.addi %81, %82 : i32
    %84 = arith.muli %83, %c4_i32 : i32
    %85 = arith.addi %c904_i32, %84 : i32
    %86 = cgra.lwi %85 : i32->i32
    %87 = arith.muli %79, %86 : i32
    %88 = arith.index_cast %50 : index to i32
    %89 = arith.muli %88, %c8_i32 : i32
    %90 = arith.index_cast %53 : index to i32
    %91 = arith.addi %89, %90 : i32
    %92 = arith.muli %91, %c4_i32 : i32
    %93 = arith.addi %c1160_i32, %92 : i32
    %94 = cgra.lwi %93 : i32->i32
    %95 = arith.addi %94, %87 : i32
    %96 = arith.index_cast %50 : index to i32
    %97 = arith.muli %96, %c8_i32 : i32
    %98 = arith.index_cast %53 : index to i32
    %99 = arith.addi %97, %98 : i32
    %100 = arith.muli %99, %c4_i32 : i32
    %101 = arith.addi %c1160_i32, %100 : i32
    cgra.swi %95, %101 : i32, i32
    %102 = arith.addi %70, %c1 : index
    cf.br ^bb11(%102 : index)
  ^bb13:  // pred: ^bb11
    %103 = arith.addi %53, %c1 : index
    cf.br ^bb9(%103 : index)
  ^bb14:  // pred: ^bb9
    %104 = arith.addi %50, %c1 : index
    cf.br ^bb8(%104 : index)
  ^bb15:  // pred: ^bb8
    return
  }
}

