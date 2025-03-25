module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>, %arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>, %arg6: memref<4x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c4_i32_1 = arith.constant 4 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c4_i32_3 = arith.constant 4 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c4_i32_5 = arith.constant 4 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c4_i32_7 = arith.constant 4 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c4_i32_9 = arith.constant 4 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c4_i32_11 = arith.constant 4 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c4_i32_13 = arith.constant 4 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %c4_i32_15 = arith.constant 4 : i32
    %c0_i32_16 = arith.constant 0 : i32
    %c0_i32_17 = arith.constant 0 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %c0_i32_19 = arith.constant 0 : i32
    %c0_i32_20 = arith.constant 0 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %c0_i32_22 = arith.constant 0 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %c0_i32_24 = arith.constant 0 : i32
    %c0_i32_25 = arith.constant 0 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %c0_i32_27 = arith.constant 0 : i32
    %c0_i32_28 = arith.constant 0 : i32
    %c0_i32_29 = arith.constant 0 : i32
    %c0_i32_30 = arith.constant 0 : i32
    %c0_i32_31 = arith.constant 0 : i32
    %c0_i32_32 = arith.constant 0 : i32
    %c0_i32_33 = arith.constant 0 : i32
    %c0_i32_34 = arith.constant 0 : i32
    %c0_i32_35 = arith.constant 0 : i32
    %c0_i32_36 = arith.constant 0 : i32
    %c0_i32_37 = arith.constant 0 : i32
    %c0_i32_38 = arith.constant 0 : i32
    %c0_i32_39 = arith.constant 0 : i32
    %c4_i32_40 = arith.constant 4 : i32
    %c4_i32_41 = arith.constant {DimProd = 1 : i32, arg = 6 : i32} 4 : i32
    %c512_i32 = arith.constant {BaseAddr = "arg6"} 512 : i32
    %c4_i32_42 = arith.constant {DimProd = 1 : i32, arg = 5 : i32} 4 : i32
    %c448_i32 = arith.constant {BaseAddr = "arg5"} 448 : i32
    %c4_i32_43 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 4 : i32
    %c384_i32 = arith.constant {BaseAddr = "arg4"} 384 : i32
    %c4_i32_44 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 4 : i32
    %c320_i32 = arith.constant {BaseAddr = "arg3"} 320 : i32
    %c4_i32_45 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 4 : i32
    %c256_i32 = arith.constant {BaseAddr = "arg2"} 256 : i32
    %c4_i32_46 = arith.constant {DimProd = 1 : i32, arg = 1 : i32} 4 : i32
    %c192_i32 = arith.constant {BaseAddr = "arg1"} 192 : i32
    %c4_i32_47 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 4 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = arith.addi %c0_i32_32, %c0_i32_33 {constant = 0 : i32} : i32
    cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb7
    %2 = arith.addi %c0_i32_14, %c4_i32_15 {constant = 4 : i32} : i32
    %3 = arith.addi %c0_i32_28, %c0_i32_29 {constant = 0 : i32} : i32
    %4 = arith.addi %c0_i32_30, %c0_i32_31 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb8(%4 : i32), ^bb2(%3 : i32)
  ^bb2(%5: i32):  // 2 preds: ^bb1, ^bb6
    %6 = arith.addi %c0_i32_0, %c4_i32_1 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%5 : i32, %6 : i32], ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %7 = arith.muli %1, %c4_i32_43 : i32
    %8 = arith.addi %7, %5 : i32
    %9 = arith.muli %8, %c4_i32_40 : i32
    %10 = arith.addi %9, %c384_i32 : i32
    %11 = arith.addi %c0_i32_38, %c0_i32_39 {constant = 0 : i32} : i32
    cgra.swi %11, %10 : i32, i32
    %12 = arith.addi %c0_i32_26, %c0_i32_27 {constant = 0 : i32} : i32
    cf.br ^bb4(%12 : i32)
  ^bb4(%13: i32):  // 2 preds: ^bb3, ^bb5
    %14 = arith.addi %c0_i32, %c4_i32 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%13 : i32, %14 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %15 = arith.muli %1, %c4_i32_47 : i32
    %16 = arith.addi %15, %13 : i32
    %17 = arith.muli %16, %c4_i32_40 : i32
    %18 = arith.addi %17, %c128_i32 : i32
    %19 = cgra.lwi %18 : i32->i32
    %20 = arith.muli %13, %c4_i32_46 : i32
    %21 = arith.addi %20, %5 : i32
    %22 = arith.muli %21, %c4_i32_40 : i32
    %23 = arith.addi %22, %c192_i32 : i32
    %24 = cgra.lwi %23 : i32->i32
    %25 = arith.muli %19, %24 : i32
    %26 = arith.muli %1, %c4_i32_43 : i32
    %27 = arith.addi %26, %5 : i32
    %28 = arith.muli %27, %c4_i32_40 : i32
    %29 = arith.addi %28, %c384_i32 : i32
    %30 = cgra.lwi %29 : i32->i32
    %31 = arith.addi %30, %25 : i32
    %32 = arith.muli %1, %c4_i32_43 : i32
    %33 = arith.addi %32, %5 : i32
    %34 = arith.muli %33, %c4_i32_40 : i32
    %35 = arith.addi %34, %c384_i32 : i32
    cgra.swi %31, %35 : i32, i32
    %36 = arith.addi %13, %c1_i32 : i32
    cf.br ^bb4(%36 : i32)
  ^bb6:  // pred: ^bb4
    %37 = arith.addi %5, %c1_i32 : i32
    cf.br ^bb2(%37 : i32)
  ^bb7:  // pred: ^bb2
    %38 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%38 : i32)
  ^bb8(%39: i32):  // 2 preds: ^bb1, ^bb14
    %40 = arith.addi %c0_i32_12, %c4_i32_13 {constant = 4 : i32} : i32
    %41 = arith.addi %c0_i32_22, %c0_i32_23 {constant = 0 : i32} : i32
    %42 = arith.addi %c0_i32_24, %c0_i32_25 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%39 : i32, %40 : i32], ^bb15(%42 : i32), ^bb9(%41 : i32)
  ^bb9(%43: i32):  // 2 preds: ^bb8, ^bb13
    %44 = arith.addi %c0_i32_4, %c4_i32_5 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%43 : i32, %44 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %45 = arith.muli %39, %c4_i32_42 : i32
    %46 = arith.addi %45, %43 : i32
    %47 = arith.muli %46, %c4_i32_40 : i32
    %48 = arith.addi %47, %c448_i32 : i32
    %49 = arith.addi %c0_i32_36, %c0_i32_37 {constant = 0 : i32} : i32
    cgra.swi %49, %48 : i32, i32
    %50 = arith.addi %c0_i32_20, %c0_i32_21 {constant = 0 : i32} : i32
    cf.br ^bb11(%50 : i32)
  ^bb11(%51: i32):  // 2 preds: ^bb10, ^bb12
    %52 = arith.addi %c0_i32_2, %c4_i32_3 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%51 : i32, %52 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %53 = arith.muli %39, %c4_i32_45 : i32
    %54 = arith.addi %53, %51 : i32
    %55 = arith.muli %54, %c4_i32_40 : i32
    %56 = arith.addi %55, %c256_i32 : i32
    %57 = cgra.lwi %56 : i32->i32
    %58 = arith.muli %51, %c4_i32_44 : i32
    %59 = arith.addi %58, %43 : i32
    %60 = arith.muli %59, %c4_i32_40 : i32
    %61 = arith.addi %60, %c320_i32 : i32
    %62 = cgra.lwi %61 : i32->i32
    %63 = arith.muli %57, %62 : i32
    %64 = arith.muli %39, %c4_i32_42 : i32
    %65 = arith.addi %64, %43 : i32
    %66 = arith.muli %65, %c4_i32_40 : i32
    %67 = arith.addi %66, %c448_i32 : i32
    %68 = cgra.lwi %67 : i32->i32
    %69 = arith.addi %68, %63 : i32
    %70 = arith.muli %39, %c4_i32_42 : i32
    %71 = arith.addi %70, %43 : i32
    %72 = arith.muli %71, %c4_i32_40 : i32
    %73 = arith.addi %72, %c448_i32 : i32
    cgra.swi %69, %73 : i32, i32
    %74 = arith.addi %51, %c1_i32 : i32
    cf.br ^bb11(%74 : i32)
  ^bb13:  // pred: ^bb11
    %75 = arith.addi %43, %c1_i32 : i32
    cf.br ^bb9(%75 : i32)
  ^bb14:  // pred: ^bb9
    %76 = arith.addi %39, %c1_i32 : i32
    cf.br ^bb8(%76 : i32)
  ^bb15(%77: i32):  // 2 preds: ^bb8, ^bb21
    %78 = arith.addi %c0_i32_10, %c4_i32_11 {constant = 4 : i32} : i32
    %79 = arith.addi %c0_i32_18, %c0_i32_19 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%77 : i32, %78 : i32], ^bb22, ^bb16(%79 : i32)
  ^bb16(%80: i32):  // 2 preds: ^bb15, ^bb20
    %81 = arith.addi %c0_i32_8, %c4_i32_9 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%80 : i32, %81 : i32], ^bb21, ^bb17
  ^bb17:  // pred: ^bb16
    %82 = arith.muli %77, %c4_i32_41 : i32
    %83 = arith.addi %82, %80 : i32
    %84 = arith.muli %83, %c4_i32_40 : i32
    %85 = arith.addi %84, %c512_i32 : i32
    %86 = arith.addi %c0_i32_34, %c0_i32_35 {constant = 0 : i32} : i32
    cgra.swi %86, %85 : i32, i32
    %87 = arith.addi %c0_i32_16, %c0_i32_17 {constant = 0 : i32} : i32
    cf.br ^bb18(%87 : i32)
  ^bb18(%88: i32):  // 2 preds: ^bb17, ^bb19
    %89 = arith.addi %c0_i32_6, %c4_i32_7 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%88 : i32, %89 : i32], ^bb20, ^bb19
  ^bb19:  // pred: ^bb18
    %90 = arith.muli %77, %c4_i32_43 : i32
    %91 = arith.addi %90, %88 : i32
    %92 = arith.muli %91, %c4_i32_40 : i32
    %93 = arith.addi %92, %c384_i32 : i32
    %94 = cgra.lwi %93 : i32->i32
    %95 = arith.muli %88, %c4_i32_42 : i32
    %96 = arith.addi %95, %80 : i32
    %97 = arith.muli %96, %c4_i32_40 : i32
    %98 = arith.addi %97, %c448_i32 : i32
    %99 = cgra.lwi %98 : i32->i32
    %100 = arith.muli %94, %99 : i32
    %101 = arith.muli %77, %c4_i32_41 : i32
    %102 = arith.addi %101, %80 : i32
    %103 = arith.muli %102, %c4_i32_40 : i32
    %104 = arith.addi %103, %c512_i32 : i32
    %105 = cgra.lwi %104 : i32->i32
    %106 = arith.addi %105, %100 : i32
    %107 = arith.muli %77, %c4_i32_41 : i32
    %108 = arith.addi %107, %80 : i32
    %109 = arith.muli %108, %c4_i32_40 : i32
    %110 = arith.addi %109, %c512_i32 : i32
    cgra.swi %106, %110 : i32, i32
    %111 = arith.addi %88, %c1_i32 : i32
    cf.br ^bb18(%111 : i32)
  ^bb20:  // pred: ^bb18
    %112 = arith.addi %80, %c1_i32 : i32
    cf.br ^bb16(%112 : i32)
  ^bb21:  // pred: ^bb16
    %113 = arith.addi %77, %c1_i32 : i32
    cf.br ^bb15(%113 : i32)
  ^bb22:  // pred: ^bb15
    return
  }
}

