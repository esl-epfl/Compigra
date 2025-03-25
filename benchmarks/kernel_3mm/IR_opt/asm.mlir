Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>, %arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>, %arg6: memref<4x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
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
    %8 = arith.addi %7, %c0_i32 : i32
    %9 = arith.addi %8, %5 : i32
    %10 = arith.muli %9, %c4_i32_40 : i32
    %11 = arith.addi %10, %c384_i32 : i32
    %12 = arith.addi %c0_i32_38, %c0_i32_39 {constant = 0 : i32} : i32
    cgra.swi %12, %11 : i32, i32
    %13 = arith.addi %c0_i32_26, %c0_i32_27 {constant = 0 : i32} : i32
    cf.br ^bb4(%13 : i32)
  ^bb4(%14: i32):  // 2 preds: ^bb3, ^bb5
    %15 = arith.addi %c0_i32, %c4_i32 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%14 : i32, %15 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %16 = arith.muli %1, %c4_i32_47 : i32
    %17 = arith.addi %16, %14 : i32
    %18 = arith.muli %17, %c4_i32_40 : i32
    %19 = arith.addi %18, %c128_i32 : i32
    %20 = cgra.lwi %19 : i32->i32
    %21 = arith.muli %14, %c4_i32_46 : i32
    %22 = arith.addi %21, %c0_i32 : i32
    %23 = arith.addi %22, %c0_i32 : i32
    %24 = arith.addi %23, %5 : i32
    %25 = arith.muli %24, %c4_i32_40 : i32
    %26 = arith.addi %25, %c192_i32 : i32
    %27 = cgra.lwi %26 : i32->i32
    %28 = arith.muli %20, %27 : i32
    %29 = arith.muli %1, %c4_i32_43 : i32
    %30 = arith.addi %29, %c0_i32 : i32
    %31 = arith.addi %30, %5 : i32
    %32 = arith.muli %31, %c4_i32_40 : i32
    %33 = arith.addi %32, %c384_i32 : i32
    %34 = cgra.lwi %33 : i32->i32
    %35 = arith.addi %34, %28 : i32
    %36 = arith.muli %1, %c4_i32_43 : i32
    %37 = arith.addi %36, %c0_i32 : i32
    %38 = arith.addi %37, %5 : i32
    %39 = arith.muli %38, %c4_i32_40 : i32
    %40 = arith.addi %39, %c384_i32 : i32
    cgra.swi %35, %40 : i32, i32
    %41 = arith.addi %14, %c1_i32 : i32
    cf.br ^bb4(%41 : i32)
  ^bb6:  // pred: ^bb4
    %42 = arith.addi %5, %c1_i32 : i32
    cf.br ^bb2(%42 : i32)
  ^bb7:  // pred: ^bb2
    %43 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%43 : i32)
  ^bb8(%44: i32):  // 2 preds: ^bb1, ^bb14
    %45 = arith.addi %c0_i32_12, %c4_i32_13 {constant = 4 : i32} : i32
    %46 = arith.addi %c0_i32_22, %c0_i32_23 {constant = 0 : i32} : i32
    %47 = arith.addi %c0_i32_24, %c0_i32_25 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%44 : i32, %45 : i32], ^bb15(%47 : i32), ^bb9(%46 : i32)
  ^bb9(%48: i32):  // 2 preds: ^bb8, ^bb13
    %49 = arith.addi %c0_i32_4, %c4_i32_5 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%48 : i32, %49 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %50 = arith.muli %44, %c4_i32_42 : i32
    %51 = arith.addi %50, %48 : i32
    %52 = arith.muli %51, %c4_i32_40 : i32
    %53 = arith.addi %52, %c448_i32 : i32
    %54 = arith.addi %c0_i32_36, %c0_i32_37 {constant = 0 : i32} : i32
    cgra.swi %54, %53 : i32, i32
    %55 = arith.addi %c0_i32_20, %c0_i32_21 {constant = 0 : i32} : i32
    cf.br ^bb11(%55 : i32)
  ^bb11(%56: i32):  // 2 preds: ^bb10, ^bb12
    %57 = arith.addi %c0_i32_2, %c4_i32_3 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%56 : i32, %57 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %58 = arith.muli %44, %c4_i32_45 : i32
    %59 = arith.addi %58, %c0_i32 : i32
    %60 = arith.addi %59, %56 : i32
    %61 = arith.muli %60, %c4_i32_40 : i32
    %62 = arith.addi %61, %c256_i32 : i32
    %63 = cgra.lwi %62 : i32->i32
    %64 = arith.muli %56, %c4_i32_44 : i32
    %65 = arith.addi %64, %48 : i32
    %66 = arith.muli %65, %c4_i32_40 : i32
    %67 = arith.addi %66, %c320_i32 : i32
    %68 = cgra.lwi %67 : i32->i32
    %69 = arith.muli %63, %68 : i32
    %70 = arith.muli %44, %c4_i32_42 : i32
    %71 = arith.addi %70, %48 : i32
    %72 = arith.muli %71, %c4_i32_40 : i32
    %73 = arith.addi %72, %c448_i32 : i32
    %74 = cgra.lwi %73 : i32->i32
    %75 = arith.addi %74, %69 : i32
    %76 = arith.muli %44, %c4_i32_42 : i32
    %77 = arith.addi %76, %48 : i32
    %78 = arith.muli %77, %c4_i32_40 : i32
    %79 = arith.addi %78, %c448_i32 : i32
    cgra.swi %75, %79 : i32, i32
    %80 = arith.addi %56, %c1_i32 : i32
    cf.br ^bb11(%80 : i32)
  ^bb13:  // pred: ^bb11
    %81 = arith.addi %48, %c1_i32 : i32
    cf.br ^bb9(%81 : i32)
  ^bb14:  // pred: ^bb9
    %82 = arith.addi %44, %c1_i32 : i32
    cf.br ^bb8(%82 : i32)
  ^bb15(%83: i32):  // 2 preds: ^bb8, ^bb21
    %84 = arith.addi %c0_i32_10, %c4_i32_11 {constant = 4 : i32} : i32
    %85 = arith.addi %c0_i32_18, %c0_i32_19 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%83 : i32, %84 : i32], ^bb22, ^bb16(%85 : i32)
  ^bb16(%86: i32):  // 2 preds: ^bb15, ^bb20
    %87 = arith.addi %c0_i32_8, %c4_i32_9 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%86 : i32, %87 : i32], ^bb21, ^bb17
  ^bb17:  // pred: ^bb16
    %88 = arith.muli %83, %c4_i32_41 : i32
    %89 = arith.addi %88, %c0_i32 : i32
    %90 = arith.addi %89, %86 : i32
    %91 = arith.muli %90, %c4_i32_40 : i32
    %92 = arith.addi %91, %c512_i32 : i32
    %93 = arith.addi %c0_i32_34, %c0_i32_35 {constant = 0 : i32} : i32
    cgra.swi %93, %92 : i32, i32
    %94 = arith.addi %c0_i32_16, %c0_i32_17 {constant = 0 : i32} : i32
    cf.br ^bb18(%94 : i32)
  ^bb18(%95: i32):  // 2 preds: ^bb17, ^bb19
    %96 = arith.addi %c0_i32_6, %c4_i32_7 {constant = 4 : i32} : i32
    cgra.cond_br<ge> [%95 : i32, %96 : i32], ^bb20, ^bb19
  ^bb19:  // pred: ^bb18
    %97 = arith.muli %83, %c4_i32_43 : i32
    %98 = arith.addi %97, %95 : i32
    %99 = arith.muli %98, %c4_i32_40 : i32
    %100 = arith.addi %99, %c384_i32 : i32
    %101 = cgra.lwi %100 : i32->i32
    %102 = arith.muli %95, %c4_i32_42 : i32
    %103 = arith.addi %102, %86 : i32
    %104 = arith.muli %103, %c4_i32_40 : i32
    %105 = arith.addi %104, %c448_i32 : i32
    %106 = cgra.lwi %105 : i32->i32
    %107 = arith.muli %101, %106 : i32
    %108 = arith.muli %83, %c4_i32_41 : i32
    %109 = arith.addi %108, %c0_i32 : i32
    %110 = arith.addi %109, %86 : i32
    %111 = arith.muli %110, %c4_i32_40 : i32
    %112 = arith.addi %111, %c512_i32 : i32
    %113 = cgra.lwi %112 : i32->i32
    %114 = arith.addi %113, %107 : i32
    %115 = arith.muli %83, %c4_i32_41 : i32
    %116 = arith.addi %115, %c0_i32 : i32
    %117 = arith.addi %116, %86 : i32
    %118 = arith.muli %117, %c4_i32_40 : i32
    %119 = arith.addi %118, %c512_i32 : i32
    cgra.swi %114, %119 : i32, i32
    %120 = arith.addi %95, %c1_i32 : i32
    cf.br ^bb18(%120 : i32)
  ^bb20:  // pred: ^bb18
    %121 = arith.addi %86, %c1_i32 : i32
    cf.br ^bb16(%121 : i32)
  ^bb21:  // pred: ^bb16
    %122 = arith.addi %83, %c1_i32 : i32
    cf.br ^bb15(%122 : i32)
  ^bb22:  // pred: ^bb15
    return
  }
}

