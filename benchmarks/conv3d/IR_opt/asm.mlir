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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv3d(%arg0: memref<3x15x15xf32>, %arg1: memref<2x5x5xf32>, %arg2: memref<2x11x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c2_i32_1 = arith.constant 2 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32_14 = arith.constant 0 : i32
    %c11_i32 = arith.constant 11 : i32
    %c0_i32_15 = arith.constant 0 : i32
    %c11_i32_16 = arith.constant 11 : i32
    %c0_i32_17 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %c5_i32_19 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c11_i32_20 = arith.constant {DimProd = 2 : i32, arg = 2 : i32} 11 : i32
    %c121_i32 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 121 : i32
    %c3028_i32 = arith.constant {BaseAddr = "arg2"} 3028 : i32
    %c5_i32_21 = arith.constant {DimProd = 2 : i32, arg = 1 : i32} 5 : i32
    %c25_i32 = arith.constant {DimProd = 1 : i32, arg = 1 : i32} 25 : i32
    %c2828_i32 = arith.constant {BaseAddr = "arg1"} 2828 : i32
    %c15_i32 = arith.constant {DimProd = 2 : i32, arg = 0 : i32} 15 : i32
    %c225_i32 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 225 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = arith.addi %c0_i32_12, %c0_i32_13 {constant = 0 : i32} : i32
    %c4_i32_22 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c12_i32 = arith.constant 12 : i32
    %c24_i32 = arith.constant 24 : i32
    cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb13
    %2 = arith.addi %c0_i32_0, %c2_i32_1 {constant = 2 : i32} : i32
    %3 = arith.addi %c0_i32_10, %c0_i32_11 {constant = 0 : i32} : i32
    %c24_i32_23 = arith.constant 24 : i32
    cgra.swi %3, %c24_i32_23 : i32, i32 {memLoc = 24 : i32}
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb14, ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb12
    %4 = arith.addi %c0_i32_8, %c0_i32_9 {constant = 0 : i32} : i32
    %5 = arith.addi %c0_i32_15, %c11_i32_16 {constant = 11 : i32} : i32
    %c24_i32_24 = arith.constant 24 : i32
    %6 = cgra.lwi %c24_i32_24 : i32->i32
    cgra.cond_br<ge> [%6 : i32, %5 : i32], ^bb13, ^bb3(%4 : i32)
  ^bb3(%7: i32):  // 2 preds: ^bb2, ^bb11
    %8 = arith.addi %c0_i32_14, %c11_i32 {constant = 11 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb12, ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.muli %1, %c121_i32 : i32
    %c24_i32_25 = arith.constant 24 : i32
    %10 = cgra.lwi %c24_i32_25 : i32->i32
    %11 = arith.muli %10, %c11_i32_20 : i32
    %c0_i32_26 = arith.constant 0 : i32
    cgra.swi %11, %c0_i32_26 : i32, i32 {memLoc = 0 : i32}
    %12 = cgra.lwi %c0_i32_26 : i32->i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.addi %13, %7 : i32
    %15 = arith.muli %14, %c4_i32 : i32
    %16 = arith.addi %15, %c3028_i32 : i32
    cgra.swi %cst, %16 : f32, i32
    %17 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    %c4_i32_27 = arith.constant 4 : i32
    cgra.swi %17, %c4_i32_27 : i32, i32 {memLoc = 4 : i32}
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb10
    %18 = arith.addi %c0_i32, %c2_i32 {constant = 2 : i32} : i32
    %19 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    %c8_i32_28 = arith.constant 8 : i32
    cgra.swi %19, %c8_i32_28 : i32, i32 {memLoc = 8 : i32}
    %c4_i32_29 = arith.constant 4 : i32
    %20 = cgra.lwi %c4_i32_29 : i32->i32
    cgra.cond_br<ge> [%20 : i32, %18 : i32], ^bb11, ^bb6
  ^bb6:  // 2 preds: ^bb5, ^bb9
    %21 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    %c12_i32_30 = arith.constant 12 : i32
    cgra.swi %21, %c12_i32_30 : i32, i32 {memLoc = 12 : i32}
    %22 = arith.addi %c0_i32_18, %c5_i32_19 {constant = 5 : i32} : i32
    %c8_i32_31 = arith.constant 8 : i32
    %23 = cgra.lwi %c8_i32_31 : i32->i32
    cgra.cond_br<ge> [%23 : i32, %22 : i32], ^bb10, ^bb7
  ^bb7:  // 2 preds: ^bb6, ^bb8
    %24 = arith.addi %c0_i32_17, %c5_i32 {constant = 5 : i32} : i32
    %c12_i32_32 = arith.constant 12 : i32
    %25 = cgra.lwi %c12_i32_32 : i32->i32
    cgra.cond_br<ge> [%25 : i32, %24 : i32], ^bb9, ^bb8
  ^bb8:  // pred: ^bb7
    %26 = cgra.lwi %c4_i32_22 : i32->i32
    %27 = arith.addi %1, %26 : i32
    %28 = cgra.lwi %c8_i32 : i32->i32
    %29 = cgra.lwi %c24_i32 : i32->i32
    %30 = arith.addi %29, %28 : i32
    %31 = cgra.lwi %c12_i32 : i32->i32
    %32 = arith.addi %7, %31 : i32
    %33 = arith.muli %27, %c225_i32 : i32
    %34 = arith.muli %30, %c15_i32 : i32
    %35 = arith.addi %33, %34 : i32
    %36 = arith.addi %35, %32 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %37, %c128_i32 : i32
    %39 = cgra.lwi %38 : i32->f32
    %40 = arith.muli %26, %c25_i32 : i32
    %41 = arith.muli %28, %c5_i32_21 : i32
    %42 = arith.addi %40, %41 : i32
    %43 = arith.addi %42, %31 : i32
    %44 = arith.muli %43, %c4_i32 : i32
    %45 = arith.addi %44, %c2828_i32 : i32
    %46 = cgra.lwi %45 : i32->f32
    %47 = arith.mulf %39, %46 : f32
    %48 = arith.muli %1, %c121_i32 : i32
    %49 = arith.muli %29, %c11_i32_20 : i32
    %c16_i32 = arith.constant 16 : i32
    cgra.swi %49, %c16_i32 : i32, i32 {memLoc = 16 : i32}
    %50 = cgra.lwi %c16_i32 : i32->i32
    %51 = arith.addi %48, %50 : i32
    %52 = arith.addi %51, %7 : i32
    %53 = arith.muli %52, %c4_i32 : i32
    %54 = arith.addi %53, %c3028_i32 : i32
    %55 = cgra.lwi %54 : i32->f32
    %56 = arith.addf %55, %47 : f32
    %57 = arith.muli %1, %c121_i32 : i32
    %58 = arith.muli %29, %c11_i32_20 : i32
    %c20_i32 = arith.constant 20 : i32
    cgra.swi %58, %c20_i32 : i32, i32 {memLoc = 20 : i32}
    %59 = cgra.lwi %c20_i32 : i32->i32
    %60 = arith.addi %57, %59 : i32
    %61 = arith.addi %60, %7 : i32
    %62 = arith.muli %61, %c4_i32 : i32
    %63 = arith.addi %62, %c3028_i32 : i32
    cgra.swi %56, %63 : f32, i32
    %64 = arith.addi %31, %c1_i32 : i32
    %c12_i32_33 = arith.constant 12 : i32
    cgra.swi %64, %c12_i32_33 : i32, i32 {memLoc = 12 : i32}
    cf.br ^bb7
  ^bb9:  // pred: ^bb7
    %c8_i32_34 = arith.constant 8 : i32
    %65 = cgra.lwi %c8_i32_34 : i32->i32
    %66 = arith.addi %65, %c1_i32 : i32
    %c8_i32_35 = arith.constant 8 : i32
    cgra.swi %66, %c8_i32_35 : i32, i32 {memLoc = 8 : i32}
    cf.br ^bb6
  ^bb10:  // pred: ^bb6
    %c4_i32_36 = arith.constant 4 : i32
    %67 = cgra.lwi %c4_i32_36 : i32->i32
    %68 = arith.addi %67, %c1_i32 : i32
    %c4_i32_37 = arith.constant 4 : i32
    cgra.swi %68, %c4_i32_37 : i32, i32 {memLoc = 4 : i32}
    cf.br ^bb5
  ^bb11:  // pred: ^bb5
    %69 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb3(%69 : i32)
  ^bb12:  // pred: ^bb3
    %c24_i32_38 = arith.constant 24 : i32
    %70 = cgra.lwi %c24_i32_38 : i32->i32
    %71 = arith.addi %70, %c1_i32 : i32
    %c24_i32_39 = arith.constant 24 : i32
    cgra.swi %71, %c24_i32_39 : i32, i32 {memLoc = 24 : i32}
    cf.br ^bb2
  ^bb13:  // pred: ^bb2
    %72 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%72 : i32)
  ^bb14:  // pred: ^bb1
    return
  }
}

