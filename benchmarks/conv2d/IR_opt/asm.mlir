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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c756_i32 = arith.constant {BaseAddr = "arg4"} 756 : i32
    %c3_i32 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 3 : i32
    %c720_i32 = arith.constant {BaseAddr = "arg3"} 720 : i32
    %c12_i32 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 12 : i32
    %c144_i32 = arith.constant {BaseAddr = "arg2"} 144 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg1"} 136 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %c132_i32 = arith.constant 132 : i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = cgra.lwi %c136_i32 : i32->i32
    %c8_i32 = arith.constant 8 : i32
    cgra.swi %2, %c8_i32 : i32, i32 {memLoc = 8 : i32}
    %c140_i32 = arith.constant 140 : i32
    %3 = cgra.lwi %c140_i32 : i32->i32
    %c12_i32_7 = arith.constant 12 : i32
    cgra.swi %3, %c12_i32_7 : i32, i32 {memLoc = 12 : i32}
    %4 = arith.subi %0, %2 : i32
    %5 = arith.subi %1, %3 : i32
    %6 = arith.addi %4, %c1_i32 : i32
    %c0_i32_8 = arith.constant 0 : i32
    cgra.swi %6, %c0_i32_8 : i32, i32 {memLoc = 0 : i32}
    %7 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    %c24_i32 = arith.constant 24 : i32
    cgra.swi %7, %c24_i32 : i32, i32 {memLoc = 24 : i32}
    %c16_i32 = arith.constant 16 : i32
    %c20_i32 = arith.constant 20 : i32
    %c24_i32_9 = arith.constant 24 : i32
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb10
    %c0_i32_10 = arith.constant 0 : i32
    %8 = cgra.lwi %c0_i32_10 : i32->i32
    %c24_i32_11 = arith.constant 24 : i32
    %9 = cgra.lwi %c24_i32_11 : i32->i32
    cgra.cond_br<ge> [%9 : i32, %8 : i32], ^bb11, ^bb2
  ^bb2:  // pred: ^bb1
    %10 = arith.addi %5, %c1_i32 : i32
    %c4_i32_12 = arith.constant 4 : i32
    cgra.swi %10, %c4_i32_12 : i32, i32 {memLoc = 4 : i32}
    %11 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    cf.br ^bb3(%11 : i32)
  ^bb3(%12: i32):  // 2 preds: ^bb2, ^bb9
    %c4_i32_13 = arith.constant 4 : i32
    %13 = cgra.lwi %c4_i32_13 : i32->i32
    cgra.cond_br<ge> [%12 : i32, %13 : i32], ^bb10, ^bb4
  ^bb4:  // pred: ^bb3
    %c24_i32_14 = arith.constant 24 : i32
    %14 = cgra.lwi %c24_i32_14 : i32->i32
    %15 = arith.muli %14, %c10_i32 : i32
    %16 = arith.addi %15, %c0_i32 : i32
    %17 = arith.addi %16, %12 : i32
    %18 = arith.muli %17, %c4_i32 : i32
    %19 = arith.addi %18, %c756_i32 : i32
    cgra.swi %cst, %19 : f32, i32
    %20 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %c16_i32_15 = arith.constant 16 : i32
    cgra.swi %20, %c16_i32_15 : i32, i32 {memLoc = 16 : i32}
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb8
    %21 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    %c20_i32_16 = arith.constant 20 : i32
    cgra.swi %21, %c20_i32_16 : i32, i32 {memLoc = 20 : i32}
    %c8_i32_17 = arith.constant 8 : i32
    %22 = cgra.lwi %c8_i32_17 : i32->i32
    %c16_i32_18 = arith.constant 16 : i32
    %23 = cgra.lwi %c16_i32_18 : i32->i32
    cgra.cond_br<ge> [%23 : i32, %22 : i32], ^bb9, ^bb6
  ^bb6:  // 2 preds: ^bb5, ^bb7
    %c12_i32_19 = arith.constant 12 : i32
    %24 = cgra.lwi %c12_i32_19 : i32->i32
    %c20_i32_20 = arith.constant 20 : i32
    %25 = cgra.lwi %c20_i32_20 : i32->i32
    cgra.cond_br<ge> [%25 : i32, %24 : i32], ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %26 = cgra.lwi %c16_i32 : i32->i32
    %27 = cgra.lwi %c24_i32_9 : i32->i32
    %28 = arith.addi %27, %26 : i32
    %29 = cgra.lwi %c20_i32 : i32->i32
    %30 = arith.addi %12, %29 : i32
    %31 = arith.muli %28, %c12_i32 : i32
    %32 = arith.addi %31, %30 : i32
    %33 = arith.muli %32, %c4_i32 : i32
    %34 = arith.addi %33, %c144_i32 : i32
    %35 = cgra.lwi %34 : i32->f32
    %36 = arith.muli %26, %c3_i32 : i32
    %37 = arith.addi %36, %29 : i32
    %38 = arith.muli %37, %c4_i32 : i32
    %39 = arith.addi %38, %c720_i32 : i32
    %40 = cgra.lwi %39 : i32->f32
    %41 = arith.mulf %35, %40 : f32
    %42 = arith.muli %27, %c10_i32 : i32
    %43 = arith.addi %42, %c0_i32 : i32
    %44 = arith.addi %43, %12 : i32
    %45 = arith.muli %44, %c4_i32 : i32
    %46 = arith.addi %45, %c756_i32 : i32
    %47 = cgra.lwi %46 : i32->f32
    %48 = arith.addf %47, %41 : f32
    %49 = arith.muli %27, %c10_i32 : i32
    %50 = arith.addi %49, %c0_i32 : i32
    %51 = arith.addi %50, %12 : i32
    %52 = arith.muli %51, %c4_i32 : i32
    %53 = arith.addi %52, %c756_i32 : i32
    cgra.swi %48, %53 : f32, i32
    %54 = arith.addi %29, %c1_i32 : i32
    %c20_i32_21 = arith.constant 20 : i32
    cgra.swi %54, %c20_i32_21 : i32, i32 {memLoc = 20 : i32}
    cf.br ^bb6
  ^bb8:  // pred: ^bb6
    %c16_i32_22 = arith.constant 16 : i32
    %55 = cgra.lwi %c16_i32_22 : i32->i32
    %56 = arith.addi %55, %c1_i32 : i32
    %c16_i32_23 = arith.constant 16 : i32
    cgra.swi %56, %c16_i32_23 : i32, i32 {memLoc = 16 : i32}
    cf.br ^bb5
  ^bb9:  // pred: ^bb5
    %57 = arith.addi %12, %c1_i32 : i32
    cf.br ^bb3(%57 : i32)
  ^bb10:  // pred: ^bb3
    %c24_i32_24 = arith.constant 24 : i32
    %58 = cgra.lwi %c24_i32_24 : i32->i32
    %59 = arith.addi %58, %c1_i32 : i32
    %c24_i32_25 = arith.constant 24 : i32
    cgra.swi %59, %c24_i32_25 : i32, i32 {memLoc = 24 : i32}
    cf.br ^bb1
  ^bb11:  // pred: ^bb1
    return
  }
}

