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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bicg(%arg0: memref<30x20xf32>, %arg1: memref<20xf32>, %arg2: memref<30xf32>, %arg3: memref<20xf32>, %arg4: memref<30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c20_i32_1 = arith.constant 20 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32_8 = arith.constant 0 : i32
    %c30_i32 = arith.constant 30 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2808_i32 = arith.constant {BaseAddr = "arg4"} 2808 : i32
    %c2728_i32 = arith.constant {BaseAddr = "arg3"} 2728 : i32
    %c2608_i32 = arith.constant {BaseAddr = "arg2"} 2608 : i32
    %c2528_i32 = arith.constant {BaseAddr = "arg1"} 2528 : i32
    %c20_i32_9 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 20 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.addi %c0_i32_0, %c20_i32_1 {constant = 20 : i32} : i32
    %3 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb3(%3 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %4 = arith.muli %1, %c4_i32 : i32
    %5 = arith.addi %4, %c2528_i32 : i32
    cgra.swi %cst, %5 : f32, i32
    %6 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%6 : i32)
  ^bb3(%7: i32):  // 2 preds: ^bb1, ^bb7
    %8 = arith.addi %c0_i32_8, %c30_i32 {constant = 30 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb8, ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.muli %7, %c4_i32 : i32
    %10 = arith.addi %9, %c2608_i32 : i32
    cgra.swi %cst, %10 : f32, i32
    %11 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cf.br ^bb5(%11 : i32)
  ^bb5(%12: i32):  // 2 preds: ^bb4, ^bb6
    %13 = arith.addi %c0_i32, %c20_i32 {constant = 20 : i32} : i32
    cgra.cond_br<ge> [%12 : i32, %13 : i32], ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    %14 = arith.muli %12, %c4_i32 : i32
    %15 = arith.addi %14, %c2528_i32 : i32
    %16 = cgra.lwi %15 : i32->f32
    %17 = arith.muli %7, %c4_i32 : i32
    %18 = arith.addi %17, %c2808_i32 : i32
    %19 = cgra.lwi %18 : i32->f32
    %20 = arith.muli %7, %c20_i32_9 : i32
    %21 = arith.addi %20, %12 : i32
    %22 = arith.muli %21, %c4_i32 : i32
    %23 = arith.addi %22, %c128_i32 : i32
    %24 = cgra.lwi %23 : i32->f32
    %25 = arith.mulf %19, %24 : f32
    %26 = arith.addf %16, %25 : f32
    %27 = arith.muli %12, %c4_i32 : i32
    %28 = arith.addi %27, %c2528_i32 : i32
    cgra.swi %26, %28 : f32, i32
    %29 = arith.muli %7, %c4_i32 : i32
    %30 = arith.addi %29, %c2608_i32 : i32
    %31 = cgra.lwi %30 : i32->f32
    %32 = arith.muli %7, %c20_i32_9 : i32
    %33 = arith.addi %32, %12 : i32
    %34 = arith.muli %33, %c4_i32 : i32
    %35 = arith.addi %34, %c128_i32 : i32
    %36 = cgra.lwi %35 : i32->f32
    %37 = arith.muli %12, %c4_i32 : i32
    %38 = arith.addi %37, %c2728_i32 : i32
    %39 = cgra.lwi %38 : i32->f32
    %40 = arith.mulf %36, %39 : f32
    %41 = arith.addf %31, %40 : f32
    %42 = arith.muli %7, %c4_i32 : i32
    %43 = arith.addi %42, %c2608_i32 : i32
    cgra.swi %41, %43 : f32, i32
    %44 = arith.addi %12, %c1_i32 : i32
    cf.br ^bb5(%44 : i32)
  ^bb7:  // pred: ^bb5
    %45 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb3(%45 : i32)
  ^bb8:  // pred: ^bb3
    return
  }
}

