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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @symm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c10_i32_9 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32_10 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c936_i32 = arith.constant {BaseAddr = "arg4"} 936 : i32
    %c10_i32_11 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 10 : i32
    %c536_i32 = arith.constant {BaseAddr = "arg3"} 536 : i32
    %c10_i32_12 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 10 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c132_i32 : i32->i32
    %1 = cgra.lwi %c128_i32 : i32->i32
    %2 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    %c0_i32_13 = arith.constant 0 : i32
    cgra.swi %2, %c0_i32_13 : i32, i32 {memLoc = 0 : i32}
    %c0_i32_14 = arith.constant 0 : i32
    %c0_i32_15 = arith.constant 0 : i32
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb6
    %3 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %4 = arith.addi %c0_i32_8, %c10_i32_9 {constant = 10 : i32} : i32
    %c0_i32_16 = arith.constant 0 : i32
    %5 = cgra.lwi %c0_i32_16 : i32->i32
    cgra.cond_br<ge> [%5 : i32, %4 : i32], ^bb7, ^bb2(%3 : i32)
  ^bb2(%6: i32):  // 2 preds: ^bb1, ^bb5
    %7 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    %8 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    %c4_i32_17 = arith.constant 4 : i32
    cgra.swi %8, %c4_i32_17 : i32, i32 {memLoc = 4 : i32}
    %9 = arith.addi %c0_i32_7, %c10_i32 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%6 : i32, %9 : i32], ^bb6, ^bb3(%7 : i32)
  ^bb3(%10: i32):  // 2 preds: ^bb2, ^bb4
    %c0_i32_18 = arith.constant 0 : i32
    %11 = cgra.lwi %c0_i32_18 : i32->i32
    cgra.cond_br<ge> [%10 : i32, %11 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %12 = cgra.lwi %c0_i32_15 : i32->i32
    %13 = arith.muli %12, %c10_i32_10 : i32
    %14 = arith.addi %13, %6 : i32
    %15 = arith.muli %14, %c4_i32 : i32
    %16 = arith.addi %15, %c936_i32 : i32
    %17 = cgra.lwi %16 : i32->i32
    %18 = arith.muli %1, %17 : i32
    %19 = arith.muli %12, %c10_i32_11 : i32
    %20 = arith.addi %19, %10 : i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.addi %21, %c536_i32 : i32
    %23 = cgra.lwi %22 : i32->i32
    %24 = arith.muli %18, %23 : i32
    %25 = arith.muli %10, %c10_i32_12 : i32
    %26 = arith.addi %25, %6 : i32
    %27 = arith.muli %26, %c4_i32 : i32
    %28 = arith.addi %27, %c136_i32 : i32
    %29 = cgra.lwi %28 : i32->i32
    %30 = arith.addi %29, %24 : i32
    %31 = arith.muli %10, %c10_i32_12 : i32
    %32 = arith.addi %31, %6 : i32
    %33 = arith.muli %32, %c4_i32 : i32
    %34 = arith.addi %33, %c136_i32 : i32
    cgra.swi %30, %34 : i32, i32
    %35 = arith.muli %10, %c10_i32_10 : i32
    %36 = arith.addi %35, %6 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %37, %c936_i32 : i32
    %39 = cgra.lwi %38 : i32->i32
    %40 = arith.muli %12, %c10_i32_11 : i32
    %41 = arith.addi %40, %10 : i32
    %42 = arith.muli %41, %c4_i32 : i32
    %43 = arith.addi %42, %c536_i32 : i32
    %44 = cgra.lwi %43 : i32->i32
    %45 = arith.muli %39, %44 : i32
    %c4_i32_19 = arith.constant 4 : i32
    %46 = cgra.lwi %c4_i32_19 : i32->i32
    %47 = arith.addi %46, %45 : i32
    %c4_i32_20 = arith.constant 4 : i32
    cgra.swi %47, %c4_i32_20 : i32, i32 {memLoc = 4 : i32}
    %48 = arith.addi %10, %c1_i32 : i32
    cf.br ^bb3(%48 : i32)
  ^bb5:  // pred: ^bb3
    %49 = cgra.lwi %c0_i32_14 : i32->i32
    %50 = arith.muli %49, %c10_i32_12 : i32
    %51 = arith.addi %50, %6 : i32
    %52 = arith.muli %51, %c4_i32 : i32
    %53 = arith.addi %52, %c136_i32 : i32
    %54 = cgra.lwi %53 : i32->i32
    %55 = arith.muli %0, %54 : i32
    %56 = arith.muli %49, %c10_i32_10 : i32
    %57 = arith.addi %56, %6 : i32
    %58 = arith.muli %57, %c4_i32 : i32
    %59 = arith.addi %58, %c936_i32 : i32
    %60 = cgra.lwi %59 : i32->i32
    %61 = arith.muli %1, %60 : i32
    %62 = arith.muli %49, %c10_i32_11 : i32
    %63 = arith.addi %62, %49 : i32
    %64 = arith.muli %63, %c4_i32 : i32
    %65 = arith.addi %64, %c536_i32 : i32
    %66 = cgra.lwi %65 : i32->i32
    %67 = arith.muli %61, %66 : i32
    %68 = arith.addi %55, %67 : i32
    %c4_i32_21 = arith.constant 4 : i32
    %69 = cgra.lwi %c4_i32_21 : i32->i32
    %70 = arith.muli %1, %69 : i32
    %71 = arith.addi %68, %70 : i32
    %72 = arith.muli %49, %c10_i32_12 : i32
    %73 = arith.addi %72, %6 : i32
    %74 = arith.muli %73, %c4_i32 : i32
    %75 = arith.addi %74, %c136_i32 : i32
    cgra.swi %71, %75 : i32, i32
    %76 = arith.addi %6, %c1_i32 : i32
    cf.br ^bb2(%76 : i32)
  ^bb6:  // pred: ^bb2
    %c0_i32_22 = arith.constant 0 : i32
    %77 = cgra.lwi %c0_i32_22 : i32->i32
    %78 = arith.addi %77, %c1_i32 : i32
    %c0_i32_23 = arith.constant 0 : i32
    cgra.swi %78, %c0_i32_23 : i32, i32 {memLoc = 0 : i32}
    cf.br ^bb1
  ^bb7:  // pred: ^bb1
    return
  }
}

