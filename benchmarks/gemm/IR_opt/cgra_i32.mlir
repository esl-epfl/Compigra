module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %c0_i32_10 = arith.constant 0 : i32
    %c10_i32_11 = arith.constant 10 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c10_i32_13 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32_14 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c936_i32 = arith.constant {BaseAddr = "arg4"} 936 : i32
    %c10_i32_15 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 10 : i32
    %c536_i32 = arith.constant {BaseAddr = "arg3"} 536 : i32
    %c10_i32_16 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 10 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    cf.br ^bb1(%2 : i32)
  ^bb1(%3: i32):  // 2 preds: ^bb0, ^bb8
    %4 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    %5 = arith.addi %c0_i32_12, %c10_i32_13 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%3 : i32, %5 : i32], ^bb9, ^bb2(%4 : i32)
  ^bb2(%6: i32):  // 2 preds: ^bb1, ^bb3
    %7 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %8 = arith.addi %c0_i32_10, %c10_i32_11 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%6 : i32, %8 : i32], ^bb4(%7 : i32), ^bb3
  ^bb3:  // pred: ^bb2
    %9 = arith.muli %3, %c10_i32_16 : i32
    %10 = arith.addi %9, %6 : i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.addi %11, %c136_i32 : i32
    %13 = cgra.lwi %12 : i32->i32
    %14 = arith.muli %13, %1 : i32
    %15 = arith.muli %3, %c10_i32_16 : i32
    %16 = arith.addi %15, %6 : i32
    %17 = arith.muli %16, %c4_i32 : i32
    %18 = arith.addi %17, %c136_i32 : i32
    cgra.swi %14, %18 : i32, i32
    %19 = arith.addi %6, %c1_i32 : i32
    cf.br ^bb2(%19 : i32)
  ^bb4(%20: i32):  // 2 preds: ^bb2, ^bb7
    %21 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    %22 = arith.addi %c0_i32_8, %c10_i32_9 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%20 : i32, %22 : i32], ^bb8, ^bb5(%21 : i32)
  ^bb5(%23: i32):  // 2 preds: ^bb4, ^bb6
    %24 = arith.addi %c0_i32_7, %c10_i32 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%23 : i32, %24 : i32], ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    %25 = arith.muli %3, %c10_i32_15 : i32
    %26 = arith.addi %25, %20 : i32
    %27 = arith.muli %26, %c4_i32 : i32
    %28 = arith.addi %27, %c536_i32 : i32
    %29 = cgra.lwi %28 : i32->i32
    %30 = arith.muli %0, %29 : i32
    %31 = arith.muli %20, %c10_i32_14 : i32
    %32 = arith.addi %31, %23 : i32
    %33 = arith.muli %32, %c4_i32 : i32
    %34 = arith.addi %33, %c936_i32 : i32
    %35 = cgra.lwi %34 : i32->i32
    %36 = arith.muli %30, %35 : i32
    %37 = arith.muli %3, %c10_i32_16 : i32
    %38 = arith.addi %37, %23 : i32
    %39 = arith.muli %38, %c4_i32 : i32
    %40 = arith.addi %39, %c136_i32 : i32
    %41 = cgra.lwi %40 : i32->i32
    %42 = arith.addi %41, %36 : i32
    %43 = arith.muli %3, %c10_i32_16 : i32
    %44 = arith.addi %43, %23 : i32
    %45 = arith.muli %44, %c4_i32 : i32
    %46 = arith.addi %45, %c136_i32 : i32
    cgra.swi %42, %46 : i32, i32
    %47 = arith.addi %23, %c1_i32 : i32
    cf.br ^bb5(%47 : i32)
  ^bb7:  // pred: ^bb5
    %48 = arith.addi %20, %c1_i32 : i32
    cf.br ^bb4(%48 : i32)
  ^bb8:  // pred: ^bb4
    %49 = arith.addi %3, %c1_i32 : i32
    cf.br ^bb1(%49 : i32)
  ^bb9:  // pred: ^bb1
    return
  }
}

