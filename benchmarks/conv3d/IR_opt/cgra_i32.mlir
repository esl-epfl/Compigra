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
    cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb13
    %2 = arith.addi %c0_i32_0, %c2_i32_1 {constant = 2 : i32} : i32
    %3 = arith.addi %c0_i32_10, %c0_i32_11 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb14, ^bb2(%3 : i32)
  ^bb2(%4: i32):  // 2 preds: ^bb1, ^bb12
    %5 = arith.addi %c0_i32_8, %c0_i32_9 {constant = 0 : i32} : i32
    %6 = arith.addi %c0_i32_15, %c11_i32_16 {constant = 11 : i32} : i32
    cgra.cond_br<ge> [%4 : i32, %6 : i32], ^bb13, ^bb3(%5 : i32)
  ^bb3(%7: i32):  // 2 preds: ^bb2, ^bb11
    %8 = arith.addi %c0_i32_14, %c11_i32 {constant = 11 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb12, ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.muli %1, %c121_i32 : i32
    %10 = arith.muli %4, %c11_i32_20 : i32
    %11 = arith.addi %9, %10 : i32
    %12 = arith.addi %11, %7 : i32
    %13 = arith.muli %12, %c4_i32 : i32
    %14 = arith.addi %13, %c3028_i32 : i32
    cgra.swi %cst, %14 : f32, i32
    %15 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    cf.br ^bb5(%15 : i32)
  ^bb5(%16: i32):  // 2 preds: ^bb4, ^bb10
    %17 = arith.addi %c0_i32, %c2_i32 {constant = 2 : i32} : i32
    %18 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%16 : i32, %17 : i32], ^bb11, ^bb6(%18 : i32)
  ^bb6(%19: i32):  // 2 preds: ^bb5, ^bb9
    %20 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    %21 = arith.addi %c0_i32_18, %c5_i32_19 {constant = 5 : i32} : i32
    cgra.cond_br<ge> [%19 : i32, %21 : i32], ^bb10, ^bb7(%20 : i32)
  ^bb7(%22: i32):  // 2 preds: ^bb6, ^bb8
    %23 = arith.addi %c0_i32_17, %c5_i32 {constant = 5 : i32} : i32
    cgra.cond_br<ge> [%22 : i32, %23 : i32], ^bb9, ^bb8
  ^bb8:  // pred: ^bb7
    %24 = arith.addi %1, %16 : i32
    %25 = arith.addi %4, %19 : i32
    %26 = arith.addi %7, %22 : i32
    %27 = arith.muli %24, %c225_i32 : i32
    %28 = arith.muli %25, %c15_i32 : i32
    %29 = arith.addi %27, %28 : i32
    %30 = arith.addi %29, %26 : i32
    %31 = arith.muli %30, %c4_i32 : i32
    %32 = arith.addi %31, %c128_i32 : i32
    %33 = cgra.lwi %32 : i32->f32
    %34 = arith.muli %16, %c25_i32 : i32
    %35 = arith.muli %19, %c5_i32_21 : i32
    %36 = arith.addi %34, %35 : i32
    %37 = arith.addi %36, %22 : i32
    %38 = arith.muli %37, %c4_i32 : i32
    %39 = arith.addi %38, %c2828_i32 : i32
    %40 = cgra.lwi %39 : i32->f32
    %41 = arith.mulf %33, %40 : f32
    %42 = arith.muli %1, %c121_i32 : i32
    %43 = arith.muli %4, %c11_i32_20 : i32
    %44 = arith.addi %42, %43 : i32
    %45 = arith.addi %44, %7 : i32
    %46 = arith.muli %45, %c4_i32 : i32
    %47 = arith.addi %46, %c3028_i32 : i32
    %48 = cgra.lwi %47 : i32->f32
    %49 = arith.addf %48, %41 : f32
    %50 = arith.muli %1, %c121_i32 : i32
    %51 = arith.muli %4, %c11_i32_20 : i32
    %52 = arith.addi %50, %51 : i32
    %53 = arith.addi %52, %7 : i32
    %54 = arith.muli %53, %c4_i32 : i32
    %55 = arith.addi %54, %c3028_i32 : i32
    cgra.swi %49, %55 : f32, i32
    %56 = arith.addi %22, %c1_i32 : i32
    cf.br ^bb7(%56 : i32)
  ^bb9:  // pred: ^bb7
    %57 = arith.addi %19, %c1_i32 : i32
    cf.br ^bb6(%57 : i32)
  ^bb10:  // pred: ^bb6
    %58 = arith.addi %16, %c1_i32 : i32
    cf.br ^bb5(%58 : i32)
  ^bb11:  // pred: ^bb5
    %59 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb3(%59 : i32)
  ^bb12:  // pred: ^bb3
    %60 = arith.addi %4, %c1_i32 : i32
    cf.br ^bb2(%60 : i32)
  ^bb13:  // pred: ^bb2
    %61 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%61 : i32)
  ^bb14:  // pred: ^bb1
    return
  }
}

