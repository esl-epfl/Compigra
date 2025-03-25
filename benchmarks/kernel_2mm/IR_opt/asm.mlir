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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<8x8xi32>, %arg3: memref<8x8xi32>, %arg4: memref<8x8xi32>, %arg5: memref<8x8xi32>, %arg6: memref<8x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
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
    %c8_i32 = arith.constant 8 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %c8_i32_15 = arith.constant 8 : i32
    %c0_i32_16 = arith.constant 0 : i32
    %c8_i32_17 = arith.constant 8 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %c8_i32_19 = arith.constant 8 : i32
    %c0_i32_20 = arith.constant 0 : i32
    %c8_i32_21 = arith.constant 8 : i32
    %c0_i32_22 = arith.constant 0 : i32
    %c8_i32_23 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32_24 = arith.constant {DimProd = 1 : i32, arg = 6 : i32} 8 : i32
    %c1160_i32 = arith.constant {BaseAddr = "arg6"} 1160 : i32
    %c8_i32_25 = arith.constant {DimProd = 1 : i32, arg = 5 : i32} 8 : i32
    %c904_i32 = arith.constant {BaseAddr = "arg5"} 904 : i32
    %c8_i32_26 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 8 : i32
    %c648_i32 = arith.constant {BaseAddr = "arg4"} 648 : i32
    %c8_i32_27 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 8 : i32
    %c392_i32 = arith.constant {BaseAddr = "arg3"} 392 : i32
    %c8_i32_28 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 8 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = arith.addi %c0_i32_9, %c0_i32_10 {constant = 0 : i32} : i32
    cf.br ^bb1(%2 : i32)
  ^bb1(%3: i32):  // 2 preds: ^bb0, ^bb7
    %4 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    %5 = arith.addi %c0_i32_7, %c0_i32_8 {constant = 0 : i32} : i32
    %6 = arith.addi %c0_i32_22, %c8_i32_23 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%3 : i32, %6 : i32], ^bb8(%5 : i32), ^bb2(%4 : i32)
  ^bb2(%7: i32):  // 2 preds: ^bb1, ^bb6
    %8 = arith.addi %c0_i32_14, %c8_i32_15 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %9 = arith.muli %3, %c8_i32_28 : i32
    %10 = arith.addi %9, %7 : i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.addi %11, %c136_i32 : i32
    %13 = arith.addi %c0_i32_11, %c0_i32_12 {constant = 0 : i32} : i32
    cgra.swi %13, %12 : i32, i32
    %14 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    cf.br ^bb4(%14 : i32)
  ^bb4(%15: i32):  // 2 preds: ^bb3, ^bb5
    %16 = arith.addi %c0_i32_13, %c8_i32 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%15 : i32, %16 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %17 = arith.muli %3, %c8_i32_27 : i32
    %18 = arith.addi %17, %c0_i32 : i32
    %19 = arith.addi %18, %15 : i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.addi %20, %c392_i32 : i32
    %22 = cgra.lwi %21 : i32->i32
    %23 = arith.muli %0, %22 : i32
    %24 = arith.muli %15, %c8_i32_26 : i32
    %25 = arith.addi %24, %c0_i32 : i32
    %26 = arith.addi %25, %c0_i32 : i32
    %27 = arith.addi %26, %7 : i32
    %28 = arith.muli %27, %c4_i32 : i32
    %29 = arith.addi %28, %c648_i32 : i32
    %30 = cgra.lwi %29 : i32->i32
    %31 = arith.muli %23, %30 : i32
    %32 = arith.muli %3, %c8_i32_28 : i32
    %33 = arith.addi %32, %7 : i32
    %34 = arith.muli %33, %c4_i32 : i32
    %35 = arith.addi %34, %c136_i32 : i32
    %36 = cgra.lwi %35 : i32->i32
    %37 = arith.addi %36, %31 : i32
    %38 = arith.muli %3, %c8_i32_28 : i32
    %39 = arith.addi %38, %7 : i32
    %40 = arith.muli %39, %c4_i32 : i32
    %41 = arith.addi %40, %c136_i32 : i32
    cgra.swi %37, %41 : i32, i32
    %42 = arith.addi %15, %c1_i32 : i32
    cf.br ^bb4(%42 : i32)
  ^bb6:  // pred: ^bb4
    %43 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb2(%43 : i32)
  ^bb7:  // pred: ^bb2
    %44 = arith.addi %3, %c1_i32 : i32
    cf.br ^bb1(%44 : i32)
  ^bb8(%45: i32):  // 2 preds: ^bb1, ^bb14
    %46 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %47 = arith.addi %c0_i32_20, %c8_i32_21 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%45 : i32, %47 : i32], ^bb15, ^bb9(%46 : i32)
  ^bb9(%48: i32):  // 2 preds: ^bb8, ^bb13
    %49 = arith.addi %c0_i32_18, %c8_i32_19 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%48 : i32, %49 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %50 = arith.muli %45, %c8_i32_24 : i32
    %51 = arith.addi %50, %c0_i32 : i32
    %52 = arith.addi %51, %48 : i32
    %53 = arith.muli %52, %c4_i32 : i32
    %54 = arith.addi %53, %c1160_i32 : i32
    %55 = cgra.lwi %54 : i32->i32
    %56 = arith.muli %55, %1 : i32
    %57 = arith.muli %45, %c8_i32_24 : i32
    %58 = arith.addi %57, %c0_i32 : i32
    %59 = arith.addi %58, %48 : i32
    %60 = arith.muli %59, %c4_i32 : i32
    %61 = arith.addi %60, %c1160_i32 : i32
    cgra.swi %56, %61 : i32, i32
    %62 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    cf.br ^bb11(%62 : i32)
  ^bb11(%63: i32):  // 2 preds: ^bb10, ^bb12
    %64 = arith.addi %c0_i32_16, %c8_i32_17 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%63 : i32, %64 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %65 = arith.muli %45, %c8_i32_28 : i32
    %66 = arith.addi %65, %c0_i32 : i32
    %67 = arith.addi %66, %63 : i32
    %68 = arith.muli %67, %c4_i32 : i32
    %69 = arith.addi %68, %c136_i32 : i32
    %70 = cgra.lwi %69 : i32->i32
    %71 = arith.muli %63, %c8_i32_25 : i32
    %72 = arith.addi %71, %c0_i32 : i32
    %73 = arith.addi %72, %48 : i32
    %74 = arith.muli %73, %c4_i32 : i32
    %75 = arith.addi %74, %c904_i32 : i32
    %76 = cgra.lwi %75 : i32->i32
    %77 = arith.muli %70, %76 : i32
    %78 = arith.muli %45, %c8_i32_24 : i32
    %79 = arith.addi %78, %c0_i32 : i32
    %80 = arith.addi %79, %48 : i32
    %81 = arith.muli %80, %c4_i32 : i32
    %82 = arith.addi %81, %c1160_i32 : i32
    %83 = cgra.lwi %82 : i32->i32
    %84 = arith.addi %83, %77 : i32
    %85 = arith.muli %45, %c8_i32_24 : i32
    %86 = arith.addi %85, %c0_i32 : i32
    %87 = arith.addi %86, %48 : i32
    %88 = arith.muli %87, %c4_i32 : i32
    %89 = arith.addi %88, %c1160_i32 : i32
    cgra.swi %84, %89 : i32, i32
    %90 = arith.addi %63, %c1_i32 : i32
    cf.br ^bb11(%90 : i32)
  ^bb13:  // pred: ^bb11
    %91 = arith.addi %48, %c1_i32 : i32
    cf.br ^bb9(%91 : i32)
  ^bb14:  // pred: ^bb9
    %92 = arith.addi %45, %c1_i32 : i32
    cf.br ^bb8(%92 : i32)
  ^bb15:  // pred: ^bb8
    return
  }
}

