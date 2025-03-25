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
    %18 = arith.addi %17, %15 : i32
    %19 = arith.muli %18, %c4_i32 : i32
    %20 = arith.addi %19, %c392_i32 : i32
    %21 = cgra.lwi %20 : i32->i32
    %22 = arith.muli %0, %21 : i32
    %23 = arith.muli %15, %c8_i32_26 : i32
    %24 = arith.addi %23, %7 : i32
    %25 = arith.muli %24, %c4_i32 : i32
    %26 = arith.addi %25, %c648_i32 : i32
    %27 = cgra.lwi %26 : i32->i32
    %28 = arith.muli %22, %27 : i32
    %29 = arith.muli %3, %c8_i32_28 : i32
    %30 = arith.addi %29, %7 : i32
    %31 = arith.muli %30, %c4_i32 : i32
    %32 = arith.addi %31, %c136_i32 : i32
    %33 = cgra.lwi %32 : i32->i32
    %34 = arith.addi %33, %28 : i32
    %35 = arith.muli %3, %c8_i32_28 : i32
    %36 = arith.addi %35, %7 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %37, %c136_i32 : i32
    cgra.swi %34, %38 : i32, i32
    %39 = arith.addi %15, %c1_i32 : i32
    cf.br ^bb4(%39 : i32)
  ^bb6:  // pred: ^bb4
    %40 = arith.addi %7, %c1_i32 : i32
    cf.br ^bb2(%40 : i32)
  ^bb7:  // pred: ^bb2
    %41 = arith.addi %3, %c1_i32 : i32
    cf.br ^bb1(%41 : i32)
  ^bb8(%42: i32):  // 2 preds: ^bb1, ^bb14
    %43 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %44 = arith.addi %c0_i32_20, %c8_i32_21 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%42 : i32, %44 : i32], ^bb15, ^bb9(%43 : i32)
  ^bb9(%45: i32):  // 2 preds: ^bb8, ^bb13
    %46 = arith.addi %c0_i32_18, %c8_i32_19 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%45 : i32, %46 : i32], ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %47 = arith.muli %42, %c8_i32_24 : i32
    %48 = arith.addi %47, %45 : i32
    %49 = arith.muli %48, %c4_i32 : i32
    %50 = arith.addi %49, %c1160_i32 : i32
    %51 = cgra.lwi %50 : i32->i32
    %52 = arith.muli %51, %1 : i32
    %53 = arith.muli %42, %c8_i32_24 : i32
    %54 = arith.addi %53, %45 : i32
    %55 = arith.muli %54, %c4_i32 : i32
    %56 = arith.addi %55, %c1160_i32 : i32
    cgra.swi %52, %56 : i32, i32
    %57 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    cf.br ^bb11(%57 : i32)
  ^bb11(%58: i32):  // 2 preds: ^bb10, ^bb12
    %59 = arith.addi %c0_i32_16, %c8_i32_17 {constant = 8 : i32} : i32
    cgra.cond_br<ge> [%58 : i32, %59 : i32], ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %60 = arith.muli %42, %c8_i32_28 : i32
    %61 = arith.addi %60, %58 : i32
    %62 = arith.muli %61, %c4_i32 : i32
    %63 = arith.addi %62, %c136_i32 : i32
    %64 = cgra.lwi %63 : i32->i32
    %65 = arith.muli %58, %c8_i32_25 : i32
    %66 = arith.addi %65, %45 : i32
    %67 = arith.muli %66, %c4_i32 : i32
    %68 = arith.addi %67, %c904_i32 : i32
    %69 = cgra.lwi %68 : i32->i32
    %70 = arith.muli %64, %69 : i32
    %71 = arith.muli %42, %c8_i32_24 : i32
    %72 = arith.addi %71, %45 : i32
    %73 = arith.muli %72, %c4_i32 : i32
    %74 = arith.addi %73, %c1160_i32 : i32
    %75 = cgra.lwi %74 : i32->i32
    %76 = arith.addi %75, %70 : i32
    %77 = arith.muli %42, %c8_i32_24 : i32
    %78 = arith.addi %77, %45 : i32
    %79 = arith.muli %78, %c4_i32 : i32
    %80 = arith.addi %79, %c1160_i32 : i32
    cgra.swi %76, %80 : i32, i32
    %81 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb11(%81 : i32)
  ^bb13:  // pred: ^bb11
    %82 = arith.addi %45, %c1_i32 : i32
    cf.br ^bb9(%82 : i32)
  ^bb14:  // pred: ^bb9
    %83 = arith.addi %42, %c1_i32 : i32
    cf.br ^bb8(%83 : i32)
  ^bb15:  // pred: ^bb8
    return
  }
}

