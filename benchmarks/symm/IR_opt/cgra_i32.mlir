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
    cf.br ^bb1(%2 : i32)
  ^bb1(%3: i32):  // 2 preds: ^bb0, ^bb6
    %4 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    %5 = arith.addi %c0_i32_8, %c10_i32_9 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%3 : i32, %5 : i32], ^bb7, ^bb2(%4 : i32)
  ^bb2(%6: i32):  // 2 preds: ^bb1, ^bb5
    %7 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    %8 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    %9 = arith.addi %c0_i32_7, %c10_i32 {constant = 10 : i32} : i32
    cgra.cond_br<ge> [%6 : i32, %9 : i32], ^bb6, ^bb3(%7, %8 : i32, i32)
  ^bb3(%10: i32, %11: i32):  // 2 preds: ^bb2, ^bb4
    cgra.cond_br<ge> [%10 : i32, %3 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %12 = arith.muli %3, %c10_i32_10 : i32
    %13 = arith.addi %12, %6 : i32
    %14 = arith.muli %13, %c4_i32 : i32
    %15 = arith.addi %14, %c936_i32 : i32
    %16 = cgra.lwi %15 : i32->i32
    %17 = arith.muli %1, %16 : i32
    %18 = arith.muli %3, %c10_i32_11 : i32
    %19 = arith.addi %18, %10 : i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.addi %20, %c536_i32 : i32
    %22 = cgra.lwi %21 : i32->i32
    %23 = arith.muli %17, %22 : i32
    %24 = arith.muli %10, %c10_i32_12 : i32
    %25 = arith.addi %24, %6 : i32
    %26 = arith.muli %25, %c4_i32 : i32
    %27 = arith.addi %26, %c136_i32 : i32
    %28 = cgra.lwi %27 : i32->i32
    %29 = arith.addi %28, %23 : i32
    %30 = arith.muli %10, %c10_i32_12 : i32
    %31 = arith.addi %30, %6 : i32
    %32 = arith.muli %31, %c4_i32 : i32
    %33 = arith.addi %32, %c136_i32 : i32
    cgra.swi %29, %33 : i32, i32
    %34 = arith.muli %10, %c10_i32_10 : i32
    %35 = arith.addi %34, %6 : i32
    %36 = arith.muli %35, %c4_i32 : i32
    %37 = arith.addi %36, %c936_i32 : i32
    %38 = cgra.lwi %37 : i32->i32
    %39 = arith.muli %3, %c10_i32_11 : i32
    %40 = arith.addi %39, %10 : i32
    %41 = arith.muli %40, %c4_i32 : i32
    %42 = arith.addi %41, %c536_i32 : i32
    %43 = cgra.lwi %42 : i32->i32
    %44 = arith.muli %38, %43 : i32
    %45 = arith.addi %11, %44 : i32
    %46 = arith.addi %10, %c1_i32 : i32
    cf.br ^bb3(%46, %45 : i32, i32)
  ^bb5:  // pred: ^bb3
    %47 = arith.muli %3, %c10_i32_12 : i32
    %48 = arith.addi %47, %6 : i32
    %49 = arith.muli %48, %c4_i32 : i32
    %50 = arith.addi %49, %c136_i32 : i32
    %51 = cgra.lwi %50 : i32->i32
    %52 = arith.muli %0, %51 : i32
    %53 = arith.muli %3, %c10_i32_10 : i32
    %54 = arith.addi %53, %6 : i32
    %55 = arith.muli %54, %c4_i32 : i32
    %56 = arith.addi %55, %c936_i32 : i32
    %57 = cgra.lwi %56 : i32->i32
    %58 = arith.muli %1, %57 : i32
    %59 = arith.muli %3, %c10_i32_11 : i32
    %60 = arith.addi %59, %3 : i32
    %61 = arith.muli %60, %c4_i32 : i32
    %62 = arith.addi %61, %c536_i32 : i32
    %63 = cgra.lwi %62 : i32->i32
    %64 = arith.muli %58, %63 : i32
    %65 = arith.addi %52, %64 : i32
    %66 = arith.muli %1, %11 : i32
    %67 = arith.addi %65, %66 : i32
    %68 = arith.muli %3, %c10_i32_12 : i32
    %69 = arith.addi %68, %6 : i32
    %70 = arith.muli %69, %c4_i32 : i32
    %71 = arith.addi %70, %c136_i32 : i32
    cgra.swi %67, %71 : i32, i32
    %72 = arith.addi %6, %c1_i32 : i32
    cf.br ^bb2(%72 : i32)
  ^bb6:  // pred: ^bb2
    %73 = arith.addi %3, %c1_i32 : i32
    cf.br ^bb1(%73 : i32)
  ^bb7:  // pred: ^bb1
    return
  }
}

