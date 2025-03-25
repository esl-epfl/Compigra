module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c936_i32 = arith.constant {BaseAddr = "arg4"} 936 : i32
    %c10_i32_0 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 10 : i32
    %c536_i32 = arith.constant {BaseAddr = "arg3"} 536 : i32
    %c10_i32_1 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 10 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb8
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%3 : i32, %4 : i32], ^bb9, ^bb2(%c0 : index)
  ^bb2(%5: index):  // 2 preds: ^bb1, ^bb3
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%6 : i32, %7 : i32], ^bb4(%c0 : index), ^bb3
  ^bb3:  // pred: ^bb2
    %8 = arith.index_cast %2 : index to i32
    %9 = arith.muli %8, %c10_i32_1 : i32
    %10 = arith.index_cast %5 : index to i32
    %11 = arith.addi %9, %10 : i32
    %12 = arith.muli %11, %c4_i32 : i32
    %13 = arith.addi %c136_i32, %12 : i32
    %14 = cgra.lwi %13 : i32->i32
    %15 = arith.muli %14, %1 : i32
    %16 = arith.index_cast %2 : index to i32
    %17 = arith.muli %16, %c10_i32_1 : i32
    %18 = arith.index_cast %5 : index to i32
    %19 = arith.addi %17, %18 : i32
    %20 = arith.muli %19, %c4_i32 : i32
    %21 = arith.addi %c136_i32, %20 : i32
    cgra.swi %15, %21 : i32, i32
    %22 = arith.addi %5, %c1 : index
    cf.br ^bb2(%22 : index)
  ^bb4(%23: index):  // 2 preds: ^bb2, ^bb7
    %24 = arith.index_cast %23 : index to i32
    %25 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%24 : i32, %25 : i32], ^bb8, ^bb5(%c0 : index)
  ^bb5(%26: index):  // 2 preds: ^bb4, ^bb6
    %27 = arith.index_cast %26 : index to i32
    %28 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%27 : i32, %28 : i32], ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    %29 = arith.index_cast %2 : index to i32
    %30 = arith.muli %29, %c10_i32_0 : i32
    %31 = arith.index_cast %23 : index to i32
    %32 = arith.addi %30, %31 : i32
    %33 = arith.muli %32, %c4_i32 : i32
    %34 = arith.addi %c536_i32, %33 : i32
    %35 = cgra.lwi %34 : i32->i32
    %36 = arith.muli %0, %35 : i32
    %37 = arith.index_cast %23 : index to i32
    %38 = arith.muli %37, %c10_i32 : i32
    %39 = arith.index_cast %26 : index to i32
    %40 = arith.addi %38, %39 : i32
    %41 = arith.muli %40, %c4_i32 : i32
    %42 = arith.addi %c936_i32, %41 : i32
    %43 = cgra.lwi %42 : i32->i32
    %44 = arith.muli %36, %43 : i32
    %45 = arith.index_cast %2 : index to i32
    %46 = arith.muli %45, %c10_i32_1 : i32
    %47 = arith.index_cast %26 : index to i32
    %48 = arith.addi %46, %47 : i32
    %49 = arith.muli %48, %c4_i32 : i32
    %50 = arith.addi %c136_i32, %49 : i32
    %51 = cgra.lwi %50 : i32->i32
    %52 = arith.addi %51, %44 : i32
    %53 = arith.index_cast %2 : index to i32
    %54 = arith.muli %53, %c10_i32_1 : i32
    %55 = arith.index_cast %26 : index to i32
    %56 = arith.addi %54, %55 : i32
    %57 = arith.muli %56, %c4_i32 : i32
    %58 = arith.addi %c136_i32, %57 : i32
    cgra.swi %52, %58 : i32, i32
    %59 = arith.addi %26, %c1 : index
    cf.br ^bb5(%59 : index)
  ^bb7:  // pred: ^bb5
    %60 = arith.addi %23, %c1 : index
    cf.br ^bb4(%60 : index)
  ^bb8:  // pred: ^bb4
    %61 = arith.addi %2, %c1 : index
    cf.br ^bb1(%61 : index)
  ^bb9:  // pred: ^bb1
    return
  }
}

