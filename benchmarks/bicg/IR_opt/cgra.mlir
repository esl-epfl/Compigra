module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bicg(%arg0: memref<30x20xf32>, %arg1: memref<20xf32>, %arg2: memref<30xf32>, %arg3: memref<20xf32>, %arg4: memref<30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c30 = arith.constant 30 : index
    %c4_i32 = arith.constant 4 : i32
    %c2808_i32 = arith.constant {BaseAddr = "arg4"} 2808 : i32
    %c2728_i32 = arith.constant {BaseAddr = "arg3"} 2728 : i32
    %c2608_i32 = arith.constant {BaseAddr = "arg2"} 2608 : i32
    %c2528_i32 = arith.constant {BaseAddr = "arg1"} 2528 : i32
    %c20_i32 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 20 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %c20 : index to i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb3(%c0 : index), ^bb2
  ^bb2:  // pred: ^bb1
    %3 = arith.index_cast %0 : index to i32
    %4 = arith.muli %3, %c4_i32 : i32
    %5 = arith.addi %c2528_i32, %4 : i32
    cgra.swi %cst, %5 : f32, i32
    %6 = arith.addi %0, %c1 : index
    cf.br ^bb1(%6 : index)
  ^bb3(%7: index):  // 2 preds: ^bb1, ^bb7
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.index_cast %c30 : index to i32
    cgra.cond_br<ge> [%8 : i32, %9 : i32], ^bb8, ^bb4
  ^bb4:  // pred: ^bb3
    %10 = arith.index_cast %7 : index to i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.addi %c2608_i32, %11 : i32
    cgra.swi %cst, %12 : f32, i32
    cf.br ^bb5(%c0 : index)
  ^bb5(%13: index):  // 2 preds: ^bb4, ^bb6
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.index_cast %c20 : index to i32
    cgra.cond_br<ge> [%14 : i32, %15 : i32], ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    %16 = arith.index_cast %13 : index to i32
    %17 = arith.muli %16, %c4_i32 : i32
    %18 = arith.addi %c2528_i32, %17 : i32
    %19 = cgra.lwi %18 : i32->f32
    %20 = arith.index_cast %7 : index to i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.addi %c2808_i32, %21 : i32
    %23 = cgra.lwi %22 : i32->f32
    %24 = arith.index_cast %7 : index to i32
    %25 = arith.muli %24, %c20_i32 : i32
    %26 = arith.index_cast %13 : index to i32
    %27 = arith.addi %25, %26 : i32
    %28 = arith.muli %27, %c4_i32 : i32
    %29 = arith.addi %c128_i32, %28 : i32
    %30 = cgra.lwi %29 : i32->f32
    %31 = arith.mulf %23, %30 : f32
    %32 = arith.addf %19, %31 : f32
    %33 = arith.index_cast %13 : index to i32
    %34 = arith.muli %33, %c4_i32 : i32
    %35 = arith.addi %c2528_i32, %34 : i32
    cgra.swi %32, %35 : f32, i32
    %36 = arith.index_cast %7 : index to i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %c2608_i32, %37 : i32
    %39 = cgra.lwi %38 : i32->f32
    %40 = arith.index_cast %7 : index to i32
    %41 = arith.muli %40, %c20_i32 : i32
    %42 = arith.index_cast %13 : index to i32
    %43 = arith.addi %41, %42 : i32
    %44 = arith.muli %43, %c4_i32 : i32
    %45 = arith.addi %c128_i32, %44 : i32
    %46 = cgra.lwi %45 : i32->f32
    %47 = arith.index_cast %13 : index to i32
    %48 = arith.muli %47, %c4_i32 : i32
    %49 = arith.addi %c2728_i32, %48 : i32
    %50 = cgra.lwi %49 : i32->f32
    %51 = arith.mulf %46, %50 : f32
    %52 = arith.addf %39, %51 : f32
    %53 = arith.index_cast %7 : index to i32
    %54 = arith.muli %53, %c4_i32 : i32
    %55 = arith.addi %c2608_i32, %54 : i32
    cgra.swi %52, %55 : f32, i32
    %56 = arith.addi %13, %c1 : index
    cf.br ^bb5(%56 : index)
  ^bb7:  // pred: ^bb5
    %57 = arith.addi %7, %c1 : index
    cf.br ^bb3(%57 : index)
  ^bb8:  // pred: ^bb3
    return
  }
}

