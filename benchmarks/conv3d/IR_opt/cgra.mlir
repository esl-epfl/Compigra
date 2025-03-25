module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv3d(%arg0: memref<3x15x15xf32>, %arg1: memref<2x5x5xf32>, %arg2: memref<2x11x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c11 = arith.constant 11 : index
    %c5 = arith.constant 5 : index
    %c4_i32 = arith.constant 4 : i32
    %c11_i32 = arith.constant {DimProd = 2 : i32, arg = 2 : i32} 11 : i32
    %c121_i32 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 121 : i32
    %c3028_i32 = arith.constant {BaseAddr = "arg2"} 3028 : i32
    %c5_i32 = arith.constant {DimProd = 2 : i32, arg = 1 : i32} 5 : i32
    %c25_i32 = arith.constant {DimProd = 1 : i32, arg = 1 : i32} 25 : i32
    %c2828_i32 = arith.constant {BaseAddr = "arg1"} 2828 : i32
    %c15_i32 = arith.constant {DimProd = 2 : i32, arg = 0 : i32} 15 : i32
    %c225_i32 = arith.constant {DimProd = 1 : i32, arg = 0 : i32} 225 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb13
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %c2 : index to i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb14, ^bb2(%c0 : index)
  ^bb2(%3: index):  // 2 preds: ^bb1, ^bb12
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.index_cast %c11 : index to i32
    cgra.cond_br<ge> [%4 : i32, %5 : i32], ^bb13, ^bb3(%c0 : index)
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb11
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.index_cast %c11 : index to i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb12, ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.index_cast %0 : index to i32
    %10 = arith.muli %9, %c121_i32 : i32
    %11 = arith.index_cast %3 : index to i32
    %12 = arith.muli %11, %c11_i32 : i32
    %13 = arith.addi %10, %12 : i32
    %14 = arith.index_cast %6 : index to i32
    %15 = arith.addi %13, %14 : i32
    %16 = arith.muli %15, %c4_i32 : i32
    %17 = arith.addi %c3028_i32, %16 : i32
    cgra.swi %cst, %17 : f32, i32
    cf.br ^bb5(%c0 : index)
  ^bb5(%18: index):  // 2 preds: ^bb4, ^bb10
    %19 = arith.index_cast %18 : index to i32
    %20 = arith.index_cast %c2 : index to i32
    cgra.cond_br<ge> [%19 : i32, %20 : i32], ^bb11, ^bb6(%c0 : index)
  ^bb6(%21: index):  // 2 preds: ^bb5, ^bb9
    %22 = arith.index_cast %21 : index to i32
    %23 = arith.index_cast %c5 : index to i32
    cgra.cond_br<ge> [%22 : i32, %23 : i32], ^bb10, ^bb7(%c0 : index)
  ^bb7(%24: index):  // 2 preds: ^bb6, ^bb8
    %25 = arith.index_cast %24 : index to i32
    %26 = arith.index_cast %c5 : index to i32
    cgra.cond_br<ge> [%25 : i32, %26 : i32], ^bb9, ^bb8
  ^bb8:  // pred: ^bb7
    %27 = arith.addi %0, %18 : index
    %28 = arith.addi %3, %21 : index
    %29 = arith.addi %6, %24 : index
    %30 = arith.index_cast %27 : index to i32
    %31 = arith.muli %30, %c225_i32 : i32
    %32 = arith.index_cast %28 : index to i32
    %33 = arith.muli %32, %c15_i32 : i32
    %34 = arith.addi %31, %33 : i32
    %35 = arith.index_cast %29 : index to i32
    %36 = arith.addi %34, %35 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %c128_i32, %37 : i32
    %39 = cgra.lwi %38 : i32->f32
    %40 = arith.index_cast %18 : index to i32
    %41 = arith.muli %40, %c25_i32 : i32
    %42 = arith.index_cast %21 : index to i32
    %43 = arith.muli %42, %c5_i32 : i32
    %44 = arith.addi %41, %43 : i32
    %45 = arith.index_cast %24 : index to i32
    %46 = arith.addi %44, %45 : i32
    %47 = arith.muli %46, %c4_i32 : i32
    %48 = arith.addi %c2828_i32, %47 : i32
    %49 = cgra.lwi %48 : i32->f32
    %50 = arith.mulf %39, %49 : f32
    %51 = arith.index_cast %0 : index to i32
    %52 = arith.muli %51, %c121_i32 : i32
    %53 = arith.index_cast %3 : index to i32
    %54 = arith.muli %53, %c11_i32 : i32
    %55 = arith.addi %52, %54 : i32
    %56 = arith.index_cast %6 : index to i32
    %57 = arith.addi %55, %56 : i32
    %58 = arith.muli %57, %c4_i32 : i32
    %59 = arith.addi %c3028_i32, %58 : i32
    %60 = cgra.lwi %59 : i32->f32
    %61 = arith.addf %60, %50 : f32
    %62 = arith.index_cast %0 : index to i32
    %63 = arith.muli %62, %c121_i32 : i32
    %64 = arith.index_cast %3 : index to i32
    %65 = arith.muli %64, %c11_i32 : i32
    %66 = arith.addi %63, %65 : i32
    %67 = arith.index_cast %6 : index to i32
    %68 = arith.addi %66, %67 : i32
    %69 = arith.muli %68, %c4_i32 : i32
    %70 = arith.addi %c3028_i32, %69 : i32
    cgra.swi %61, %70 : f32, i32
    %71 = arith.addi %24, %c1 : index
    cf.br ^bb7(%71 : index)
  ^bb9:  // pred: ^bb7
    %72 = arith.addi %21, %c1 : index
    cf.br ^bb6(%72 : index)
  ^bb10:  // pred: ^bb6
    %73 = arith.addi %18, %c1 : index
    cf.br ^bb5(%73 : index)
  ^bb11:  // pred: ^bb5
    %74 = arith.addi %6, %c1 : index
    cf.br ^bb3(%74 : index)
  ^bb12:  // pred: ^bb3
    %75 = arith.addi %3, %c1 : index
    cf.br ^bb2(%75 : index)
  ^bb13:  // pred: ^bb2
    %76 = arith.addi %0, %c1 : index
    cf.br ^bb1(%76 : index)
  ^bb14:  // pred: ^bb1
    return
  }
}

