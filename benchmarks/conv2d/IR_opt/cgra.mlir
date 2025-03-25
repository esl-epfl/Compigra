module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c4_i32_0 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4_i32_1 = arith.constant 4 : i32
    %c10_i32 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c756_i32 = arith.constant {BaseAddr = "arg4"} 756 : i32
    %c3_i32 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 3 : i32
    %c720_i32 = arith.constant {BaseAddr = "arg3"} 720 : i32
    %c12_i32 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 12 : i32
    %c144_i32 = arith.constant {BaseAddr = "arg2"} 144 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg1"} 136 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = arith.addi %c128_i32, %c4_i32_0 : i32
    %2 = cgra.lwi %1 : i32->i32
    %3 = cgra.lwi %c136_i32 : i32->i32
    %4 = arith.addi %c136_i32, %c4_i32 : i32
    %5 = cgra.lwi %4 : i32->i32
    %6 = arith.subi %0, %3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.subi %2, %5 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %5 : i32 to index
    %12 = arith.addi %7, %c1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%13: index):  // 2 preds: ^bb0, ^bb10
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.index_cast %12 : index to i32
    cgra.cond_br<ge> [%14 : i32, %15 : i32], ^bb11, ^bb2
  ^bb2:  // pred: ^bb1
    %16 = arith.addi %9, %c1 : index
    cf.br ^bb3(%c0 : index)
  ^bb3(%17: index):  // 2 preds: ^bb2, ^bb9
    %18 = arith.index_cast %17 : index to i32
    %19 = arith.index_cast %16 : index to i32
    cgra.cond_br<ge> [%18 : i32, %19 : i32], ^bb10, ^bb4
  ^bb4:  // pred: ^bb3
    %20 = arith.index_cast %13 : index to i32
    %21 = arith.muli %20, %c10_i32 : i32
    %22 = arith.index_cast %17 : index to i32
    %23 = arith.addi %21, %22 : i32
    %24 = arith.muli %23, %c4_i32_1 : i32
    %25 = arith.addi %c756_i32, %24 : i32
    cgra.swi %cst, %25 : f32, i32
    cf.br ^bb5(%c0 : index)
  ^bb5(%26: index):  // 2 preds: ^bb4, ^bb8
    %27 = arith.index_cast %26 : index to i32
    %28 = arith.index_cast %10 : index to i32
    cgra.cond_br<ge> [%27 : i32, %28 : i32], ^bb9, ^bb6(%c0 : index)
  ^bb6(%29: index):  // 2 preds: ^bb5, ^bb7
    %30 = arith.index_cast %29 : index to i32
    %31 = arith.index_cast %11 : index to i32
    cgra.cond_br<ge> [%30 : i32, %31 : i32], ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %32 = arith.addi %13, %26 : index
    %33 = arith.addi %17, %29 : index
    %34 = arith.index_cast %32 : index to i32
    %35 = arith.muli %34, %c12_i32 : i32
    %36 = arith.index_cast %33 : index to i32
    %37 = arith.addi %35, %36 : i32
    %38 = arith.muli %37, %c4_i32_1 : i32
    %39 = arith.addi %c144_i32, %38 : i32
    %40 = cgra.lwi %39 : i32->f32
    %41 = arith.index_cast %26 : index to i32
    %42 = arith.muli %41, %c3_i32 : i32
    %43 = arith.index_cast %29 : index to i32
    %44 = arith.addi %42, %43 : i32
    %45 = arith.muli %44, %c4_i32_1 : i32
    %46 = arith.addi %c720_i32, %45 : i32
    %47 = cgra.lwi %46 : i32->f32
    %48 = arith.mulf %40, %47 : f32
    %49 = arith.index_cast %13 : index to i32
    %50 = arith.muli %49, %c10_i32 : i32
    %51 = arith.index_cast %17 : index to i32
    %52 = arith.addi %50, %51 : i32
    %53 = arith.muli %52, %c4_i32_1 : i32
    %54 = arith.addi %c756_i32, %53 : i32
    %55 = cgra.lwi %54 : i32->f32
    %56 = arith.addf %55, %48 : f32
    %57 = arith.index_cast %13 : index to i32
    %58 = arith.muli %57, %c10_i32 : i32
    %59 = arith.index_cast %17 : index to i32
    %60 = arith.addi %58, %59 : i32
    %61 = arith.muli %60, %c4_i32_1 : i32
    %62 = arith.addi %c756_i32, %61 : i32
    cgra.swi %56, %62 : f32, i32
    %63 = arith.addi %29, %c1 : index
    cf.br ^bb6(%63 : index)
  ^bb8:  // pred: ^bb6
    %64 = arith.addi %26, %c1 : index
    cf.br ^bb5(%64 : index)
  ^bb9:  // pred: ^bb5
    %65 = arith.addi %17, %c1 : index
    cf.br ^bb3(%65 : index)
  ^bb10:  // pred: ^bb3
    %66 = arith.addi %13, %c1 : index
    cf.br ^bb1(%66 : index)
  ^bb11:  // pred: ^bb1
    return
  }
}

