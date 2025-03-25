module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c756_i32 = arith.constant {BaseAddr = "arg4"} 756 : i32
    %c3_i32 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 3 : i32
    %c720_i32 = arith.constant {BaseAddr = "arg3"} 720 : i32
    %c12_i32 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 12 : i32
    %c144_i32 = arith.constant {BaseAddr = "arg2"} 144 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg1"} 136 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %c132_i32 = arith.constant 132 : i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = cgra.lwi %c136_i32 : i32->i32
    %c140_i32 = arith.constant 140 : i32
    %3 = cgra.lwi %c140_i32 : i32->i32
    %4 = arith.subi %0, %2 : i32
    %5 = arith.subi %1, %3 : i32
    %6 = arith.addi %4, %c1_i32 : i32
    %7 = arith.addi %c0_i32_5, %c0_i32_6 {constant = 0 : i32} : i32
    cf.br ^bb1(%7 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb10
    cgra.cond_br<ge> [%8 : i32, %6 : i32], ^bb11, ^bb2
  ^bb2:  // pred: ^bb1
    %9 = arith.addi %5, %c1_i32 : i32
    %10 = arith.addi %c0_i32_3, %c0_i32_4 {constant = 0 : i32} : i32
    cf.br ^bb3(%10 : i32)
  ^bb3(%11: i32):  // 2 preds: ^bb2, ^bb9
    cgra.cond_br<ge> [%11 : i32, %9 : i32], ^bb10, ^bb4
  ^bb4:  // pred: ^bb3
    %12 = arith.muli %8, %c10_i32 : i32
    %13 = arith.addi %12, %11 : i32
    %14 = arith.muli %13, %c4_i32 : i32
    %15 = arith.addi %14, %c756_i32 : i32
    cgra.swi %cst, %15 : f32, i32
    %16 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    cf.br ^bb5(%16 : i32)
  ^bb5(%17: i32):  // 2 preds: ^bb4, ^bb8
    %18 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%17 : i32, %2 : i32], ^bb9, ^bb6(%18 : i32)
  ^bb6(%19: i32):  // 2 preds: ^bb5, ^bb7
    cgra.cond_br<ge> [%19 : i32, %3 : i32], ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %20 = arith.addi %8, %17 : i32
    %21 = arith.addi %11, %19 : i32
    %22 = arith.muli %20, %c12_i32 : i32
    %23 = arith.addi %22, %21 : i32
    %24 = arith.muli %23, %c4_i32 : i32
    %25 = arith.addi %24, %c144_i32 : i32
    %26 = cgra.lwi %25 : i32->f32
    %27 = arith.muli %17, %c3_i32 : i32
    %28 = arith.addi %27, %19 : i32
    %29 = arith.muli %28, %c4_i32 : i32
    %30 = arith.addi %29, %c720_i32 : i32
    %31 = cgra.lwi %30 : i32->f32
    %32 = arith.mulf %26, %31 : f32
    %33 = arith.muli %8, %c10_i32 : i32
    %34 = arith.addi %33, %11 : i32
    %35 = arith.muli %34, %c4_i32 : i32
    %36 = arith.addi %35, %c756_i32 : i32
    %37 = cgra.lwi %36 : i32->f32
    %38 = arith.addf %37, %32 : f32
    %39 = arith.muli %8, %c10_i32 : i32
    %40 = arith.addi %39, %11 : i32
    %41 = arith.muli %40, %c4_i32 : i32
    %42 = arith.addi %41, %c756_i32 : i32
    cgra.swi %38, %42 : f32, i32
    %43 = arith.addi %19, %c1_i32 : i32
    cf.br ^bb6(%43 : i32)
  ^bb8:  // pred: ^bb6
    %44 = arith.addi %17, %c1_i32 : i32
    cf.br ^bb5(%44 : i32)
  ^bb9:  // pred: ^bb5
    %45 = arith.addi %11, %c1_i32 : i32
    cf.br ^bb3(%45 : i32)
  ^bb10:  // pred: ^bb3
    %46 = arith.addi %8, %c1_i32 : i32
    cf.br ^bb1(%46 : i32)
  ^bb11:  // pred: ^bb1
    return
  }
}

