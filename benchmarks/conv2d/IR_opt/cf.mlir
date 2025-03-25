module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<2xi32>
    %1 = memref.load %arg0[%c1] : memref<2xi32>
    %2 = memref.load %arg1[%c0] : memref<2xi32>
    %3 = memref.load %arg1[%c1] : memref<2xi32>
    %4 = arith.subi %0, %2 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.subi %1, %3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %2 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.addi %5, %c1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%11: index):  // 2 preds: ^bb0, ^bb10
    %12 = arith.cmpi slt, %11, %10 : index
    cf.cond_br %12, ^bb2, ^bb11
  ^bb2:  // pred: ^bb1
    %13 = arith.addi %7, %c1 : index
    cf.br ^bb3(%c0 : index)
  ^bb3(%14: index):  // 2 preds: ^bb2, ^bb9
    %15 = arith.cmpi slt, %14, %13 : index
    cf.cond_br %15, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    memref.store %cst, %arg4[%11, %14] : memref<10x10xf32>
    cf.br ^bb5(%c0 : index)
  ^bb5(%16: index):  // 2 preds: ^bb4, ^bb8
    %17 = arith.cmpi slt, %16, %8 : index
    cf.cond_br %17, ^bb6(%c0 : index), ^bb9
  ^bb6(%18: index):  // 2 preds: ^bb5, ^bb7
    %19 = arith.cmpi slt, %18, %9 : index
    cf.cond_br %19, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %20 = arith.addi %11, %16 : index
    %21 = arith.addi %14, %18 : index
    %22 = memref.load %arg2[%20, %21] : memref<12x12xf32>
    %23 = memref.load %arg3[%16, %18] : memref<3x3xf32>
    %24 = arith.mulf %22, %23 : f32
    %25 = memref.load %arg4[%11, %14] : memref<10x10xf32>
    %26 = arith.addf %25, %24 : f32
    memref.store %26, %arg4[%11, %14] : memref<10x10xf32>
    %27 = arith.addi %18, %c1 : index
    cf.br ^bb6(%27 : index)
  ^bb8:  // pred: ^bb6
    %28 = arith.addi %16, %c1 : index
    cf.br ^bb5(%28 : index)
  ^bb9:  // pred: ^bb5
    %29 = arith.addi %14, %c1 : index
    cf.br ^bb3(%29 : index)
  ^bb10:  // pred: ^bb3
    %30 = arith.addi %11, %c1 : index
    cf.br ^bb1(%30 : index)
  ^bb11:  // pred: ^bb1
    return
  }
}

