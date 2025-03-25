module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>, %arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>, %arg6: memref<4x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb7
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb2(%c0 : index), ^bb8(%c0 : index)
  ^bb2(%2: index):  // 2 preds: ^bb1, ^bb6
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    memref.store %c0_i32, %arg4[%0, %2] : memref<4x4xi32>
    cf.br ^bb4(%c0 : index)
  ^bb4(%4: index):  // 2 preds: ^bb3, ^bb5
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %6 = memref.load %arg0[%0, %4] : memref<4x4xi32>
    %7 = memref.load %arg1[%4, %2] : memref<4x4xi32>
    %8 = arith.muli %6, %7 : i32
    %9 = memref.load %arg4[%0, %2] : memref<4x4xi32>
    %10 = arith.addi %9, %8 : i32
    memref.store %10, %arg4[%0, %2] : memref<4x4xi32>
    %11 = arith.addi %4, %c1 : index
    cf.br ^bb4(%11 : index)
  ^bb6:  // pred: ^bb4
    %12 = arith.addi %2, %c1 : index
    cf.br ^bb2(%12 : index)
  ^bb7:  // pred: ^bb2
    %13 = arith.addi %0, %c1 : index
    cf.br ^bb1(%13 : index)
  ^bb8(%14: index):  // 2 preds: ^bb1, ^bb14
    %15 = arith.cmpi slt, %14, %c4 : index
    cf.cond_br %15, ^bb9(%c0 : index), ^bb15(%c0 : index)
  ^bb9(%16: index):  // 2 preds: ^bb8, ^bb13
    %17 = arith.cmpi slt, %16, %c4 : index
    cf.cond_br %17, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    memref.store %c0_i32, %arg5[%14, %16] : memref<4x4xi32>
    cf.br ^bb11(%c0 : index)
  ^bb11(%18: index):  // 2 preds: ^bb10, ^bb12
    %19 = arith.cmpi slt, %18, %c4 : index
    cf.cond_br %19, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %20 = memref.load %arg2[%14, %18] : memref<4x4xi32>
    %21 = memref.load %arg3[%18, %16] : memref<4x4xi32>
    %22 = arith.muli %20, %21 : i32
    %23 = memref.load %arg5[%14, %16] : memref<4x4xi32>
    %24 = arith.addi %23, %22 : i32
    memref.store %24, %arg5[%14, %16] : memref<4x4xi32>
    %25 = arith.addi %18, %c1 : index
    cf.br ^bb11(%25 : index)
  ^bb13:  // pred: ^bb11
    %26 = arith.addi %16, %c1 : index
    cf.br ^bb9(%26 : index)
  ^bb14:  // pred: ^bb9
    %27 = arith.addi %14, %c1 : index
    cf.br ^bb8(%27 : index)
  ^bb15(%28: index):  // 2 preds: ^bb8, ^bb21
    %29 = arith.cmpi slt, %28, %c4 : index
    cf.cond_br %29, ^bb16(%c0 : index), ^bb22
  ^bb16(%30: index):  // 2 preds: ^bb15, ^bb20
    %31 = arith.cmpi slt, %30, %c4 : index
    cf.cond_br %31, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    memref.store %c0_i32, %arg6[%28, %30] : memref<4x4xi32>
    cf.br ^bb18(%c0 : index)
  ^bb18(%32: index):  // 2 preds: ^bb17, ^bb19
    %33 = arith.cmpi slt, %32, %c4 : index
    cf.cond_br %33, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %34 = memref.load %arg4[%28, %32] : memref<4x4xi32>
    %35 = memref.load %arg5[%32, %30] : memref<4x4xi32>
    %36 = arith.muli %34, %35 : i32
    %37 = memref.load %arg6[%28, %30] : memref<4x4xi32>
    %38 = arith.addi %37, %36 : i32
    memref.store %38, %arg6[%28, %30] : memref<4x4xi32>
    %39 = arith.addi %32, %c1 : index
    cf.br ^bb18(%39 : index)
  ^bb20:  // pred: ^bb18
    %40 = arith.addi %30, %c1 : index
    cf.br ^bb16(%40 : index)
  ^bb21:  // pred: ^bb16
    %41 = arith.addi %28, %c1 : index
    cf.br ^bb15(%41 : index)
  ^bb22:  // pred: ^bb15
    return
  }
}

