module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<8x8xi32>, %arg3: memref<8x8xi32>, %arg4: memref<8x8xi32>, %arg5: memref<8x8xi32>, %arg6: memref<8x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xi32>
    %1 = memref.load %arg1[%c0] : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb7
    %3 = arith.cmpi slt, %2, %c8 : index
    cf.cond_br %3, ^bb2(%c0 : index), ^bb8(%c0 : index)
  ^bb2(%4: index):  // 2 preds: ^bb1, ^bb6
    %5 = arith.cmpi slt, %4, %c8 : index
    cf.cond_br %5, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    memref.store %c0_i32, %arg2[%2, %4] : memref<8x8xi32>
    cf.br ^bb4(%c0 : index)
  ^bb4(%6: index):  // 2 preds: ^bb3, ^bb5
    %7 = arith.cmpi slt, %6, %c8 : index
    cf.cond_br %7, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %8 = memref.load %arg3[%2, %6] : memref<8x8xi32>
    %9 = arith.muli %0, %8 : i32
    %10 = memref.load %arg4[%6, %4] : memref<8x8xi32>
    %11 = arith.muli %9, %10 : i32
    %12 = memref.load %arg2[%2, %4] : memref<8x8xi32>
    %13 = arith.addi %12, %11 : i32
    memref.store %13, %arg2[%2, %4] : memref<8x8xi32>
    %14 = arith.addi %6, %c1 : index
    cf.br ^bb4(%14 : index)
  ^bb6:  // pred: ^bb4
    %15 = arith.addi %4, %c1 : index
    cf.br ^bb2(%15 : index)
  ^bb7:  // pred: ^bb2
    %16 = arith.addi %2, %c1 : index
    cf.br ^bb1(%16 : index)
  ^bb8(%17: index):  // 2 preds: ^bb1, ^bb14
    %18 = arith.cmpi slt, %17, %c8 : index
    cf.cond_br %18, ^bb9(%c0 : index), ^bb15
  ^bb9(%19: index):  // 2 preds: ^bb8, ^bb13
    %20 = arith.cmpi slt, %19, %c8 : index
    cf.cond_br %20, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    %21 = memref.load %arg6[%17, %19] : memref<8x8xi32>
    %22 = arith.muli %21, %1 : i32
    memref.store %22, %arg6[%17, %19] : memref<8x8xi32>
    cf.br ^bb11(%c0 : index)
  ^bb11(%23: index):  // 2 preds: ^bb10, ^bb12
    %24 = arith.cmpi slt, %23, %c8 : index
    cf.cond_br %24, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %25 = memref.load %arg2[%17, %23] : memref<8x8xi32>
    %26 = memref.load %arg5[%23, %19] : memref<8x8xi32>
    %27 = arith.muli %25, %26 : i32
    %28 = memref.load %arg6[%17, %19] : memref<8x8xi32>
    %29 = arith.addi %28, %27 : i32
    memref.store %29, %arg6[%17, %19] : memref<8x8xi32>
    %30 = arith.addi %23, %c1 : index
    cf.br ^bb11(%30 : index)
  ^bb13:  // pred: ^bb11
    %31 = arith.addi %19, %c1 : index
    cf.br ^bb9(%31 : index)
  ^bb14:  // pred: ^bb9
    %32 = arith.addi %17, %c1 : index
    cf.br ^bb8(%32 : index)
  ^bb15:  // pred: ^bb8
    return
  }
}

