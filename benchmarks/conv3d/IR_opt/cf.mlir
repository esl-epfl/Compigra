module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv3d(%arg0: memref<3x15x15xf32>, %arg1: memref<2x5x5xf32>, %arg2: memref<2x11x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c11 = arith.constant 11 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb13
    %1 = arith.cmpi slt, %0, %c2 : index
    cf.cond_br %1, ^bb2(%c0 : index), ^bb14
  ^bb2(%2: index):  // 2 preds: ^bb1, ^bb12
    %3 = arith.cmpi slt, %2, %c11 : index
    cf.cond_br %3, ^bb3(%c0 : index), ^bb13
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb11
    %5 = arith.cmpi slt, %4, %c11 : index
    cf.cond_br %5, ^bb4, ^bb12
  ^bb4:  // pred: ^bb3
    memref.store %cst, %arg2[%0, %2, %4] : memref<2x11x11xf32>
    cf.br ^bb5(%c0 : index)
  ^bb5(%6: index):  // 2 preds: ^bb4, ^bb10
    %7 = arith.cmpi slt, %6, %c2 : index
    cf.cond_br %7, ^bb6(%c0 : index), ^bb11
  ^bb6(%8: index):  // 2 preds: ^bb5, ^bb9
    %9 = arith.cmpi slt, %8, %c5 : index
    cf.cond_br %9, ^bb7(%c0 : index), ^bb10
  ^bb7(%10: index):  // 2 preds: ^bb6, ^bb8
    %11 = arith.cmpi slt, %10, %c5 : index
    cf.cond_br %11, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %12 = arith.addi %0, %6 : index
    %13 = arith.addi %2, %8 : index
    %14 = arith.addi %4, %10 : index
    %15 = memref.load %arg0[%12, %13, %14] : memref<3x15x15xf32>
    %16 = memref.load %arg1[%6, %8, %10] : memref<2x5x5xf32>
    %17 = arith.mulf %15, %16 : f32
    %18 = memref.load %arg2[%0, %2, %4] : memref<2x11x11xf32>
    %19 = arith.addf %18, %17 : f32
    memref.store %19, %arg2[%0, %2, %4] : memref<2x11x11xf32>
    %20 = arith.addi %10, %c1 : index
    cf.br ^bb7(%20 : index)
  ^bb9:  // pred: ^bb7
    %21 = arith.addi %8, %c1 : index
    cf.br ^bb6(%21 : index)
  ^bb10:  // pred: ^bb6
    %22 = arith.addi %6, %c1 : index
    cf.br ^bb5(%22 : index)
  ^bb11:  // pred: ^bb5
    %23 = arith.addi %4, %c1 : index
    cf.br ^bb3(%23 : index)
  ^bb12:  // pred: ^bb3
    %24 = arith.addi %2, %c1 : index
    cf.br ^bb2(%24 : index)
  ^bb13:  // pred: ^bb2
    %25 = arith.addi %0, %c1 : index
    cf.br ^bb1(%25 : index)
  ^bb14:  // pred: ^bb1
    return
  }
}

