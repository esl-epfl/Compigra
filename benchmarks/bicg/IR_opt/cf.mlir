module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bicg(%arg0: memref<30x20xf32>, %arg1: memref<20xf32>, %arg2: memref<30xf32>, %arg3: memref<20xf32>, %arg4: memref<30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c30 = arith.constant 30 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c20 : index
    cf.cond_br %1, ^bb2, ^bb3(%c0 : index)
  ^bb2:  // pred: ^bb1
    memref.store %cst, %arg1[%0] : memref<20xf32>
    %2 = arith.addi %0, %c1 : index
    cf.br ^bb1(%2 : index)
  ^bb3(%3: index):  // 2 preds: ^bb1, ^bb7
    %4 = arith.cmpi slt, %3, %c30 : index
    cf.cond_br %4, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    memref.store %cst, %arg2[%3] : memref<30xf32>
    cf.br ^bb5(%c0 : index)
  ^bb5(%5: index):  // 2 preds: ^bb4, ^bb6
    %6 = arith.cmpi slt, %5, %c20 : index
    cf.cond_br %6, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %7 = memref.load %arg1[%5] : memref<20xf32>
    %8 = memref.load %arg4[%3] : memref<30xf32>
    %9 = memref.load %arg0[%3, %5] : memref<30x20xf32>
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %7, %10 : f32
    memref.store %11, %arg1[%5] : memref<20xf32>
    %12 = memref.load %arg2[%3] : memref<30xf32>
    %13 = memref.load %arg0[%3, %5] : memref<30x20xf32>
    %14 = memref.load %arg3[%5] : memref<20xf32>
    %15 = arith.mulf %13, %14 : f32
    %16 = arith.addf %12, %15 : f32
    memref.store %16, %arg2[%3] : memref<30xf32>
    %17 = arith.addi %5, %c1 : index
    cf.br ^bb5(%17 : index)
  ^bb7:  // pred: ^bb5
    %18 = arith.addi %3, %c1 : index
    cf.br ^bb3(%18 : index)
  ^bb8:  // pred: ^bb3
    return
  }
}

