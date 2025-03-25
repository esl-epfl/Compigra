module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @FIR(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb7
    %1 = arith.cmpi slt, %0, %c16 : index
    cf.cond_br %1, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    memref.store %c0_i32, %arg1[%0] : memref<16xi32>
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb6
    %3 = arith.cmpi slt, %2, %c5 : index
    cf.cond_br %3, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %4 = arith.subi %0, %2 : index
    %5 = arith.cmpi sge, %4, %c0 : index
    cf.cond_br %5, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %6 = memref.load %arg2[%2] : memref<5xi32>
    %7 = arith.subi %0, %2 : index
    %8 = memref.load %arg0[%7] : memref<16xi32>
    %9 = arith.muli %6, %8 : i32
    %10 = memref.load %arg1[%0] : memref<16xi32>
    %11 = arith.addi %10, %9 : i32
    memref.store %11, %arg1[%0] : memref<16xi32>
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %12 = arith.addi %2, %c1 : index
    cf.br ^bb3(%12 : index)
  ^bb7:  // pred: ^bb3
    %13 = arith.addi %0, %c1 : index
    cf.br ^bb1(%13 : index)
  ^bb8:  // pred: ^bb1
    return
  }
}

