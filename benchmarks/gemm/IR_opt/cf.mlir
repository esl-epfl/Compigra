module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xi32>
    %1 = memref.load %arg1[%c0] : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb8
    %3 = arith.cmpi slt, %2, %c10 : index
    cf.cond_br %3, ^bb2(%c0 : index), ^bb9
  ^bb2(%4: index):  // 2 preds: ^bb1, ^bb3
    %5 = arith.cmpi slt, %4, %c10 : index
    cf.cond_br %5, ^bb3, ^bb4(%c0 : index)
  ^bb3:  // pred: ^bb2
    %6 = memref.load %arg2[%2, %4] : memref<10x10xi32>
    %7 = arith.muli %6, %1 : i32
    memref.store %7, %arg2[%2, %4] : memref<10x10xi32>
    %8 = arith.addi %4, %c1 : index
    cf.br ^bb2(%8 : index)
  ^bb4(%9: index):  // 2 preds: ^bb2, ^bb7
    %10 = arith.cmpi slt, %9, %c10 : index
    cf.cond_br %10, ^bb5(%c0 : index), ^bb8
  ^bb5(%11: index):  // 2 preds: ^bb4, ^bb6
    %12 = arith.cmpi slt, %11, %c10 : index
    cf.cond_br %12, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %13 = memref.load %arg3[%2, %9] : memref<10x10xi32>
    %14 = arith.muli %0, %13 : i32
    %15 = memref.load %arg4[%9, %11] : memref<10x10xi32>
    %16 = arith.muli %14, %15 : i32
    %17 = memref.load %arg2[%2, %11] : memref<10x10xi32>
    %18 = arith.addi %17, %16 : i32
    memref.store %18, %arg2[%2, %11] : memref<10x10xi32>
    %19 = arith.addi %11, %c1 : index
    cf.br ^bb5(%19 : index)
  ^bb7:  // pred: ^bb5
    %20 = arith.addi %9, %c1 : index
    cf.br ^bb4(%20 : index)
  ^bb8:  // pred: ^bb4
    %21 = arith.addi %2, %c1 : index
    cf.br ^bb1(%21 : index)
  ^bb9:  // pred: ^bb1
    return
  }
}

