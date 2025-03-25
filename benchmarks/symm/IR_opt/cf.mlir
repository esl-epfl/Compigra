module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @symm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg1[%c0] : memref<1xi32>
    %1 = memref.load %arg0[%c0] : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb6
    %3 = arith.cmpi slt, %2, %c10 : index
    cf.cond_br %3, ^bb2(%c0 : index), ^bb7
  ^bb2(%4: index):  // 2 preds: ^bb1, ^bb5
    %5 = arith.cmpi slt, %4, %c10 : index
    cf.cond_br %5, ^bb3(%c0, %c0_i32 : index, i32), ^bb6
  ^bb3(%6: index, %7: i32):  // 2 preds: ^bb2, ^bb4
    %8 = arith.cmpi slt, %6, %2 : index
    cf.cond_br %8, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %9 = memref.load %arg4[%2, %4] : memref<10x10xi32>
    %10 = arith.muli %1, %9 : i32
    %11 = memref.load %arg3[%2, %6] : memref<10x10xi32>
    %12 = arith.muli %10, %11 : i32
    %13 = memref.load %arg2[%6, %4] : memref<10x10xi32>
    %14 = arith.addi %13, %12 : i32
    memref.store %14, %arg2[%6, %4] : memref<10x10xi32>
    %15 = memref.load %arg4[%6, %4] : memref<10x10xi32>
    %16 = memref.load %arg3[%2, %6] : memref<10x10xi32>
    %17 = arith.muli %15, %16 : i32
    %18 = arith.addi %7, %17 : i32
    %19 = arith.addi %6, %c1 : index
    cf.br ^bb3(%19, %18 : index, i32)
  ^bb5:  // pred: ^bb3
    %20 = memref.load %arg2[%2, %4] : memref<10x10xi32>
    %21 = arith.muli %0, %20 : i32
    %22 = memref.load %arg4[%2, %4] : memref<10x10xi32>
    %23 = arith.muli %1, %22 : i32
    %24 = memref.load %arg3[%2, %2] : memref<10x10xi32>
    %25 = arith.muli %23, %24 : i32
    %26 = arith.addi %21, %25 : i32
    %27 = arith.muli %1, %7 : i32
    %28 = arith.addi %26, %27 : i32
    memref.store %28, %arg2[%2, %4] : memref<10x10xi32>
    %29 = arith.addi %4, %c1 : index
    cf.br ^bb2(%29 : index)
  ^bb6:  // pred: ^bb2
    %30 = arith.addi %2, %c1 : index
    cf.br ^bb1(%30 : index)
  ^bb7:  // pred: ^bb1
    return
  }
}

