module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv1d(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<64xf32>, %arg3: memref<3xf32>, %arg4: memref<62xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xi32>
    %1 = memref.load %arg1[%c0] : memref<1xi32>
    %2 = arith.subi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.addi %3, %c1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%6: index):  // 2 preds: ^bb0, ^bb5
    %7 = arith.cmpi slt, %6, %5 : index
    cf.cond_br %7, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    memref.store %cst, %arg4[%6] : memref<62xf32>
    cf.br ^bb3(%c0 : index)
  ^bb3(%8: index):  // 2 preds: ^bb2, ^bb4
    %9 = arith.cmpi slt, %8, %4 : index
    cf.cond_br %9, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %10 = arith.addi %6, %8 : index
    %11 = memref.load %arg2[%10] : memref<64xf32>
    %12 = memref.load %arg3[%8] : memref<3xf32>
    %13 = arith.mulf %11, %12 : f32
    %14 = memref.load %arg4[%6] : memref<62xf32>
    %15 = arith.addf %14, %13 : f32
    memref.store %15, %arg4[%6] : memref<62xf32>
    %16 = arith.addi %8, %c1 : index
    cf.br ^bb3(%16 : index)
  ^bb5:  // pred: ^bb3
    %17 = arith.addi %6, %c1 : index
    cf.br ^bb1(%17 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}

