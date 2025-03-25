module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv1d(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<64xf32>, %arg3: memref<3xf32>, %arg4: memref<62xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xi32>
    %c0_0 = arith.constant 0 : index
    %1 = memref.load %arg1[%c0_0] : memref<1xi32>
    %2 = arith.subi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %c0_1 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %5 = arith.addi %3, %c1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg5 = %c0_1 to %5 step %c1_2 {
      memref.store %cst, %arg4[%arg5] : memref<62xf32>
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg6 = %c0_3 to %4 step %c1_4 {
        %6 = arith.addi %arg5, %arg6 : index
        %7 = memref.load %arg2[%6] : memref<64xf32>
        %8 = memref.load %arg3[%arg6] : memref<3xf32>
        %9 = arith.mulf %7, %8 : f32
        %10 = memref.load %arg4[%arg5] : memref<62xf32>
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg4[%arg5] : memref<62xf32>
      }
    }
    return
  }
}

