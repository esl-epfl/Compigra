#map = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv1d(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<64xf32>, %arg3: memref<3xf32>, %arg4: memref<62xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.load %arg0[0] : memref<1xi32>
    %1 = affine.load %arg1[0] : memref<1xi32>
    %2 = arith.subi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    affine.for %arg5 = 0 to #map()[%3] {
      affine.store %cst, %arg4[%arg5] : memref<62xf32>
      affine.for %arg6 = 0 to %4 {
        %5 = affine.load %arg2[%arg5 + %arg6] : memref<64xf32>
        %6 = affine.load %arg3[%arg6] : memref<3xf32>
        %7 = arith.mulf %5, %6 : f32
        %8 = affine.load %arg4[%arg5] : memref<62xf32>
        %9 = arith.addf %8, %7 : f32
        affine.store %9, %arg4[%arg5] : memref<62xf32>
      }
    }
    return
  }
}
