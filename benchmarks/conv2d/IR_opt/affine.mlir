#map = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.load %arg0[0] : memref<2xi32>
    %1 = affine.load %arg0[1] : memref<2xi32>
    %2 = affine.load %arg1[0] : memref<2xi32>
    %3 = affine.load %arg1[1] : memref<2xi32>
    %4 = arith.subi %0, %2 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.subi %1, %3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %2 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    affine.for %arg5 = 0 to #map()[%5] {
      affine.for %arg6 = 0 to #map()[%7] {
        affine.store %cst, %arg4[%arg5, %arg6] : memref<10x10xf32>
        affine.for %arg7 = 0 to %8 {
          affine.for %arg8 = 0 to %9 {
            %10 = affine.load %arg2[%arg5 + %arg7, %arg6 + %arg8] : memref<12x12xf32>
            %11 = affine.load %arg3[%arg7, %arg8] : memref<3x3xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = affine.load %arg4[%arg5, %arg6] : memref<10x10xf32>
            %14 = arith.addf %13, %12 : f32
            affine.store %14, %arg4[%arg5, %arg6] : memref<10x10xf32>
          }
        }
      }
    }
    return
  }
}
