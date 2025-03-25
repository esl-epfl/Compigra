module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv3d(%arg0: memref<3x15x15xf32>, %arg1: memref<2x5x5xf32>, %arg2: memref<2x11x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 11 {
        affine.for %arg5 = 0 to 11 {
          affine.store %cst, %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 5 {
              affine.for %arg8 = 0 to 5 {
                %0 = affine.load %arg0[%arg3 + %arg6, %arg4 + %arg7, %arg5 + %arg8] : memref<3x15x15xf32>
                %1 = affine.load %arg1[%arg6, %arg7, %arg8] : memref<2x5x5xf32>
                %2 = arith.mulf %0, %1 : f32
                %3 = affine.load %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
                %4 = arith.addf %3, %2 : f32
                affine.store %4, %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
