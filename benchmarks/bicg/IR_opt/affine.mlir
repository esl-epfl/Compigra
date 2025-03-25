module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bicg(%arg0: memref<30x20xf32>, %arg1: memref<20xf32>, %arg2: memref<30xf32>, %arg3: memref<20xf32>, %arg4: memref<30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 20 {
      affine.store %cst, %arg1[%arg5] : memref<20xf32>
    }
    affine.for %arg5 = 0 to 30 {
      affine.store %cst, %arg2[%arg5] : memref<30xf32>
      affine.for %arg6 = 0 to 20 {
        %0 = affine.load %arg1[%arg6] : memref<20xf32>
        %1 = affine.load %arg4[%arg5] : memref<30xf32>
        %2 = affine.load %arg0[%arg5, %arg6] : memref<30x20xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        affine.store %4, %arg1[%arg6] : memref<20xf32>
        %5 = affine.load %arg2[%arg5] : memref<30xf32>
        %6 = affine.load %arg0[%arg5, %arg6] : memref<30x20xf32>
        %7 = affine.load %arg3[%arg6] : memref<20xf32>
        %8 = arith.mulf %6, %7 : f32
        %9 = arith.addf %5, %8 : f32
        affine.store %9, %arg2[%arg5] : memref<30xf32>
      }
    }
    return
  }
}
