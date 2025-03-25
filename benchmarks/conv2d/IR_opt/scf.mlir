module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv2d(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<12x12xf32>, %arg3: memref<3x3xf32>, %arg4: memref<10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<2xi32>
    %c1 = arith.constant 1 : index
    %1 = memref.load %arg0[%c1] : memref<2xi32>
    %c0_0 = arith.constant 0 : index
    %2 = memref.load %arg1[%c0_0] : memref<2xi32>
    %c1_1 = arith.constant 1 : index
    %3 = memref.load %arg1[%c1_1] : memref<2xi32>
    %4 = arith.subi %0, %2 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.subi %1, %3 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %2 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %10 = arith.addi %5, %c1_3 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg5 = %c0_2 to %10 step %c1_4 {
      %c0_5 = arith.constant 0 : index
      %c1_6 = arith.constant 1 : index
      %11 = arith.addi %7, %c1_6 : index
      %c1_7 = arith.constant 1 : index
      scf.for %arg6 = %c0_5 to %11 step %c1_7 {
        memref.store %cst, %arg4[%arg5, %arg6] : memref<10x10xf32>
        %c0_8 = arith.constant 0 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg7 = %c0_8 to %8 step %c1_9 {
          %c0_10 = arith.constant 0 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg8 = %c0_10 to %9 step %c1_11 {
            %12 = arith.addi %arg5, %arg7 : index
            %13 = arith.addi %arg6, %arg8 : index
            %14 = memref.load %arg2[%12, %13] : memref<12x12xf32>
            %15 = memref.load %arg3[%arg7, %arg8] : memref<3x3xf32>
            %16 = arith.mulf %14, %15 : f32
            %17 = memref.load %arg4[%arg5, %arg6] : memref<10x10xf32>
            %18 = arith.addf %17, %16 : f32
            memref.store %18, %arg4[%arg5, %arg6] : memref<10x10xf32>
          }
        }
      }
    }
    return
  }
}

