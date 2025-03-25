module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv3d(%arg0: memref<3x15x15xf32>, %arg1: memref<2x5x5xf32>, %arg2: memref<2x11x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c11 = arith.constant 11 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c11 step %c1_1 {
        %c0_2 = arith.constant 0 : index
        %c11_3 = arith.constant 11 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg5 = %c0_2 to %c11_3 step %c1_4 {
          memref.store %cst, %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
          %c0_5 = arith.constant 0 : index
          %c2_6 = arith.constant 2 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg6 = %c0_5 to %c2_6 step %c1_7 {
            %c0_8 = arith.constant 0 : index
            %c5 = arith.constant 5 : index
            %c1_9 = arith.constant 1 : index
            scf.for %arg7 = %c0_8 to %c5 step %c1_9 {
              %c0_10 = arith.constant 0 : index
              %c5_11 = arith.constant 5 : index
              %c1_12 = arith.constant 1 : index
              scf.for %arg8 = %c0_10 to %c5_11 step %c1_12 {
                %0 = arith.addi %arg3, %arg6 : index
                %1 = arith.addi %arg4, %arg7 : index
                %2 = arith.addi %arg5, %arg8 : index
                %3 = memref.load %arg0[%0, %1, %2] : memref<3x15x15xf32>
                %4 = memref.load %arg1[%arg6, %arg7, %arg8] : memref<2x5x5xf32>
                %5 = arith.mulf %3, %4 : f32
                %6 = memref.load %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
                %7 = arith.addf %6, %5 : f32
                memref.store %7, %arg2[%arg3, %arg4, %arg5] : memref<2x11x11xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

