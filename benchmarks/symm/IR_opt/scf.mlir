module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @symm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg1[%c0] : memref<1xi32>
    %c0_0 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0_0] : memref<1xi32>
    %c0_1 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0_1 to %c10 step %c1 {
      %c0_2 = arith.constant 0 : index
      %c10_3 = arith.constant 10 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg6 = %c0_2 to %c10_3 step %c1_4 {
        %c0_5 = arith.constant 0 : index
        %c1_6 = arith.constant 1 : index
        %2 = scf.for %arg7 = %c0_5 to %arg5 step %c1_6 iter_args(%arg8 = %c0_i32) -> (i32) {
          %12 = memref.load %arg4[%arg5, %arg6] : memref<10x10xi32>
          %13 = arith.muli %1, %12 : i32
          %14 = memref.load %arg3[%arg5, %arg7] : memref<10x10xi32>
          %15 = arith.muli %13, %14 : i32
          %16 = memref.load %arg2[%arg7, %arg6] : memref<10x10xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %arg2[%arg7, %arg6] : memref<10x10xi32>
          %18 = memref.load %arg4[%arg7, %arg6] : memref<10x10xi32>
          %19 = memref.load %arg3[%arg5, %arg7] : memref<10x10xi32>
          %20 = arith.muli %18, %19 : i32
          %21 = arith.addi %arg8, %20 : i32
          scf.yield %21 : i32
        }
        %3 = memref.load %arg2[%arg5, %arg6] : memref<10x10xi32>
        %4 = arith.muli %0, %3 : i32
        %5 = memref.load %arg4[%arg5, %arg6] : memref<10x10xi32>
        %6 = arith.muli %1, %5 : i32
        %7 = memref.load %arg3[%arg5, %arg5] : memref<10x10xi32>
        %8 = arith.muli %6, %7 : i32
        %9 = arith.addi %4, %8 : i32
        %10 = arith.muli %1, %2 : i32
        %11 = arith.addi %9, %10 : i32
        memref.store %11, %arg2[%arg5, %arg6] : memref<10x10xi32>
      }
    }
    return
  }
}

