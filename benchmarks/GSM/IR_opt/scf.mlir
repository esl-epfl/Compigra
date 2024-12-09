module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @GSM(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32767_i32 = arith.constant 32767 : i32
    %c-32768_i32 = arith.constant -32768 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<?xi32>
    %c0_0 = arith.constant 0 : index
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    %1 = scf.for %arg2 = %c0_0 to %c40 step %c1 iter_args(%arg3 = %0) -> (i32) {
      %2 = memref.load %arg1[%arg2] : memref<?xi32>
      %3 = arith.cmpi slt, %2, %c0_i32 : i32
      %4 = arith.cmpi eq, %2, %c-32768_i32 : i32
      %5 = scf.if %3 -> (i32) {
        %8 = scf.if %4 -> (i32) {
          scf.yield %c32767_i32 : i32
        } else {
          %9 = arith.subi %c0_i32, %2 : i32
          scf.yield %9 : i32
        }
        scf.yield %8 : i32
      } else {
        scf.yield %2 : i32
      }
      %6 = arith.cmpi sgt, %5, %arg3 : i32
      %7 = arith.select %6, %5, %arg3 : i32
      scf.yield %7 : i32
    }
    %c0_1 = arith.constant 0 : index
    memref.store %1, %arg0[%c0_1] : memref<?xi32>
    return %c0_i32 : i32
  }
}

