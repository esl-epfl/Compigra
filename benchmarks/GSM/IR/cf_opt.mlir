module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @GSM(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c40_i32 = arith.constant 40 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32767_i32 = arith.constant 32767 : i32
    %c-32768_i32 = arith.constant -32768 : i32
    cf.br ^bb1(%c0_i32, %arg0 : i32, i32)
  ^bb1(%0: i32, %1: i32):  // 2 preds: ^bb0, ^bb1
    %2 = arith.cmpi slt, %0, %c40_i32 : i32
    %3 = arith.index_cast %0 : i32 to index
    %4 = memref.load %arg2[%3] : memref<?xi32>
    %5 = arith.cmpi slt, %4, %c0_i32 : i32
    %6 = arith.cmpi eq, %4, %c-32768_i32 : i32
    %7 = arith.subi %c0_i32, %4 : i32
    %8 = arith.select %6, %c32767_i32, %7 : i32
    %9 = arith.select %5, %8, %4 : i32
    %10 = arith.cmpi sgt, %9, %1 : i32
    %11 = arith.select %10, %9, %1 : i32
    %12 = arith.addi %0, %c1_i32 : i32
    cf.cond_br %2, ^bb1(%12, %11 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    return %1 : i32
  }
}

