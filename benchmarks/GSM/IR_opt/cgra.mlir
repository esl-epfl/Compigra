module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @GSM(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant {BaseAddr = "arg0"} 0 : i32
    %c4_i32 = arith.constant {BaseAddr = "arg1"} 4 : i32
    %c4_i32_0 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c32767_i32 = arith.constant 32767 : i32
    %c-32768_i32 = arith.constant -32768 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = cgra.lwi %c0_i32 : i32->i32
    cf.br ^bb1(%c0, %0 : index, i32)
  ^bb1(%1: index, %2: i32):  // 2 preds: ^bb0, ^bb5
    %3 = arith.index_cast %1 : index to i32
    %4 = arith.index_cast %c40 : index to i32
    cgra.cond_br<ge> [%3 : i32, %4 : i32], ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %5 = arith.index_cast %1 : index to i32
    %6 = arith.muli %5, %c4_i32_0 : i32
    %7 = arith.addi %c4_i32, %6 : i32
    %8 = cgra.lwi %7 : i32->i32
    cgra.cond_br<ge> [%8 : i32, %c0_i32_1 : i32], ^bb5(%8 : i32), ^bb3
  ^bb3:  // pred: ^bb2
    cgra.cond_br<eq> [%8 : i32, %c-32768_i32 : i32], ^bb5(%c32767_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.subi %c0_i32_1, %8 : i32
    cf.br ^bb5(%9 : i32)
  ^bb5(%10: i32):  // 3 preds: ^bb2, ^bb3, ^bb4
    %11 = arith.cmpi sgt, %10, %2 : i32
    %12 = arith.select %11, %10, %2 : i32
    %13 = arith.addi %1, %c1 : index
    cf.br ^bb1(%13, %12 : index, i32)
  ^bb6:  // pred: ^bb1
    cgra.swi %2, %c0_i32 : i32, i32
    return %c0_i32_1 : i32
  }
}

