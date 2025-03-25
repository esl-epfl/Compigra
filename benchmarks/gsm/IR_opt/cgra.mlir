module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gsm(%arg0: memref<1xi32>, %arg1: memref<50xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c-32768_i32 = arith.constant -32768 : i32
    %c32767_i32 = arith.constant 32767 : i32
    %c4_i32 = arith.constant 4 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb5
    %2 = arith.index_cast %0 : index to i32
    %3 = arith.index_cast %c40 : index to i32
    cgra.cond_br<ge> [%2 : i32, %3 : i32], ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %4 = arith.index_cast %0 : index to i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.addi %c132_i32, %5 : i32
    %7 = cgra.lwi %6 : i32->i32
    cgra.cond_br<ge> [%7 : i32, %c0_i32 : i32], ^bb5(%7 : i32), ^bb3
  ^bb3:  // pred: ^bb2
    cgra.cond_br<eq> [%7 : i32, %c-32768_i32 : i32], ^bb5(%c32767_i32 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %8 = arith.subi %c0_i32, %7 : i32
    cf.br ^bb5(%8 : i32)
  ^bb5(%9: i32):  // 3 preds: ^bb2, ^bb3, ^bb4
    %10 = arith.subi %1, %9 : i32
    %11 = cgra.bsfa %10 : i32 [%9, %1]  : i32
    %12 = arith.addi %0, %c1 : index
    cf.br ^bb1(%12, %11 : index, i32)
  ^bb6:  // pred: ^bb1
    cgra.swi %1, %c128_i32 : i32, i32
    return %c0_i32 : i32
  }
}

