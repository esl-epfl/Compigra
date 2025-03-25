module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gsm(%arg0: memref<1xi32>, %arg1: memref<50xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c40_i32 = arith.constant 40 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c4088_i32 = arith.constant 4088 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %0 = arith.addi %c4088_i32, %c0_i32_8 : i32
    %c12_i32 = arith.constant 12 : i32
    %1 = arith.shli %0, %c12_i32 : i32
    %2 = arith.addi %c0_i32_7, %1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %3 = arith.addi %c-1_i32, %c0_i32_9 : i32
    %c24_i32 = arith.constant 24 : i32
    %4 = arith.shli %3, %c24_i32 : i32
    %5 = arith.addi %2, %4 {constant = -32768 : i32} : i32
    %c4095_i32 = arith.constant 4095 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %6 = arith.addi %c7_i32, %c0_i32_10 : i32
    %c12_i32_11 = arith.constant 12 : i32
    %7 = arith.shli %6, %c12_i32_11 : i32
    %8 = arith.addi %c4095_i32, %7 {constant = 32767 : i32} : i32
    %c4_i32 = arith.constant 4 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %9 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    %10 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cf.br ^bb1(%9, %10 : i32, i32)
  ^bb1(%11: i32, %12: i32):  // 2 preds: ^bb0, ^bb5
    %13 = arith.addi %c0_i32, %c40_i32 {constant = 40 : i32} : i32
    cgra.cond_br<ge> [%11 : i32, %13 : i32], ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %14 = arith.muli %11, %c4_i32 : i32
    %15 = arith.addi %14, %c132_i32 : i32
    %16 = cgra.lwi %15 : i32->i32
    %17 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cgra.cond_br<ge> [%16 : i32, %17 : i32], ^bb5(%16 : i32), ^bb3
  ^bb3:  // pred: ^bb2
    cgra.cond_br<eq> [%16 : i32, %5 : i32], ^bb5(%8 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %18 = arith.subi %c0_i32_6, %16 : i32
    cf.br ^bb5(%18 : i32)
  ^bb5(%19: i32):  // 3 preds: ^bb2, ^bb3, ^bb4
    %20 = arith.subi %12, %19 : i32
    %21 = cgra.bsfa %20 : i32 [%19, %12]  : i32
    %22 = arith.addi %11, %c1_i32 : i32
    cf.br ^bb1(%22, %21 : i32, i32)
  ^bb6:  // pred: ^bb1
    cgra.swi %12, %c128_i32 : i32, i32
    return %c0_i32_6 : i32
  }
}

