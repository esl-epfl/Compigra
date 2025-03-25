Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
Set parameter Username
Academic license - for non-commercial use only - expires 2025-06-22
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @FIR(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant {BaseAddr = "arg2"} 256 : i32
    %c192_i32 = arith.constant {BaseAddr = "arg1"} 192 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    %c0_i32_9 = arith.constant 0 : i32
    cf.br ^bb1(%0 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb7
    %2 = arith.addi %c0_i32, %c16_i32 {constant = 16 : i32} : i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb8, ^bb2
  ^bb2:  // pred: ^bb1
    %3 = arith.muli %1, %c4_i32 : i32
    %4 = arith.addi %3, %c192_i32 : i32
    %5 = arith.addi %c0_i32_6, %c0_i32_7 {constant = 0 : i32} : i32
    cgra.swi %5, %4 : i32, i32
    %6 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    %c0_i32_10 = arith.constant 0 : i32
    cgra.swi %6, %c0_i32_10 : i32, i32 {memLoc = 0 : i32}
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb6
    %7 = arith.addi %c0_i32_8, %c5_i32 {constant = 5 : i32} : i32
    %c0_i32_11 = arith.constant 0 : i32
    %8 = cgra.lwi %c0_i32_11 : i32->i32
    cgra.cond_br<ge> [%8 : i32, %7 : i32], ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %c0_i32_12 = arith.constant 0 : i32
    %9 = cgra.lwi %c0_i32_12 : i32->i32
    %10 = arith.subi %1, %9 : i32
    %11 = arith.addi %c0_i32_0, %c0_i32_1 {constant = 0 : i32} : i32
    cgra.cond_br<lt> [%10 : i32, %11 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %12 = cgra.lwi %c0_i32_9 : i32->i32
    %13 = arith.muli %12, %c4_i32 : i32
    %14 = arith.addi %13, %c256_i32 : i32
    %15 = cgra.lwi %14 : i32->i32
    %16 = arith.subi %1, %12 : i32
    %17 = arith.muli %16, %c4_i32 : i32
    %18 = arith.addi %17, %c128_i32 : i32
    %19 = cgra.lwi %18 : i32->i32
    %20 = arith.muli %15, %19 : i32
    %21 = arith.muli %1, %c4_i32 : i32
    %22 = arith.addi %21, %c192_i32 : i32
    %23 = cgra.lwi %22 : i32->i32
    %24 = arith.addi %23, %20 : i32
    %25 = arith.muli %1, %c4_i32 : i32
    %26 = arith.addi %25, %c192_i32 : i32
    cgra.swi %24, %26 : i32, i32
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %c0_i32_13 = arith.constant 0 : i32
    %27 = cgra.lwi %c0_i32_13 : i32->i32
    %28 = arith.addi %27, %c1_i32 : i32
    %c0_i32_14 = arith.constant 0 : i32
    cgra.swi %28, %c0_i32_14 : i32, i32 {memLoc = 0 : i32}
    cf.br ^bb3
  ^bb7:  // pred: ^bb3
    %29 = arith.addi %1, %c1_i32 : i32
    cf.br ^bb1(%29 : i32)
  ^bb8:  // pred: ^bb1
    return
  }
}

