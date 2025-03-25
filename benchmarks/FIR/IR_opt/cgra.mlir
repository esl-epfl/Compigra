module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @FIR(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant {BaseAddr = "arg2"} 256 : i32
    %c192_i32 = arith.constant {BaseAddr = "arg1"} 192 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb7
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %c16 : index to i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb8, ^bb2
  ^bb2:  // pred: ^bb1
    %3 = arith.index_cast %0 : index to i32
    %4 = arith.muli %3, %c4_i32 : i32
    %5 = arith.addi %c192_i32, %4 : i32
    cgra.swi %c0_i32, %5 : i32, i32
    cf.br ^bb3(%c0 : index)
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb6
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.index_cast %c5 : index to i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %9 = arith.subi %0, %6 : index
    %10 = arith.index_cast %9 : index to i32
    %11 = arith.index_cast %c0 : index to i32
    cgra.cond_br<lt> [%10 : i32, %11 : i32], ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %12 = arith.index_cast %6 : index to i32
    %13 = arith.muli %12, %c4_i32 : i32
    %14 = arith.addi %c256_i32, %13 : i32
    %15 = cgra.lwi %14 : i32->i32
    %16 = arith.subi %0, %6 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.muli %17, %c4_i32 : i32
    %19 = arith.addi %c128_i32, %18 : i32
    %20 = cgra.lwi %19 : i32->i32
    %21 = arith.muli %15, %20 : i32
    %22 = arith.index_cast %0 : index to i32
    %23 = arith.muli %22, %c4_i32 : i32
    %24 = arith.addi %c192_i32, %23 : i32
    %25 = cgra.lwi %24 : i32->i32
    %26 = arith.addi %25, %21 : i32
    %27 = arith.index_cast %0 : index to i32
    %28 = arith.muli %27, %c4_i32 : i32
    %29 = arith.addi %c192_i32, %28 : i32
    cgra.swi %26, %29 : i32, i32
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %30 = arith.addi %6, %c1 : index
    cf.br ^bb3(%30 : index)
  ^bb7:  // pred: ^bb3
    %31 = arith.addi %0, %c1 : index
    cf.br ^bb1(%31 : index)
  ^bb8:  // pred: ^bb1
    return
  }
}

