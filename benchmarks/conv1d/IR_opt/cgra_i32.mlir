module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv1d(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<64xf32>, %arg3: memref<3xf32>, %arg4: memref<62xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c404_i32 = arith.constant {BaseAddr = "arg4"} 404 : i32
    %c392_i32 = arith.constant {BaseAddr = "arg3"} 392 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = arith.subi %0, %1 : i32
    %3 = arith.addi %2, %c1_i32 : i32
    %4 = arith.addi %c0_i32_1, %c0_i32_2 {constant = 0 : i32} : i32
    cf.br ^bb1(%4 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb5
    cgra.cond_br<ge> [%5 : i32, %3 : i32], ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %6 = arith.muli %5, %c4_i32 : i32
    %7 = arith.addi %6, %c404_i32 : i32
    cgra.swi %cst, %7 : f32, i32
    %8 = arith.addi %c0_i32, %c0_i32_0 {constant = 0 : i32} : i32
    cf.br ^bb3(%8 : i32)
  ^bb3(%9: i32):  // 2 preds: ^bb2, ^bb4
    cgra.cond_br<ge> [%9 : i32, %1 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %10 = arith.addi %5, %9 : i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.addi %11, %c136_i32 : i32
    %13 = cgra.lwi %12 : i32->f32
    %14 = arith.muli %9, %c4_i32 : i32
    %15 = arith.addi %14, %c392_i32 : i32
    %16 = cgra.lwi %15 : i32->f32
    %17 = arith.mulf %13, %16 : f32
    %18 = arith.muli %5, %c4_i32 : i32
    %19 = arith.addi %18, %c404_i32 : i32
    %20 = cgra.lwi %19 : i32->f32
    %21 = arith.addf %20, %17 : f32
    %22 = arith.muli %5, %c4_i32 : i32
    %23 = arith.addi %22, %c404_i32 : i32
    cgra.swi %21, %23 : f32, i32
    %24 = arith.addi %9, %c1_i32 : i32
    cf.br ^bb3(%24 : i32)
  ^bb5:  // pred: ^bb3
    %25 = arith.addi %5, %c1_i32 : i32
    cf.br ^bb1(%25 : i32)
  ^bb6:  // pred: ^bb1
    return
  }
}

