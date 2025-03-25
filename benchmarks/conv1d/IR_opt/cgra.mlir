module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @conv1d(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<64xf32>, %arg3: memref<3xf32>, %arg4: memref<62xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c404_i32 = arith.constant {BaseAddr = "arg4"} 404 : i32
    %c392_i32 = arith.constant {BaseAddr = "arg3"} 392 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c128_i32 : i32->i32
    %1 = cgra.lwi %c132_i32 : i32->i32
    %2 = arith.subi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.addi %3, %c1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%6: index):  // 2 preds: ^bb0, ^bb5
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.index_cast %5 : index to i32
    cgra.cond_br<ge> [%7 : i32, %8 : i32], ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %9 = arith.index_cast %6 : index to i32
    %10 = arith.muli %9, %c4_i32 : i32
    %11 = arith.addi %c404_i32, %10 : i32
    cgra.swi %cst, %11 : f32, i32
    cf.br ^bb3(%c0 : index)
  ^bb3(%12: index):  // 2 preds: ^bb2, ^bb4
    %13 = arith.index_cast %12 : index to i32
    %14 = arith.index_cast %4 : index to i32
    cgra.cond_br<ge> [%13 : i32, %14 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %15 = arith.addi %6, %12 : index
    %16 = arith.index_cast %15 : index to i32
    %17 = arith.muli %16, %c4_i32 : i32
    %18 = arith.addi %c136_i32, %17 : i32
    %19 = cgra.lwi %18 : i32->f32
    %20 = arith.index_cast %12 : index to i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.addi %c392_i32, %21 : i32
    %23 = cgra.lwi %22 : i32->f32
    %24 = arith.mulf %19, %23 : f32
    %25 = arith.index_cast %6 : index to i32
    %26 = arith.muli %25, %c4_i32 : i32
    %27 = arith.addi %c404_i32, %26 : i32
    %28 = cgra.lwi %27 : i32->f32
    %29 = arith.addf %28, %24 : f32
    %30 = arith.index_cast %6 : index to i32
    %31 = arith.muli %30, %c4_i32 : i32
    %32 = arith.addi %c404_i32, %31 : i32
    cgra.swi %29, %32 : f32, i32
    %33 = arith.addi %12, %c1 : index
    cf.br ^bb3(%33 : index)
  ^bb5:  // pred: ^bb3
    %34 = arith.addi %6, %c1 : index
    cf.br ^bb1(%34 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}

