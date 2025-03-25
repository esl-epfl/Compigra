module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @symm(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant {DimProd = 1 : i32, arg = 4 : i32} 10 : i32
    %c936_i32 = arith.constant {BaseAddr = "arg4"} 936 : i32
    %c10_i32_0 = arith.constant {DimProd = 1 : i32, arg = 3 : i32} 10 : i32
    %c536_i32 = arith.constant {BaseAddr = "arg3"} 536 : i32
    %c10_i32_1 = arith.constant {DimProd = 1 : i32, arg = 2 : i32} 10 : i32
    %c136_i32 = arith.constant {BaseAddr = "arg2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "arg1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "arg0"} 128 : i32
    %0 = cgra.lwi %c132_i32 : i32->i32
    %1 = cgra.lwi %c128_i32 : i32->i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb6
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%3 : i32, %4 : i32], ^bb7, ^bb2(%c0 : index)
  ^bb2(%5: index):  // 2 preds: ^bb1, ^bb5
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.index_cast %c10 : index to i32
    cgra.cond_br<ge> [%6 : i32, %7 : i32], ^bb6, ^bb3(%c0, %c0_i32 : index, i32)
  ^bb3(%8: index, %9: i32):  // 2 preds: ^bb2, ^bb4
    %10 = arith.index_cast %8 : index to i32
    %11 = arith.index_cast %2 : index to i32
    cgra.cond_br<ge> [%10 : i32, %11 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %12 = arith.index_cast %2 : index to i32
    %13 = arith.muli %12, %c10_i32 : i32
    %14 = arith.index_cast %5 : index to i32
    %15 = arith.addi %13, %14 : i32
    %16 = arith.muli %15, %c4_i32 : i32
    %17 = arith.addi %c936_i32, %16 : i32
    %18 = cgra.lwi %17 : i32->i32
    %19 = arith.muli %1, %18 : i32
    %20 = arith.index_cast %2 : index to i32
    %21 = arith.muli %20, %c10_i32_0 : i32
    %22 = arith.index_cast %8 : index to i32
    %23 = arith.addi %21, %22 : i32
    %24 = arith.muli %23, %c4_i32 : i32
    %25 = arith.addi %c536_i32, %24 : i32
    %26 = cgra.lwi %25 : i32->i32
    %27 = arith.muli %19, %26 : i32
    %28 = arith.index_cast %8 : index to i32
    %29 = arith.muli %28, %c10_i32_1 : i32
    %30 = arith.index_cast %5 : index to i32
    %31 = arith.addi %29, %30 : i32
    %32 = arith.muli %31, %c4_i32 : i32
    %33 = arith.addi %c136_i32, %32 : i32
    %34 = cgra.lwi %33 : i32->i32
    %35 = arith.addi %34, %27 : i32
    %36 = arith.index_cast %8 : index to i32
    %37 = arith.muli %36, %c10_i32_1 : i32
    %38 = arith.index_cast %5 : index to i32
    %39 = arith.addi %37, %38 : i32
    %40 = arith.muli %39, %c4_i32 : i32
    %41 = arith.addi %c136_i32, %40 : i32
    cgra.swi %35, %41 : i32, i32
    %42 = arith.index_cast %8 : index to i32
    %43 = arith.muli %42, %c10_i32 : i32
    %44 = arith.index_cast %5 : index to i32
    %45 = arith.addi %43, %44 : i32
    %46 = arith.muli %45, %c4_i32 : i32
    %47 = arith.addi %c936_i32, %46 : i32
    %48 = cgra.lwi %47 : i32->i32
    %49 = arith.index_cast %2 : index to i32
    %50 = arith.muli %49, %c10_i32_0 : i32
    %51 = arith.index_cast %8 : index to i32
    %52 = arith.addi %50, %51 : i32
    %53 = arith.muli %52, %c4_i32 : i32
    %54 = arith.addi %c536_i32, %53 : i32
    %55 = cgra.lwi %54 : i32->i32
    %56 = arith.muli %48, %55 : i32
    %57 = arith.addi %9, %56 : i32
    %58 = arith.addi %8, %c1 : index
    cf.br ^bb3(%58, %57 : index, i32)
  ^bb5:  // pred: ^bb3
    %59 = arith.index_cast %2 : index to i32
    %60 = arith.muli %59, %c10_i32_1 : i32
    %61 = arith.index_cast %5 : index to i32
    %62 = arith.addi %60, %61 : i32
    %63 = arith.muli %62, %c4_i32 : i32
    %64 = arith.addi %c136_i32, %63 : i32
    %65 = cgra.lwi %64 : i32->i32
    %66 = arith.muli %0, %65 : i32
    %67 = arith.index_cast %2 : index to i32
    %68 = arith.muli %67, %c10_i32 : i32
    %69 = arith.index_cast %5 : index to i32
    %70 = arith.addi %68, %69 : i32
    %71 = arith.muli %70, %c4_i32 : i32
    %72 = arith.addi %c936_i32, %71 : i32
    %73 = cgra.lwi %72 : i32->i32
    %74 = arith.muli %1, %73 : i32
    %75 = arith.index_cast %2 : index to i32
    %76 = arith.muli %75, %c10_i32_0 : i32
    %77 = arith.index_cast %2 : index to i32
    %78 = arith.addi %76, %77 : i32
    %79 = arith.muli %78, %c4_i32 : i32
    %80 = arith.addi %c536_i32, %79 : i32
    %81 = cgra.lwi %80 : i32->i32
    %82 = arith.muli %74, %81 : i32
    %83 = arith.addi %66, %82 : i32
    %84 = arith.muli %1, %9 : i32
    %85 = arith.addi %83, %84 : i32
    %86 = arith.index_cast %2 : index to i32
    %87 = arith.muli %86, %c10_i32_1 : i32
    %88 = arith.index_cast %5 : index to i32
    %89 = arith.addi %87, %88 : i32
    %90 = arith.muli %89, %c4_i32 : i32
    %91 = arith.addi %c136_i32, %90 : i32
    cgra.swi %85, %91 : i32, i32
    %92 = arith.addi %5, %c1 : index
    cf.br ^bb2(%92 : index)
  ^bb6:  // pred: ^bb2
    %93 = arith.addi %2, %c1 : index
    cf.br ^bb1(%93 : index)
  ^bb7:  // pred: ^bb1
    return
  }
}

