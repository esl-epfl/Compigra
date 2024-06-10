module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @SHA1(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c80_i32 = arith.constant 80 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-8_i32 = arith.constant -8 : i32
    %c-14_i32 = arith.constant -14 : i32
    %c-16_i32 = arith.constant -16 : i32
    cf.br ^bb1(%c16_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb1
    %1 = arith.cmpi slt, %0, %c80_i32 : i32
    %2 = arith.addi %0, %c-3_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = memref.load %arg0[%3] : memref<?xi32>
    %5 = arith.addi %0, %c-8_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg0[%6] : memref<?xi32>
    %8 = arith.xori %4, %7 : i32
    %9 = arith.addi %0, %c-14_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = memref.load %arg0[%10] : memref<?xi32>
    %12 = arith.xori %8, %11 : i32
    %13 = arith.addi %0, %c-16_i32 : i32
    %14 = arith.index_cast %13 : i32 to index
    %15 = memref.load %arg0[%14] : memref<?xi32>
    %16 = arith.xori %12, %15 : i32
    %17 = arith.index_cast %0 : i32 to index
    memref.store %16, %arg0[%17] : memref<?xi32>
    %18 = arith.addi %0, %c1_i32 : i32
    cf.cond_br %1, ^bb1(%18 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    return
  }
}

