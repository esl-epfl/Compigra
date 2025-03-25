module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @temp : memref<1xi32> = uninitialized
  memref.global @E : memref<1xi32> = uninitialized
  memref.global @D : memref<1xi32> = uninitialized
  memref.global @C : memref<1xi32> = uninitialized
  memref.global @B : memref<1xi32> = uninitialized
  memref.global @A : memref<1xi32> = uninitialized
  func.func @sha_transform(%arg0: memref<5xi32>, %arg1: memref<16xi32>, %arg2: memref<80xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1247986134_i64 = arith.constant 1247986134 : i64
    %c1327217884_i64 = arith.constant 1327217884 : i64
    %c1859775393_i64 = arith.constant 1859775393 : i64
    %c2_i32 = arith.constant 2 : i32
    %c30_i32 = arith.constant 30 : i32
    %c1518500249_i64 = arith.constant 1518500249 : i64
    %c-1_i32 = arith.constant -1 : i32
    %c27_i32 = arith.constant 27 : i32
    %c5_i32 = arith.constant 5 : i32
    affine.for %arg3 = 0 to 16 {
      %26 = affine.load %arg1[%arg3] : memref<16xi32>
      affine.store %26, %arg2[%arg3] : memref<80xi32>
    }
    affine.for %arg3 = 16 to 80 {
      %26 = affine.load %arg2[%arg3 - 3] : memref<80xi32>
      %27 = affine.load %arg2[%arg3 - 8] : memref<80xi32>
      %28 = arith.xori %26, %27 : i32
      %29 = affine.load %arg2[%arg3 - 14] : memref<80xi32>
      %30 = arith.xori %28, %29 : i32
      %31 = affine.load %arg2[%arg3 - 16] : memref<80xi32>
      %32 = arith.xori %30, %31 : i32
      affine.store %32, %arg2[%arg3] : memref<80xi32>
    }
    %0 = memref.get_global @A : memref<1xi32>
    %1 = affine.load %arg0[0] : memref<5xi32>
    affine.store %1, %0[0] : memref<1xi32>
    %2 = memref.get_global @B : memref<1xi32>
    %3 = affine.load %arg0[1] : memref<5xi32>
    affine.store %3, %2[0] : memref<1xi32>
    %4 = memref.get_global @C : memref<1xi32>
    %5 = affine.load %arg0[2] : memref<5xi32>
    affine.store %5, %4[0] : memref<1xi32>
    %6 = memref.get_global @D : memref<1xi32>
    %7 = affine.load %arg0[3] : memref<5xi32>
    affine.store %7, %6[0] : memref<1xi32>
    %8 = memref.get_global @E : memref<1xi32>
    %9 = affine.load %arg0[4] : memref<5xi32>
    affine.store %9, %8[0] : memref<1xi32>
    %10 = memref.get_global @temp : memref<1xi32>
    affine.for %arg3 = 0 to 20 {
      %26 = affine.load %0[0] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %30 = affine.load %2[0] : memref<1xi32>
      %31 = affine.load %4[0] : memref<1xi32>
      %32 = arith.andi %30, %31 : i32
      %33 = arith.xori %30, %c-1_i32 : i32
      %34 = affine.load %6[0] : memref<1xi32>
      %35 = arith.andi %33, %34 : i32
      %36 = arith.ori %32, %35 : i32
      %37 = arith.addi %29, %36 : i32
      %38 = affine.load %8[0] : memref<1xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = affine.load %arg2[%arg3] : memref<80xi32>
      %41 = arith.addi %39, %40 : i32
      %42 = arith.extsi %41 : i32 to i64
      %43 = arith.addi %42, %c1518500249_i64 : i64
      %44 = arith.trunci %43 : i64 to i32
      affine.store %44, %10[0] : memref<1xi32>
      %45 = affine.load %6[0] : memref<1xi32>
      affine.store %45, %8[0] : memref<1xi32>
      %46 = affine.load %4[0] : memref<1xi32>
      affine.store %46, %6[0] : memref<1xi32>
      %47 = affine.load %2[0] : memref<1xi32>
      %48 = arith.shli %47, %c30_i32 : i32
      %49 = arith.shrsi %47, %c2_i32 : i32
      %50 = arith.ori %48, %49 : i32
      affine.store %50, %4[0] : memref<1xi32>
      %51 = affine.load %0[0] : memref<1xi32>
      affine.store %51, %2[0] : memref<1xi32>
      %52 = affine.load %10[0] : memref<1xi32>
      affine.store %52, %0[0] : memref<1xi32>
    }
    affine.for %arg3 = 20 to 40 {
      %26 = affine.load %0[0] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %30 = affine.load %2[0] : memref<1xi32>
      %31 = affine.load %4[0] : memref<1xi32>
      %32 = arith.xori %30, %31 : i32
      %33 = affine.load %6[0] : memref<1xi32>
      %34 = arith.xori %32, %33 : i32
      %35 = arith.addi %29, %34 : i32
      %36 = affine.load %8[0] : memref<1xi32>
      %37 = arith.addi %35, %36 : i32
      %38 = affine.load %arg2[%arg3] : memref<80xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.addi %40, %c1859775393_i64 : i64
      %42 = arith.trunci %41 : i64 to i32
      affine.store %42, %10[0] : memref<1xi32>
      %43 = affine.load %6[0] : memref<1xi32>
      affine.store %43, %8[0] : memref<1xi32>
      %44 = affine.load %4[0] : memref<1xi32>
      affine.store %44, %6[0] : memref<1xi32>
      %45 = affine.load %2[0] : memref<1xi32>
      %46 = arith.shli %45, %c30_i32 : i32
      %47 = arith.shrsi %45, %c2_i32 : i32
      %48 = arith.ori %46, %47 : i32
      affine.store %48, %4[0] : memref<1xi32>
      %49 = affine.load %0[0] : memref<1xi32>
      affine.store %49, %2[0] : memref<1xi32>
      %50 = affine.load %10[0] : memref<1xi32>
      affine.store %50, %0[0] : memref<1xi32>
    }
    affine.for %arg3 = 40 to 60 {
      %26 = affine.load %0[0] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %30 = affine.load %2[0] : memref<1xi32>
      %31 = affine.load %4[0] : memref<1xi32>
      %32 = arith.andi %30, %31 : i32
      %33 = affine.load %6[0] : memref<1xi32>
      %34 = arith.andi %30, %33 : i32
      %35 = arith.ori %32, %34 : i32
      %36 = arith.andi %31, %33 : i32
      %37 = arith.ori %35, %36 : i32
      %38 = arith.addi %29, %37 : i32
      %39 = affine.load %8[0] : memref<1xi32>
      %40 = arith.addi %38, %39 : i32
      %41 = affine.load %arg2[%arg3] : memref<80xi32>
      %42 = arith.addi %40, %41 : i32
      %43 = arith.extsi %42 : i32 to i64
      %44 = arith.addi %43, %c1327217884_i64 : i64
      %45 = arith.trunci %44 : i64 to i32
      affine.store %45, %10[0] : memref<1xi32>
      %46 = affine.load %6[0] : memref<1xi32>
      affine.store %46, %8[0] : memref<1xi32>
      %47 = affine.load %4[0] : memref<1xi32>
      affine.store %47, %6[0] : memref<1xi32>
      %48 = affine.load %2[0] : memref<1xi32>
      %49 = arith.shli %48, %c30_i32 : i32
      %50 = arith.shrsi %48, %c2_i32 : i32
      %51 = arith.ori %49, %50 : i32
      affine.store %51, %4[0] : memref<1xi32>
      %52 = affine.load %0[0] : memref<1xi32>
      affine.store %52, %2[0] : memref<1xi32>
      %53 = affine.load %10[0] : memref<1xi32>
      affine.store %53, %0[0] : memref<1xi32>
    }
    affine.for %arg3 = 60 to 80 {
      %26 = affine.load %0[0] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %30 = affine.load %2[0] : memref<1xi32>
      %31 = affine.load %4[0] : memref<1xi32>
      %32 = arith.xori %30, %31 : i32
      %33 = affine.load %6[0] : memref<1xi32>
      %34 = arith.xori %32, %33 : i32
      %35 = arith.addi %29, %34 : i32
      %36 = affine.load %8[0] : memref<1xi32>
      %37 = arith.addi %35, %36 : i32
      %38 = affine.load %arg2[%arg3] : memref<80xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.addi %40, %c1247986134_i64 : i64
      %42 = arith.trunci %41 : i64 to i32
      affine.store %42, %10[0] : memref<1xi32>
      %43 = affine.load %6[0] : memref<1xi32>
      affine.store %43, %8[0] : memref<1xi32>
      %44 = affine.load %4[0] : memref<1xi32>
      affine.store %44, %6[0] : memref<1xi32>
      %45 = affine.load %2[0] : memref<1xi32>
      %46 = arith.shli %45, %c30_i32 : i32
      %47 = arith.shrsi %45, %c2_i32 : i32
      %48 = arith.ori %46, %47 : i32
      affine.store %48, %4[0] : memref<1xi32>
      %49 = affine.load %0[0] : memref<1xi32>
      affine.store %49, %2[0] : memref<1xi32>
      %50 = affine.load %10[0] : memref<1xi32>
      affine.store %50, %0[0] : memref<1xi32>
    }
    %11 = affine.load %0[0] : memref<1xi32>
    %12 = affine.load %arg0[0] : memref<5xi32>
    %13 = arith.addi %12, %11 : i32
    affine.store %13, %arg0[0] : memref<5xi32>
    %14 = affine.load %2[0] : memref<1xi32>
    %15 = affine.load %arg0[1] : memref<5xi32>
    %16 = arith.addi %15, %14 : i32
    affine.store %16, %arg0[1] : memref<5xi32>
    %17 = affine.load %4[0] : memref<1xi32>
    %18 = affine.load %arg0[2] : memref<5xi32>
    %19 = arith.addi %18, %17 : i32
    affine.store %19, %arg0[2] : memref<5xi32>
    %20 = affine.load %6[0] : memref<1xi32>
    %21 = affine.load %arg0[3] : memref<5xi32>
    %22 = arith.addi %21, %20 : i32
    affine.store %22, %arg0[3] : memref<5xi32>
    %23 = affine.load %8[0] : memref<1xi32>
    %24 = affine.load %arg0[4] : memref<5xi32>
    %25 = arith.addi %24, %23 : i32
    affine.store %25, %arg0[4] : memref<5xi32>
    return
  }
}
