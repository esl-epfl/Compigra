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
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c16 step %c1 {
      %26 = memref.load %arg1[%arg3] : memref<16xi32>
      memref.store %26, %arg2[%arg3] : memref<80xi32>
    }
    %c16_0 = arith.constant 16 : index
    %c80 = arith.constant 80 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg3 = %c16_0 to %c80 step %c1_1 {
      %c-3 = arith.constant -3 : index
      %26 = arith.addi %arg3, %c-3 : index
      %27 = memref.load %arg2[%26] : memref<80xi32>
      %c-8 = arith.constant -8 : index
      %28 = arith.addi %arg3, %c-8 : index
      %29 = memref.load %arg2[%28] : memref<80xi32>
      %30 = arith.xori %27, %29 : i32
      %c-14 = arith.constant -14 : index
      %31 = arith.addi %arg3, %c-14 : index
      %32 = memref.load %arg2[%31] : memref<80xi32>
      %33 = arith.xori %30, %32 : i32
      %c-16 = arith.constant -16 : index
      %34 = arith.addi %arg3, %c-16 : index
      %35 = memref.load %arg2[%34] : memref<80xi32>
      %36 = arith.xori %33, %35 : i32
      memref.store %36, %arg2[%arg3] : memref<80xi32>
    }
    %0 = memref.get_global @A : memref<1xi32>
    %c0_2 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0_2] : memref<5xi32>
    %c0_3 = arith.constant 0 : index
    memref.store %1, %0[%c0_3] : memref<1xi32>
    %2 = memref.get_global @B : memref<1xi32>
    %c1_4 = arith.constant 1 : index
    %3 = memref.load %arg0[%c1_4] : memref<5xi32>
    %c0_5 = arith.constant 0 : index
    memref.store %3, %2[%c0_5] : memref<1xi32>
    %4 = memref.get_global @C : memref<1xi32>
    %c2 = arith.constant 2 : index
    %5 = memref.load %arg0[%c2] : memref<5xi32>
    %c0_6 = arith.constant 0 : index
    memref.store %5, %4[%c0_6] : memref<1xi32>
    %6 = memref.get_global @D : memref<1xi32>
    %c3 = arith.constant 3 : index
    %7 = memref.load %arg0[%c3] : memref<5xi32>
    %c0_7 = arith.constant 0 : index
    memref.store %7, %6[%c0_7] : memref<1xi32>
    %8 = memref.get_global @E : memref<1xi32>
    %c4 = arith.constant 4 : index
    %9 = memref.load %arg0[%c4] : memref<5xi32>
    %c0_8 = arith.constant 0 : index
    memref.store %9, %8[%c0_8] : memref<1xi32>
    %10 = memref.get_global @temp : memref<1xi32>
    %c0_9 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c1_10 = arith.constant 1 : index
    scf.for %arg3 = %c0_9 to %c20 step %c1_10 {
      %c0_33 = arith.constant 0 : index
      %26 = memref.load %0[%c0_33] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %c0_34 = arith.constant 0 : index
      %30 = memref.load %2[%c0_34] : memref<1xi32>
      %c0_35 = arith.constant 0 : index
      %31 = memref.load %4[%c0_35] : memref<1xi32>
      %32 = arith.andi %30, %31 : i32
      %33 = arith.xori %30, %c-1_i32 : i32
      %c0_36 = arith.constant 0 : index
      %34 = memref.load %6[%c0_36] : memref<1xi32>
      %35 = arith.andi %33, %34 : i32
      %36 = arith.ori %32, %35 : i32
      %37 = arith.addi %29, %36 : i32
      %c0_37 = arith.constant 0 : index
      %38 = memref.load %8[%c0_37] : memref<1xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = memref.load %arg2[%arg3] : memref<80xi32>
      %41 = arith.addi %39, %40 : i32
      %42 = arith.extsi %41 : i32 to i64
      %43 = arith.addi %42, %c1518500249_i64 : i64
      %44 = arith.trunci %43 : i64 to i32
      %c0_38 = arith.constant 0 : index
      memref.store %44, %10[%c0_38] : memref<1xi32>
      %c0_39 = arith.constant 0 : index
      %45 = memref.load %6[%c0_39] : memref<1xi32>
      %c0_40 = arith.constant 0 : index
      memref.store %45, %8[%c0_40] : memref<1xi32>
      %c0_41 = arith.constant 0 : index
      %46 = memref.load %4[%c0_41] : memref<1xi32>
      %c0_42 = arith.constant 0 : index
      memref.store %46, %6[%c0_42] : memref<1xi32>
      %c0_43 = arith.constant 0 : index
      %47 = memref.load %2[%c0_43] : memref<1xi32>
      %48 = arith.shli %47, %c30_i32 : i32
      %49 = arith.shrsi %47, %c2_i32 : i32
      %50 = arith.ori %48, %49 : i32
      %c0_44 = arith.constant 0 : index
      memref.store %50, %4[%c0_44] : memref<1xi32>
      %c0_45 = arith.constant 0 : index
      %51 = memref.load %0[%c0_45] : memref<1xi32>
      %c0_46 = arith.constant 0 : index
      memref.store %51, %2[%c0_46] : memref<1xi32>
      %c0_47 = arith.constant 0 : index
      %52 = memref.load %10[%c0_47] : memref<1xi32>
      %c0_48 = arith.constant 0 : index
      memref.store %52, %0[%c0_48] : memref<1xi32>
    }
    %c20_11 = arith.constant 20 : index
    %c40 = arith.constant 40 : index
    %c1_12 = arith.constant 1 : index
    scf.for %arg3 = %c20_11 to %c40 step %c1_12 {
      %c0_33 = arith.constant 0 : index
      %26 = memref.load %0[%c0_33] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %c0_34 = arith.constant 0 : index
      %30 = memref.load %2[%c0_34] : memref<1xi32>
      %c0_35 = arith.constant 0 : index
      %31 = memref.load %4[%c0_35] : memref<1xi32>
      %32 = arith.xori %30, %31 : i32
      %c0_36 = arith.constant 0 : index
      %33 = memref.load %6[%c0_36] : memref<1xi32>
      %34 = arith.xori %32, %33 : i32
      %35 = arith.addi %29, %34 : i32
      %c0_37 = arith.constant 0 : index
      %36 = memref.load %8[%c0_37] : memref<1xi32>
      %37 = arith.addi %35, %36 : i32
      %38 = memref.load %arg2[%arg3] : memref<80xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.addi %40, %c1859775393_i64 : i64
      %42 = arith.trunci %41 : i64 to i32
      %c0_38 = arith.constant 0 : index
      memref.store %42, %10[%c0_38] : memref<1xi32>
      %c0_39 = arith.constant 0 : index
      %43 = memref.load %6[%c0_39] : memref<1xi32>
      %c0_40 = arith.constant 0 : index
      memref.store %43, %8[%c0_40] : memref<1xi32>
      %c0_41 = arith.constant 0 : index
      %44 = memref.load %4[%c0_41] : memref<1xi32>
      %c0_42 = arith.constant 0 : index
      memref.store %44, %6[%c0_42] : memref<1xi32>
      %c0_43 = arith.constant 0 : index
      %45 = memref.load %2[%c0_43] : memref<1xi32>
      %46 = arith.shli %45, %c30_i32 : i32
      %47 = arith.shrsi %45, %c2_i32 : i32
      %48 = arith.ori %46, %47 : i32
      %c0_44 = arith.constant 0 : index
      memref.store %48, %4[%c0_44] : memref<1xi32>
      %c0_45 = arith.constant 0 : index
      %49 = memref.load %0[%c0_45] : memref<1xi32>
      %c0_46 = arith.constant 0 : index
      memref.store %49, %2[%c0_46] : memref<1xi32>
      %c0_47 = arith.constant 0 : index
      %50 = memref.load %10[%c0_47] : memref<1xi32>
      %c0_48 = arith.constant 0 : index
      memref.store %50, %0[%c0_48] : memref<1xi32>
    }
    %c40_13 = arith.constant 40 : index
    %c60 = arith.constant 60 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg3 = %c40_13 to %c60 step %c1_14 {
      %c0_33 = arith.constant 0 : index
      %26 = memref.load %0[%c0_33] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %c0_34 = arith.constant 0 : index
      %30 = memref.load %2[%c0_34] : memref<1xi32>
      %c0_35 = arith.constant 0 : index
      %31 = memref.load %4[%c0_35] : memref<1xi32>
      %32 = arith.andi %30, %31 : i32
      %c0_36 = arith.constant 0 : index
      %33 = memref.load %6[%c0_36] : memref<1xi32>
      %34 = arith.andi %30, %33 : i32
      %35 = arith.ori %32, %34 : i32
      %36 = arith.andi %31, %33 : i32
      %37 = arith.ori %35, %36 : i32
      %38 = arith.addi %29, %37 : i32
      %c0_37 = arith.constant 0 : index
      %39 = memref.load %8[%c0_37] : memref<1xi32>
      %40 = arith.addi %38, %39 : i32
      %41 = memref.load %arg2[%arg3] : memref<80xi32>
      %42 = arith.addi %40, %41 : i32
      %43 = arith.extsi %42 : i32 to i64
      %44 = arith.addi %43, %c1327217884_i64 : i64
      %45 = arith.trunci %44 : i64 to i32
      %c0_38 = arith.constant 0 : index
      memref.store %45, %10[%c0_38] : memref<1xi32>
      %c0_39 = arith.constant 0 : index
      %46 = memref.load %6[%c0_39] : memref<1xi32>
      %c0_40 = arith.constant 0 : index
      memref.store %46, %8[%c0_40] : memref<1xi32>
      %c0_41 = arith.constant 0 : index
      %47 = memref.load %4[%c0_41] : memref<1xi32>
      %c0_42 = arith.constant 0 : index
      memref.store %47, %6[%c0_42] : memref<1xi32>
      %c0_43 = arith.constant 0 : index
      %48 = memref.load %2[%c0_43] : memref<1xi32>
      %49 = arith.shli %48, %c30_i32 : i32
      %50 = arith.shrsi %48, %c2_i32 : i32
      %51 = arith.ori %49, %50 : i32
      %c0_44 = arith.constant 0 : index
      memref.store %51, %4[%c0_44] : memref<1xi32>
      %c0_45 = arith.constant 0 : index
      %52 = memref.load %0[%c0_45] : memref<1xi32>
      %c0_46 = arith.constant 0 : index
      memref.store %52, %2[%c0_46] : memref<1xi32>
      %c0_47 = arith.constant 0 : index
      %53 = memref.load %10[%c0_47] : memref<1xi32>
      %c0_48 = arith.constant 0 : index
      memref.store %53, %0[%c0_48] : memref<1xi32>
    }
    %c60_15 = arith.constant 60 : index
    %c80_16 = arith.constant 80 : index
    %c1_17 = arith.constant 1 : index
    scf.for %arg3 = %c60_15 to %c80_16 step %c1_17 {
      %c0_33 = arith.constant 0 : index
      %26 = memref.load %0[%c0_33] : memref<1xi32>
      %27 = arith.shli %26, %c5_i32 : i32
      %28 = arith.shrsi %26, %c27_i32 : i32
      %29 = arith.ori %27, %28 : i32
      %c0_34 = arith.constant 0 : index
      %30 = memref.load %2[%c0_34] : memref<1xi32>
      %c0_35 = arith.constant 0 : index
      %31 = memref.load %4[%c0_35] : memref<1xi32>
      %32 = arith.xori %30, %31 : i32
      %c0_36 = arith.constant 0 : index
      %33 = memref.load %6[%c0_36] : memref<1xi32>
      %34 = arith.xori %32, %33 : i32
      %35 = arith.addi %29, %34 : i32
      %c0_37 = arith.constant 0 : index
      %36 = memref.load %8[%c0_37] : memref<1xi32>
      %37 = arith.addi %35, %36 : i32
      %38 = memref.load %arg2[%arg3] : memref<80xi32>
      %39 = arith.addi %37, %38 : i32
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.addi %40, %c1247986134_i64 : i64
      %42 = arith.trunci %41 : i64 to i32
      %c0_38 = arith.constant 0 : index
      memref.store %42, %10[%c0_38] : memref<1xi32>
      %c0_39 = arith.constant 0 : index
      %43 = memref.load %6[%c0_39] : memref<1xi32>
      %c0_40 = arith.constant 0 : index
      memref.store %43, %8[%c0_40] : memref<1xi32>
      %c0_41 = arith.constant 0 : index
      %44 = memref.load %4[%c0_41] : memref<1xi32>
      %c0_42 = arith.constant 0 : index
      memref.store %44, %6[%c0_42] : memref<1xi32>
      %c0_43 = arith.constant 0 : index
      %45 = memref.load %2[%c0_43] : memref<1xi32>
      %46 = arith.shli %45, %c30_i32 : i32
      %47 = arith.shrsi %45, %c2_i32 : i32
      %48 = arith.ori %46, %47 : i32
      %c0_44 = arith.constant 0 : index
      memref.store %48, %4[%c0_44] : memref<1xi32>
      %c0_45 = arith.constant 0 : index
      %49 = memref.load %0[%c0_45] : memref<1xi32>
      %c0_46 = arith.constant 0 : index
      memref.store %49, %2[%c0_46] : memref<1xi32>
      %c0_47 = arith.constant 0 : index
      %50 = memref.load %10[%c0_47] : memref<1xi32>
      %c0_48 = arith.constant 0 : index
      memref.store %50, %0[%c0_48] : memref<1xi32>
    }
    %c0_18 = arith.constant 0 : index
    %11 = memref.load %0[%c0_18] : memref<1xi32>
    %c0_19 = arith.constant 0 : index
    %12 = memref.load %arg0[%c0_19] : memref<5xi32>
    %13 = arith.addi %12, %11 : i32
    %c0_20 = arith.constant 0 : index
    memref.store %13, %arg0[%c0_20] : memref<5xi32>
    %c0_21 = arith.constant 0 : index
    %14 = memref.load %2[%c0_21] : memref<1xi32>
    %c1_22 = arith.constant 1 : index
    %15 = memref.load %arg0[%c1_22] : memref<5xi32>
    %16 = arith.addi %15, %14 : i32
    %c1_23 = arith.constant 1 : index
    memref.store %16, %arg0[%c1_23] : memref<5xi32>
    %c0_24 = arith.constant 0 : index
    %17 = memref.load %4[%c0_24] : memref<1xi32>
    %c2_25 = arith.constant 2 : index
    %18 = memref.load %arg0[%c2_25] : memref<5xi32>
    %19 = arith.addi %18, %17 : i32
    %c2_26 = arith.constant 2 : index
    memref.store %19, %arg0[%c2_26] : memref<5xi32>
    %c0_27 = arith.constant 0 : index
    %20 = memref.load %6[%c0_27] : memref<1xi32>
    %c3_28 = arith.constant 3 : index
    %21 = memref.load %arg0[%c3_28] : memref<5xi32>
    %22 = arith.addi %21, %20 : i32
    %c3_29 = arith.constant 3 : index
    memref.store %22, %arg0[%c3_29] : memref<5xi32>
    %c0_30 = arith.constant 0 : index
    %23 = memref.load %8[%c0_30] : memref<1xi32>
    %c4_31 = arith.constant 4 : index
    %24 = memref.load %arg0[%c4_31] : memref<5xi32>
    %25 = arith.addi %24, %23 : i32
    %c4_32 = arith.constant 4 : index
    memref.store %25, %arg0[%c4_32] : memref<5xi32>
    return
  }
}

