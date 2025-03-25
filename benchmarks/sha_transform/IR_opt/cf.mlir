module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @temp : memref<1xi32> = uninitialized
  memref.global @E : memref<1xi32> = uninitialized
  memref.global @D : memref<1xi32> = uninitialized
  memref.global @C : memref<1xi32> = uninitialized
  memref.global @B : memref<1xi32> = uninitialized
  memref.global @A : memref<1xi32> = uninitialized
  func.func @sha_transform(%arg0: memref<5xi32>, %arg1: memref<16xi32>, %arg2: memref<80xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c60 = arith.constant 60 : index
    %c40 = arith.constant 40 : index
    %c20 = arith.constant 20 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c-16 = arith.constant -16 : index
    %c-14 = arith.constant -14 : index
    %c-8 = arith.constant -8 : index
    %c-3 = arith.constant -3 : index
    %c80 = arith.constant 80 : index
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
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c16 : index
    cf.cond_br %1, ^bb2, ^bb3(%c16 : index)
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg1[%0] : memref<16xi32>
    memref.store %2, %arg2[%0] : memref<80xi32>
    %3 = arith.addi %0, %c1 : index
    cf.br ^bb1(%3 : index)
  ^bb3(%4: index):  // 2 preds: ^bb1, ^bb4
    %5 = arith.cmpi slt, %4, %c80 : index
    cf.cond_br %5, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %6 = arith.addi %4, %c-3 : index
    %7 = memref.load %arg2[%6] : memref<80xi32>
    %8 = arith.addi %4, %c-8 : index
    %9 = memref.load %arg2[%8] : memref<80xi32>
    %10 = arith.xori %7, %9 : i32
    %11 = arith.addi %4, %c-14 : index
    %12 = memref.load %arg2[%11] : memref<80xi32>
    %13 = arith.xori %10, %12 : i32
    %14 = arith.addi %4, %c-16 : index
    %15 = memref.load %arg2[%14] : memref<80xi32>
    %16 = arith.xori %13, %15 : i32
    memref.store %16, %arg2[%4] : memref<80xi32>
    %17 = arith.addi %4, %c1 : index
    cf.br ^bb3(%17 : index)
  ^bb5:  // pred: ^bb3
    %18 = memref.get_global @A : memref<1xi32>
    %19 = memref.load %arg0[%c0] : memref<5xi32>
    memref.store %19, %18[%c0] : memref<1xi32>
    %20 = memref.get_global @B : memref<1xi32>
    %21 = memref.load %arg0[%c1] : memref<5xi32>
    memref.store %21, %20[%c0] : memref<1xi32>
    %22 = memref.get_global @C : memref<1xi32>
    %23 = memref.load %arg0[%c2] : memref<5xi32>
    memref.store %23, %22[%c0] : memref<1xi32>
    %24 = memref.get_global @D : memref<1xi32>
    %25 = memref.load %arg0[%c3] : memref<5xi32>
    memref.store %25, %24[%c0] : memref<1xi32>
    %26 = memref.get_global @E : memref<1xi32>
    %27 = memref.load %arg0[%c4] : memref<5xi32>
    memref.store %27, %26[%c0] : memref<1xi32>
    %28 = memref.get_global @temp : memref<1xi32>
    cf.br ^bb6(%c0 : index)
  ^bb6(%29: index):  // 2 preds: ^bb5, ^bb7
    %30 = arith.cmpi slt, %29, %c20 : index
    cf.cond_br %30, ^bb7, ^bb8(%c20 : index)
  ^bb7:  // pred: ^bb6
    %31 = memref.load %18[%c0] : memref<1xi32>
    %32 = arith.shli %31, %c5_i32 : i32
    %33 = arith.shrsi %31, %c27_i32 : i32
    %34 = arith.ori %32, %33 : i32
    %35 = memref.load %20[%c0] : memref<1xi32>
    %36 = memref.load %22[%c0] : memref<1xi32>
    %37 = arith.andi %35, %36 : i32
    %38 = arith.xori %35, %c-1_i32 : i32
    %39 = memref.load %24[%c0] : memref<1xi32>
    %40 = arith.andi %38, %39 : i32
    %41 = arith.ori %37, %40 : i32
    %42 = arith.addi %34, %41 : i32
    %43 = memref.load %26[%c0] : memref<1xi32>
    %44 = arith.addi %42, %43 : i32
    %45 = memref.load %arg2[%29] : memref<80xi32>
    %46 = arith.addi %44, %45 : i32
    %47 = arith.extsi %46 : i32 to i64
    %48 = arith.addi %47, %c1518500249_i64 : i64
    %49 = arith.trunci %48 : i64 to i32
    memref.store %49, %28[%c0] : memref<1xi32>
    %50 = memref.load %24[%c0] : memref<1xi32>
    memref.store %50, %26[%c0] : memref<1xi32>
    %51 = memref.load %22[%c0] : memref<1xi32>
    memref.store %51, %24[%c0] : memref<1xi32>
    %52 = memref.load %20[%c0] : memref<1xi32>
    %53 = arith.shli %52, %c30_i32 : i32
    %54 = arith.shrsi %52, %c2_i32 : i32
    %55 = arith.ori %53, %54 : i32
    memref.store %55, %22[%c0] : memref<1xi32>
    %56 = memref.load %18[%c0] : memref<1xi32>
    memref.store %56, %20[%c0] : memref<1xi32>
    %57 = memref.load %28[%c0] : memref<1xi32>
    memref.store %57, %18[%c0] : memref<1xi32>
    %58 = arith.addi %29, %c1 : index
    cf.br ^bb6(%58 : index)
  ^bb8(%59: index):  // 2 preds: ^bb6, ^bb9
    %60 = arith.cmpi slt, %59, %c40 : index
    cf.cond_br %60, ^bb9, ^bb10(%c40 : index)
  ^bb9:  // pred: ^bb8
    %61 = memref.load %18[%c0] : memref<1xi32>
    %62 = arith.shli %61, %c5_i32 : i32
    %63 = arith.shrsi %61, %c27_i32 : i32
    %64 = arith.ori %62, %63 : i32
    %65 = memref.load %20[%c0] : memref<1xi32>
    %66 = memref.load %22[%c0] : memref<1xi32>
    %67 = arith.xori %65, %66 : i32
    %68 = memref.load %24[%c0] : memref<1xi32>
    %69 = arith.xori %67, %68 : i32
    %70 = arith.addi %64, %69 : i32
    %71 = memref.load %26[%c0] : memref<1xi32>
    %72 = arith.addi %70, %71 : i32
    %73 = memref.load %arg2[%59] : memref<80xi32>
    %74 = arith.addi %72, %73 : i32
    %75 = arith.extsi %74 : i32 to i64
    %76 = arith.addi %75, %c1859775393_i64 : i64
    %77 = arith.trunci %76 : i64 to i32
    memref.store %77, %28[%c0] : memref<1xi32>
    %78 = memref.load %24[%c0] : memref<1xi32>
    memref.store %78, %26[%c0] : memref<1xi32>
    %79 = memref.load %22[%c0] : memref<1xi32>
    memref.store %79, %24[%c0] : memref<1xi32>
    %80 = memref.load %20[%c0] : memref<1xi32>
    %81 = arith.shli %80, %c30_i32 : i32
    %82 = arith.shrsi %80, %c2_i32 : i32
    %83 = arith.ori %81, %82 : i32
    memref.store %83, %22[%c0] : memref<1xi32>
    %84 = memref.load %18[%c0] : memref<1xi32>
    memref.store %84, %20[%c0] : memref<1xi32>
    %85 = memref.load %28[%c0] : memref<1xi32>
    memref.store %85, %18[%c0] : memref<1xi32>
    %86 = arith.addi %59, %c1 : index
    cf.br ^bb8(%86 : index)
  ^bb10(%87: index):  // 2 preds: ^bb8, ^bb11
    %88 = arith.cmpi slt, %87, %c60 : index
    cf.cond_br %88, ^bb11, ^bb12(%c60 : index)
  ^bb11:  // pred: ^bb10
    %89 = memref.load %18[%c0] : memref<1xi32>
    %90 = arith.shli %89, %c5_i32 : i32
    %91 = arith.shrsi %89, %c27_i32 : i32
    %92 = arith.ori %90, %91 : i32
    %93 = memref.load %20[%c0] : memref<1xi32>
    %94 = memref.load %22[%c0] : memref<1xi32>
    %95 = arith.andi %93, %94 : i32
    %96 = memref.load %24[%c0] : memref<1xi32>
    %97 = arith.andi %93, %96 : i32
    %98 = arith.ori %95, %97 : i32
    %99 = arith.andi %94, %96 : i32
    %100 = arith.ori %98, %99 : i32
    %101 = arith.addi %92, %100 : i32
    %102 = memref.load %26[%c0] : memref<1xi32>
    %103 = arith.addi %101, %102 : i32
    %104 = memref.load %arg2[%87] : memref<80xi32>
    %105 = arith.addi %103, %104 : i32
    %106 = arith.extsi %105 : i32 to i64
    %107 = arith.addi %106, %c1327217884_i64 : i64
    %108 = arith.trunci %107 : i64 to i32
    memref.store %108, %28[%c0] : memref<1xi32>
    %109 = memref.load %24[%c0] : memref<1xi32>
    memref.store %109, %26[%c0] : memref<1xi32>
    %110 = memref.load %22[%c0] : memref<1xi32>
    memref.store %110, %24[%c0] : memref<1xi32>
    %111 = memref.load %20[%c0] : memref<1xi32>
    %112 = arith.shli %111, %c30_i32 : i32
    %113 = arith.shrsi %111, %c2_i32 : i32
    %114 = arith.ori %112, %113 : i32
    memref.store %114, %22[%c0] : memref<1xi32>
    %115 = memref.load %18[%c0] : memref<1xi32>
    memref.store %115, %20[%c0] : memref<1xi32>
    %116 = memref.load %28[%c0] : memref<1xi32>
    memref.store %116, %18[%c0] : memref<1xi32>
    %117 = arith.addi %87, %c1 : index
    cf.br ^bb10(%117 : index)
  ^bb12(%118: index):  // 2 preds: ^bb10, ^bb13
    %119 = arith.cmpi slt, %118, %c80 : index
    cf.cond_br %119, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %120 = memref.load %18[%c0] : memref<1xi32>
    %121 = arith.shli %120, %c5_i32 : i32
    %122 = arith.shrsi %120, %c27_i32 : i32
    %123 = arith.ori %121, %122 : i32
    %124 = memref.load %20[%c0] : memref<1xi32>
    %125 = memref.load %22[%c0] : memref<1xi32>
    %126 = arith.xori %124, %125 : i32
    %127 = memref.load %24[%c0] : memref<1xi32>
    %128 = arith.xori %126, %127 : i32
    %129 = arith.addi %123, %128 : i32
    %130 = memref.load %26[%c0] : memref<1xi32>
    %131 = arith.addi %129, %130 : i32
    %132 = memref.load %arg2[%118] : memref<80xi32>
    %133 = arith.addi %131, %132 : i32
    %134 = arith.extsi %133 : i32 to i64
    %135 = arith.addi %134, %c1247986134_i64 : i64
    %136 = arith.trunci %135 : i64 to i32
    memref.store %136, %28[%c0] : memref<1xi32>
    %137 = memref.load %24[%c0] : memref<1xi32>
    memref.store %137, %26[%c0] : memref<1xi32>
    %138 = memref.load %22[%c0] : memref<1xi32>
    memref.store %138, %24[%c0] : memref<1xi32>
    %139 = memref.load %20[%c0] : memref<1xi32>
    %140 = arith.shli %139, %c30_i32 : i32
    %141 = arith.shrsi %139, %c2_i32 : i32
    %142 = arith.ori %140, %141 : i32
    memref.store %142, %22[%c0] : memref<1xi32>
    %143 = memref.load %18[%c0] : memref<1xi32>
    memref.store %143, %20[%c0] : memref<1xi32>
    %144 = memref.load %28[%c0] : memref<1xi32>
    memref.store %144, %18[%c0] : memref<1xi32>
    %145 = arith.addi %118, %c1 : index
    cf.br ^bb12(%145 : index)
  ^bb14:  // pred: ^bb12
    %146 = memref.load %18[%c0] : memref<1xi32>
    %147 = memref.load %arg0[%c0] : memref<5xi32>
    %148 = arith.addi %147, %146 : i32
    memref.store %148, %arg0[%c0] : memref<5xi32>
    %149 = memref.load %20[%c0] : memref<1xi32>
    %150 = memref.load %arg0[%c1] : memref<5xi32>
    %151 = arith.addi %150, %149 : i32
    memref.store %151, %arg0[%c1] : memref<5xi32>
    %152 = memref.load %22[%c0] : memref<1xi32>
    %153 = memref.load %arg0[%c2] : memref<5xi32>
    %154 = arith.addi %153, %152 : i32
    memref.store %154, %arg0[%c2] : memref<5xi32>
    %155 = memref.load %24[%c0] : memref<1xi32>
    %156 = memref.load %arg0[%c3] : memref<5xi32>
    %157 = arith.addi %156, %155 : i32
    memref.store %157, %arg0[%c3] : memref<5xi32>
    %158 = memref.load %26[%c0] : memref<1xi32>
    %159 = memref.load %arg0[%c4] : memref<5xi32>
    %160 = arith.addi %159, %158 : i32
    memref.store %160, %arg0[%c4] : memref<5xi32>
    return
  }
}

