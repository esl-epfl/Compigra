module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @temp : memref<1xi32> = uninitialized
  memref.global @E : memref<1xi32> = uninitialized
  memref.global @D : memref<1xi32> = uninitialized
  memref.global @C : memref<1xi32> = uninitialized
  memref.global @B : memref<1xi32> = uninitialized
  memref.global @A : memref<1xi32> = uninitialized
  func.func @sha_transform(%arg0: memref<5xi32>, %arg1: memref<16xi32>, %arg2: memref<80xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %c12_i32 = arith.constant 12 : i32
    %c12_i32_1 = arith.constant 12 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8_i32_2 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4_i32_3 = arith.constant 4 : i32
    %c16_i32_4 = arith.constant 16 : i32
    %c12_i32_5 = arith.constant 12 : i32
    %c8_i32_6 = arith.constant 8 : i32
    %c4_i32_7 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c5_i32 = arith.constant 5 : i32
    %c27_i32 = arith.constant 27 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1518500249_i64 = arith.constant 1518500249 : i64
    %c30_i32 = arith.constant 30 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1859775393_i64 = arith.constant 1859775393 : i64
    %c1327217884_i64 = arith.constant 1327217884 : i64
    %c1247986134_i64 = arith.constant 1247986134 : i64
    %c80 = arith.constant 80 : index
    %c-3 = arith.constant -3 : index
    %c-8 = arith.constant -8 : index
    %c-14 = arith.constant -14 : index
    %c-16 = arith.constant -16 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c20 = arith.constant 20 : index
    %c40 = arith.constant 40 : index
    %c60 = arith.constant 60 : index
    %c4_i32_8 = arith.constant 4 : i32
    %c236_i32 = arith.constant {BaseAddr = "arg2"} 236 : i32
    %c172_i32 = arith.constant {BaseAddr = "arg1"} 172 : i32
    %c152_i32 = arith.constant {BaseAddr = "arg0"} 152 : i32
    %c148_i32 = arith.constant {BaseAddr = "global5"} 148 : i32
    %c144_i32 = arith.constant {BaseAddr = "global4"} 144 : i32
    %c140_i32 = arith.constant {BaseAddr = "global3"} 140 : i32
    %c136_i32 = arith.constant {BaseAddr = "global2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "global1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "global0"} 128 : i32
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.index_cast %c16 : index to i32
    cgra.cond_br<ge> [%1 : i32, %2 : i32], ^bb3(%c16 : index), ^bb2
  ^bb2:  // pred: ^bb1
    %3 = arith.index_cast %0 : index to i32
    %4 = arith.muli %3, %c4_i32_8 : i32
    %5 = arith.addi %c172_i32, %4 : i32
    %6 = cgra.lwi %5 : i32->i32
    %7 = arith.index_cast %0 : index to i32
    %8 = arith.muli %7, %c4_i32_8 : i32
    %9 = arith.addi %c236_i32, %8 : i32
    cgra.swi %6, %9 : i32, i32
    %10 = arith.addi %0, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb3(%11: index):  // 2 preds: ^bb1, ^bb4
    %12 = arith.index_cast %11 : index to i32
    %13 = arith.index_cast %c80 : index to i32
    cgra.cond_br<ge> [%12 : i32, %13 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %14 = arith.addi %11, %c-3 : index
    %15 = arith.index_cast %14 : index to i32
    %16 = arith.muli %15, %c4_i32_8 : i32
    %17 = arith.addi %c236_i32, %16 : i32
    %18 = cgra.lwi %17 : i32->i32
    %19 = arith.addi %11, %c-8 : index
    %20 = arith.index_cast %19 : index to i32
    %21 = arith.muli %20, %c4_i32_8 : i32
    %22 = arith.addi %c236_i32, %21 : i32
    %23 = cgra.lwi %22 : i32->i32
    %24 = arith.xori %18, %23 : i32
    %25 = arith.addi %11, %c-14 : index
    %26 = arith.index_cast %25 : index to i32
    %27 = arith.muli %26, %c4_i32_8 : i32
    %28 = arith.addi %c236_i32, %27 : i32
    %29 = cgra.lwi %28 : i32->i32
    %30 = arith.xori %24, %29 : i32
    %31 = arith.addi %11, %c-16 : index
    %32 = arith.index_cast %31 : index to i32
    %33 = arith.muli %32, %c4_i32_8 : i32
    %34 = arith.addi %c236_i32, %33 : i32
    %35 = cgra.lwi %34 : i32->i32
    %36 = arith.xori %30, %35 : i32
    %37 = arith.index_cast %11 : index to i32
    %38 = arith.muli %37, %c4_i32_8 : i32
    %39 = arith.addi %c236_i32, %38 : i32
    cgra.swi %36, %39 : i32, i32
    %40 = arith.addi %11, %c1 : index
    cf.br ^bb3(%40 : index)
  ^bb5:  // pred: ^bb3
    %41 = cgra.lwi %c152_i32 : i32->i32
    cgra.swi %41, %c148_i32 : i32, i32
    %42 = arith.addi %c152_i32, %c4_i32_7 : i32
    %43 = cgra.lwi %42 : i32->i32
    cgra.swi %43, %c144_i32 : i32, i32
    %44 = arith.addi %c152_i32, %c8_i32_6 : i32
    %45 = cgra.lwi %44 : i32->i32
    cgra.swi %45, %c140_i32 : i32, i32
    %46 = arith.addi %c152_i32, %c12_i32_5 : i32
    %47 = cgra.lwi %46 : i32->i32
    cgra.swi %47, %c136_i32 : i32, i32
    %48 = arith.addi %c152_i32, %c16_i32_4 : i32
    %49 = cgra.lwi %48 : i32->i32
    cgra.swi %49, %c132_i32 : i32, i32
    cf.br ^bb6(%c0 : index)
  ^bb6(%50: index):  // 2 preds: ^bb5, ^bb7
    %51 = arith.index_cast %50 : index to i32
    %52 = arith.index_cast %c20 : index to i32
    cgra.cond_br<ge> [%51 : i32, %52 : i32], ^bb8(%c20 : index), ^bb7
  ^bb7:  // pred: ^bb6
    %53 = cgra.lwi %c148_i32 : i32->i32
    %54 = arith.shli %53, %c5_i32 : i32
    %55 = arith.shrsi %53, %c27_i32 : i32
    %56 = arith.ori %54, %55 : i32
    %57 = cgra.lwi %c144_i32 : i32->i32
    %58 = cgra.lwi %c140_i32 : i32->i32
    %59 = arith.andi %57, %58 : i32
    %60 = arith.xori %57, %c-1_i32 : i32
    %61 = cgra.lwi %c136_i32 : i32->i32
    %62 = arith.andi %60, %61 : i32
    %63 = arith.ori %59, %62 : i32
    %64 = arith.addi %56, %63 : i32
    %65 = cgra.lwi %c132_i32 : i32->i32
    %66 = arith.addi %64, %65 : i32
    %67 = arith.index_cast %50 : index to i32
    %68 = arith.muli %67, %c4_i32_8 : i32
    %69 = arith.addi %c236_i32, %68 : i32
    %70 = cgra.lwi %69 : i32->i32
    %71 = arith.addi %66, %70 : i32
    %72 = arith.extsi %71 : i32 to i64
    %73 = arith.addi %72, %c1518500249_i64 : i64
    %74 = arith.trunci %73 : i64 to i32
    cgra.swi %74, %c128_i32 : i32, i32
    %75 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %75, %c132_i32 : i32, i32
    %76 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %76, %c136_i32 : i32, i32
    %77 = cgra.lwi %c144_i32 : i32->i32
    %78 = arith.shli %77, %c30_i32 : i32
    %79 = arith.shrsi %77, %c2_i32 : i32
    %80 = arith.ori %78, %79 : i32
    cgra.swi %80, %c140_i32 : i32, i32
    %81 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %81, %c144_i32 : i32, i32
    %82 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %82, %c148_i32 : i32, i32
    %83 = arith.addi %50, %c1 : index
    cf.br ^bb6(%83 : index)
  ^bb8(%84: index):  // 2 preds: ^bb6, ^bb9
    %85 = arith.index_cast %84 : index to i32
    %86 = arith.index_cast %c40 : index to i32
    cgra.cond_br<ge> [%85 : i32, %86 : i32], ^bb10(%c40 : index), ^bb9
  ^bb9:  // pred: ^bb8
    %87 = cgra.lwi %c148_i32 : i32->i32
    %88 = arith.shli %87, %c5_i32 : i32
    %89 = arith.shrsi %87, %c27_i32 : i32
    %90 = arith.ori %88, %89 : i32
    %91 = cgra.lwi %c144_i32 : i32->i32
    %92 = cgra.lwi %c140_i32 : i32->i32
    %93 = arith.xori %91, %92 : i32
    %94 = cgra.lwi %c136_i32 : i32->i32
    %95 = arith.xori %93, %94 : i32
    %96 = arith.addi %90, %95 : i32
    %97 = cgra.lwi %c132_i32 : i32->i32
    %98 = arith.addi %96, %97 : i32
    %99 = arith.index_cast %84 : index to i32
    %100 = arith.muli %99, %c4_i32_8 : i32
    %101 = arith.addi %c236_i32, %100 : i32
    %102 = cgra.lwi %101 : i32->i32
    %103 = arith.addi %98, %102 : i32
    %104 = arith.extsi %103 : i32 to i64
    %105 = arith.addi %104, %c1859775393_i64 : i64
    %106 = arith.trunci %105 : i64 to i32
    cgra.swi %106, %c128_i32 : i32, i32
    %107 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %107, %c132_i32 : i32, i32
    %108 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %108, %c136_i32 : i32, i32
    %109 = cgra.lwi %c144_i32 : i32->i32
    %110 = arith.shli %109, %c30_i32 : i32
    %111 = arith.shrsi %109, %c2_i32 : i32
    %112 = arith.ori %110, %111 : i32
    cgra.swi %112, %c140_i32 : i32, i32
    %113 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %113, %c144_i32 : i32, i32
    %114 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %114, %c148_i32 : i32, i32
    %115 = arith.addi %84, %c1 : index
    cf.br ^bb8(%115 : index)
  ^bb10(%116: index):  // 2 preds: ^bb8, ^bb11
    %117 = arith.index_cast %116 : index to i32
    %118 = arith.index_cast %c60 : index to i32
    cgra.cond_br<ge> [%117 : i32, %118 : i32], ^bb12(%c60 : index), ^bb11
  ^bb11:  // pred: ^bb10
    %119 = cgra.lwi %c148_i32 : i32->i32
    %120 = arith.shli %119, %c5_i32 : i32
    %121 = arith.shrsi %119, %c27_i32 : i32
    %122 = arith.ori %120, %121 : i32
    %123 = cgra.lwi %c144_i32 : i32->i32
    %124 = cgra.lwi %c140_i32 : i32->i32
    %125 = arith.andi %123, %124 : i32
    %126 = cgra.lwi %c136_i32 : i32->i32
    %127 = arith.andi %123, %126 : i32
    %128 = arith.ori %125, %127 : i32
    %129 = arith.andi %124, %126 : i32
    %130 = arith.ori %128, %129 : i32
    %131 = arith.addi %122, %130 : i32
    %132 = cgra.lwi %c132_i32 : i32->i32
    %133 = arith.addi %131, %132 : i32
    %134 = arith.index_cast %116 : index to i32
    %135 = arith.muli %134, %c4_i32_8 : i32
    %136 = arith.addi %c236_i32, %135 : i32
    %137 = cgra.lwi %136 : i32->i32
    %138 = arith.addi %133, %137 : i32
    %139 = arith.extsi %138 : i32 to i64
    %140 = arith.addi %139, %c1327217884_i64 : i64
    %141 = arith.trunci %140 : i64 to i32
    cgra.swi %141, %c128_i32 : i32, i32
    %142 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %142, %c132_i32 : i32, i32
    %143 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %143, %c136_i32 : i32, i32
    %144 = cgra.lwi %c144_i32 : i32->i32
    %145 = arith.shli %144, %c30_i32 : i32
    %146 = arith.shrsi %144, %c2_i32 : i32
    %147 = arith.ori %145, %146 : i32
    cgra.swi %147, %c140_i32 : i32, i32
    %148 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %148, %c144_i32 : i32, i32
    %149 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %149, %c148_i32 : i32, i32
    %150 = arith.addi %116, %c1 : index
    cf.br ^bb10(%150 : index)
  ^bb12(%151: index):  // 2 preds: ^bb10, ^bb13
    %152 = arith.index_cast %151 : index to i32
    %153 = arith.index_cast %c80 : index to i32
    cgra.cond_br<ge> [%152 : i32, %153 : i32], ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    %154 = cgra.lwi %c148_i32 : i32->i32
    %155 = arith.shli %154, %c5_i32 : i32
    %156 = arith.shrsi %154, %c27_i32 : i32
    %157 = arith.ori %155, %156 : i32
    %158 = cgra.lwi %c144_i32 : i32->i32
    %159 = cgra.lwi %c140_i32 : i32->i32
    %160 = arith.xori %158, %159 : i32
    %161 = cgra.lwi %c136_i32 : i32->i32
    %162 = arith.xori %160, %161 : i32
    %163 = arith.addi %157, %162 : i32
    %164 = cgra.lwi %c132_i32 : i32->i32
    %165 = arith.addi %163, %164 : i32
    %166 = arith.index_cast %151 : index to i32
    %167 = arith.muli %166, %c4_i32_8 : i32
    %168 = arith.addi %c236_i32, %167 : i32
    %169 = cgra.lwi %168 : i32->i32
    %170 = arith.addi %165, %169 : i32
    %171 = arith.extsi %170 : i32 to i64
    %172 = arith.addi %171, %c1247986134_i64 : i64
    %173 = arith.trunci %172 : i64 to i32
    cgra.swi %173, %c128_i32 : i32, i32
    %174 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %174, %c132_i32 : i32, i32
    %175 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %175, %c136_i32 : i32, i32
    %176 = cgra.lwi %c144_i32 : i32->i32
    %177 = arith.shli %176, %c30_i32 : i32
    %178 = arith.shrsi %176, %c2_i32 : i32
    %179 = arith.ori %177, %178 : i32
    cgra.swi %179, %c140_i32 : i32, i32
    %180 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %180, %c144_i32 : i32, i32
    %181 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %181, %c148_i32 : i32, i32
    %182 = arith.addi %151, %c1 : index
    cf.br ^bb12(%182 : index)
  ^bb14:  // pred: ^bb12
    %183 = cgra.lwi %c148_i32 : i32->i32
    %184 = cgra.lwi %c152_i32 : i32->i32
    %185 = arith.addi %184, %183 : i32
    cgra.swi %185, %c152_i32 : i32, i32
    %186 = cgra.lwi %c144_i32 : i32->i32
    %187 = arith.addi %c152_i32, %c4_i32_3 : i32
    %188 = cgra.lwi %187 : i32->i32
    %189 = arith.addi %188, %186 : i32
    %190 = arith.addi %c152_i32, %c4_i32 : i32
    cgra.swi %189, %190 : i32, i32
    %191 = cgra.lwi %c140_i32 : i32->i32
    %192 = arith.addi %c152_i32, %c8_i32_2 : i32
    %193 = cgra.lwi %192 : i32->i32
    %194 = arith.addi %193, %191 : i32
    %195 = arith.addi %c152_i32, %c8_i32 : i32
    cgra.swi %194, %195 : i32, i32
    %196 = cgra.lwi %c136_i32 : i32->i32
    %197 = arith.addi %c152_i32, %c12_i32_1 : i32
    %198 = cgra.lwi %197 : i32->i32
    %199 = arith.addi %198, %196 : i32
    %200 = arith.addi %c152_i32, %c12_i32 : i32
    cgra.swi %199, %200 : i32, i32
    %201 = cgra.lwi %c132_i32 : i32->i32
    %202 = arith.addi %c152_i32, %c16_i32_0 : i32
    %203 = cgra.lwi %202 : i32->i32
    %204 = arith.addi %203, %201 : i32
    %205 = arith.addi %c152_i32, %c16_i32 : i32
    cgra.swi %204, %205 : i32, i32
    return
  }
}

