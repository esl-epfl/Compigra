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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @temp : memref<1xi32> = uninitialized
  memref.global @E : memref<1xi32> = uninitialized
  memref.global @D : memref<1xi32> = uninitialized
  memref.global @C : memref<1xi32> = uninitialized
  memref.global @B : memref<1xi32> = uninitialized
  memref.global @A : memref<1xi32> = uninitialized
  func.func @sha_transform(%arg0: memref<5xi32>, %arg1: memref<16xi32>, %arg2: memref<80xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c16_i32_1 = arith.constant 16 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c27_i32 = arith.constant 27 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c2457_i32 = arith.constant 2457 : i32
    %c2087_i32 = arith.constant 2087 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %0 = arith.addi %c2087_i32, %c0_i32_6 : i32
    %c12_i32 = arith.constant 12 : i32
    %1 = arith.shli %0, %c12_i32 : i32
    %2 = arith.addi %c2457_i32, %1 : i32
    %c90_i32 = arith.constant 90 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %3 = arith.addi %c90_i32, %c0_i32_7 : i32
    %c24_i32 = arith.constant 24 : i32
    %4 = arith.shli %3, %c24_i32 : i32
    %5 = arith.addi %2, %4 {constant = 1518500249 : i32} : i32
    %c30_i32 = arith.constant 30 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2977_i32 = arith.constant 2977 : i32
    %c3486_i32 = arith.constant 3486 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %6 = arith.addi %c3486_i32, %c0_i32_8 : i32
    %c12_i32_9 = arith.constant 12 : i32
    %7 = arith.shli %6, %c12_i32_9 : i32
    %8 = arith.addi %c2977_i32, %7 : i32
    %c110_i32 = arith.constant 110 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %9 = arith.addi %c110_i32, %c0_i32_10 : i32
    %c24_i32_11 = arith.constant 24 : i32
    %10 = arith.shli %9, %c24_i32_11 : i32
    %11 = arith.addi %8, %10 {constant = 1859775393 : i32} : i32
    %c3292_i32 = arith.constant 3292 : i32
    %c443_i32 = arith.constant 443 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %12 = arith.addi %c443_i32, %c0_i32_12 : i32
    %c12_i32_13 = arith.constant 12 : i32
    %13 = arith.shli %12, %c12_i32_13 : i32
    %14 = arith.addi %c3292_i32, %13 : i32
    %c79_i32 = arith.constant 79 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %15 = arith.addi %c79_i32, %c0_i32_14 : i32
    %c24_i32_15 = arith.constant 24 : i32
    %16 = arith.shli %15, %c24_i32_15 : i32
    %17 = arith.addi %14, %16 {constant = 1327217884 : i32} : i32
    %c470_i32 = arith.constant 470 : i32
    %c1580_i32 = arith.constant 1580 : i32
    %c0_i32_16 = arith.constant 0 : i32
    %18 = arith.addi %c1580_i32, %c0_i32_16 : i32
    %c12_i32_17 = arith.constant 12 : i32
    %19 = arith.shli %18, %c12_i32_17 : i32
    %20 = arith.addi %c470_i32, %19 : i32
    %c74_i32 = arith.constant 74 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %21 = arith.addi %c74_i32, %c0_i32_18 : i32
    %c24_i32_19 = arith.constant 24 : i32
    %22 = arith.shli %21, %c24_i32_19 : i32
    %23 = arith.addi %20, %22 {constant = 1247986134 : i32} : i32
    %c0_i32_20 = arith.constant 0 : i32
    %c80_i32 = arith.constant 80 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %c80_i32_22 = arith.constant 80 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-8_i32 = arith.constant -8 : i32
    %c-14_i32 = arith.constant -14 : i32
    %c-16_i32 = arith.constant -16 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %c0_i32_24 = arith.constant 0 : i32
    %c20_i32_25 = arith.constant 20 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %c40_i32 = arith.constant 40 : i32
    %c0_i32_27 = arith.constant 0 : i32
    %c40_i32_28 = arith.constant 40 : i32
    %c0_i32_29 = arith.constant 0 : i32
    %c60_i32 = arith.constant 60 : i32
    %c0_i32_30 = arith.constant 0 : i32
    %c60_i32_31 = arith.constant 60 : i32
    %c4_i32 = arith.constant 4 : i32
    %c236_i32 = arith.constant {BaseAddr = "arg2"} 236 : i32
    %c172_i32 = arith.constant {BaseAddr = "arg1"} 172 : i32
    %c152_i32 = arith.constant {BaseAddr = "arg0"} 152 : i32
    %c148_i32 = arith.constant {BaseAddr = "global5"} 148 : i32
    %c144_i32 = arith.constant {BaseAddr = "global4"} 144 : i32
    %c140_i32 = arith.constant {BaseAddr = "global3"} 140 : i32
    %c136_i32 = arith.constant {BaseAddr = "global2"} 136 : i32
    %c132_i32 = arith.constant {BaseAddr = "global1"} 132 : i32
    %c128_i32 = arith.constant {BaseAddr = "global0"} 128 : i32
    %24 = arith.addi %c0_i32_4, %c0_i32_5 {constant = 0 : i32} : i32
    cf.br ^bb1(%24 : i32)
  ^bb1(%25: i32):  // 2 preds: ^bb0, ^bb2
    %26 = arith.addi %c0_i32, %c16_i32 {constant = 16 : i32} : i32
    %27 = arith.addi %c0_i32_0, %c16_i32_1 {constant = 16 : i32} : i32
    cgra.cond_br<ge> [%25 : i32, %26 : i32], ^bb3(%27 : i32), ^bb2
  ^bb2:  // pred: ^bb1
    %28 = arith.muli %25, %c4_i32 : i32
    %29 = arith.addi %28, %c172_i32 : i32
    %30 = cgra.lwi %29 : i32->i32
    %31 = arith.muli %25, %c4_i32 : i32
    %32 = arith.addi %31, %c236_i32 : i32
    cgra.swi %30, %32 : i32, i32
    %33 = arith.addi %25, %c1_i32 : i32
    cf.br ^bb1(%33 : i32)
  ^bb3(%34: i32):  // 2 preds: ^bb1, ^bb4
    %35 = arith.addi %c0_i32_21, %c80_i32_22 {constant = 80 : i32} : i32
    cgra.cond_br<ge> [%34 : i32, %35 : i32], ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %36 = arith.addi %34, %c-3_i32 : i32
    %37 = arith.muli %36, %c4_i32 : i32
    %38 = arith.addi %37, %c236_i32 : i32
    %39 = cgra.lwi %38 : i32->i32
    %40 = arith.addi %34, %c-8_i32 : i32
    %41 = arith.muli %40, %c4_i32 : i32
    %42 = arith.addi %41, %c236_i32 : i32
    %43 = cgra.lwi %42 : i32->i32
    %44 = arith.xori %39, %43 : i32
    %45 = arith.addi %34, %c-14_i32 : i32
    %46 = arith.muli %45, %c4_i32 : i32
    %47 = arith.addi %46, %c236_i32 : i32
    %48 = cgra.lwi %47 : i32->i32
    %49 = arith.xori %44, %48 : i32
    %50 = arith.addi %34, %c-16_i32 : i32
    %51 = arith.muli %50, %c4_i32 : i32
    %52 = arith.addi %51, %c236_i32 : i32
    %53 = cgra.lwi %52 : i32->i32
    %54 = arith.xori %49, %53 : i32
    %55 = arith.muli %34, %c4_i32 : i32
    %56 = arith.addi %55, %c236_i32 : i32
    cgra.swi %54, %56 : i32, i32
    %57 = arith.addi %34, %c1_i32 : i32
    cf.br ^bb3(%57 : i32)
  ^bb5:  // pred: ^bb3
    %58 = cgra.lwi %c152_i32 : i32->i32
    cgra.swi %58, %c148_i32 : i32, i32
    %c156_i32 = arith.constant 156 : i32
    %59 = cgra.lwi %c156_i32 : i32->i32
    cgra.swi %59, %c144_i32 : i32, i32
    %c160_i32 = arith.constant 160 : i32
    %60 = cgra.lwi %c160_i32 : i32->i32
    cgra.swi %60, %c140_i32 : i32, i32
    %c164_i32 = arith.constant 164 : i32
    %61 = cgra.lwi %c164_i32 : i32->i32
    cgra.swi %61, %c136_i32 : i32, i32
    %c168_i32 = arith.constant 168 : i32
    %62 = cgra.lwi %c168_i32 : i32->i32
    cgra.swi %62, %c132_i32 : i32, i32
    %63 = arith.addi %c0_i32_2, %c0_i32_3 {constant = 0 : i32} : i32
    cf.br ^bb6(%63 : i32)
  ^bb6(%64: i32):  // 2 preds: ^bb5, ^bb7
    %65 = arith.addi %c0_i32_23, %c20_i32 {constant = 20 : i32} : i32
    %66 = arith.addi %c0_i32_24, %c20_i32_25 {constant = 20 : i32} : i32
    cgra.cond_br<ge> [%64 : i32, %65 : i32], ^bb8(%66 : i32), ^bb7
  ^bb7:  // pred: ^bb6
    %67 = cgra.lwi %c148_i32 : i32->i32
    %68 = arith.shli %67, %c5_i32 : i32
    %69 = arith.shrsi %67, %c27_i32 : i32
    %70 = arith.ori %68, %69 : i32
    %71 = cgra.lwi %c144_i32 : i32->i32
    %72 = cgra.lwi %c140_i32 : i32->i32
    %73 = arith.andi %71, %72 : i32
    %74 = arith.xori %71, %c-1_i32 : i32
    %75 = cgra.lwi %c136_i32 : i32->i32
    %76 = arith.andi %74, %75 : i32
    %77 = arith.ori %73, %76 : i32
    %78 = arith.addi %70, %77 : i32
    %79 = cgra.lwi %c132_i32 : i32->i32
    %80 = arith.addi %78, %79 : i32
    %81 = arith.muli %64, %c4_i32 : i32
    %82 = arith.addi %81, %c236_i32 : i32
    %83 = cgra.lwi %82 : i32->i32
    %84 = arith.addi %80, %83 : i32
    %85 = arith.addi %84, %5 : i32
    cgra.swi %85, %c128_i32 : i32, i32
    %86 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %86, %c132_i32 : i32, i32
    %87 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %87, %c136_i32 : i32, i32
    %88 = cgra.lwi %c144_i32 : i32->i32
    %89 = arith.shli %88, %c30_i32 : i32
    %90 = arith.shrsi %88, %c2_i32 : i32
    %91 = arith.ori %89, %90 : i32
    cgra.swi %91, %c140_i32 : i32, i32
    %92 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %92, %c144_i32 : i32, i32
    %93 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %93, %c148_i32 : i32, i32
    %94 = arith.addi %64, %c1_i32 : i32
    cf.br ^bb6(%94 : i32)
  ^bb8(%95: i32):  // 2 preds: ^bb6, ^bb9
    %96 = arith.addi %c0_i32_26, %c40_i32 {constant = 40 : i32} : i32
    %97 = arith.addi %c0_i32_27, %c40_i32_28 {constant = 40 : i32} : i32
    cgra.cond_br<ge> [%95 : i32, %96 : i32], ^bb10(%97 : i32), ^bb9
  ^bb9:  // pred: ^bb8
    %98 = cgra.lwi %c148_i32 : i32->i32
    %99 = arith.shli %98, %c5_i32 : i32
    %100 = arith.shrsi %98, %c27_i32 : i32
    %101 = arith.ori %99, %100 : i32
    %102 = cgra.lwi %c144_i32 : i32->i32
    %103 = cgra.lwi %c140_i32 : i32->i32
    %104 = arith.xori %102, %103 : i32
    %105 = cgra.lwi %c136_i32 : i32->i32
    %106 = arith.xori %104, %105 : i32
    %107 = arith.addi %101, %106 : i32
    %108 = cgra.lwi %c132_i32 : i32->i32
    %109 = arith.addi %107, %108 : i32
    %110 = arith.muli %95, %c4_i32 : i32
    %111 = arith.addi %110, %c236_i32 : i32
    %112 = cgra.lwi %111 : i32->i32
    %113 = arith.addi %109, %112 : i32
    %114 = arith.addi %113, %11 : i32
    cgra.swi %114, %c128_i32 : i32, i32
    %115 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %115, %c132_i32 : i32, i32
    %116 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %116, %c136_i32 : i32, i32
    %117 = cgra.lwi %c144_i32 : i32->i32
    %118 = arith.shli %117, %c30_i32 : i32
    %119 = arith.shrsi %117, %c2_i32 : i32
    %120 = arith.ori %118, %119 : i32
    cgra.swi %120, %c140_i32 : i32, i32
    %121 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %121, %c144_i32 : i32, i32
    %122 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %122, %c148_i32 : i32, i32
    %123 = arith.addi %95, %c1_i32 : i32
    cf.br ^bb8(%123 : i32)
  ^bb10(%124: i32):  // 2 preds: ^bb8, ^bb11
    %125 = arith.addi %c0_i32_29, %c60_i32 {constant = 60 : i32} : i32
    %126 = arith.addi %c0_i32_30, %c60_i32_31 {constant = 60 : i32} : i32
    cgra.cond_br<ge> [%124 : i32, %125 : i32], ^bb12(%126 : i32), ^bb11
  ^bb11:  // pred: ^bb10
    %127 = cgra.lwi %c148_i32 : i32->i32
    %128 = arith.shli %127, %c5_i32 : i32
    %129 = arith.shrsi %127, %c27_i32 : i32
    %130 = arith.ori %128, %129 : i32
    %131 = cgra.lwi %c144_i32 : i32->i32
    %132 = cgra.lwi %c140_i32 : i32->i32
    %133 = arith.andi %131, %132 : i32
    %134 = cgra.lwi %c136_i32 : i32->i32
    %135 = arith.andi %131, %134 : i32
    %136 = arith.ori %133, %135 : i32
    %137 = arith.andi %132, %134 : i32
    %138 = arith.ori %136, %137 : i32
    %139 = arith.addi %130, %138 : i32
    %140 = cgra.lwi %c132_i32 : i32->i32
    %141 = arith.addi %139, %140 : i32
    %142 = arith.muli %124, %c4_i32 : i32
    %143 = arith.addi %142, %c236_i32 : i32
    %144 = cgra.lwi %143 : i32->i32
    %145 = arith.addi %141, %144 : i32
    %146 = arith.addi %145, %17 : i32
    cgra.swi %146, %c128_i32 : i32, i32
    %147 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %147, %c132_i32 : i32, i32
    %148 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %148, %c136_i32 : i32, i32
    %149 = cgra.lwi %c144_i32 : i32->i32
    %150 = arith.shli %149, %c30_i32 : i32
    %151 = arith.shrsi %149, %c2_i32 : i32
    %152 = arith.ori %150, %151 : i32
    cgra.swi %152, %c140_i32 : i32, i32
    %153 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %153, %c144_i32 : i32, i32
    %154 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %154, %c148_i32 : i32, i32
    %155 = arith.addi %124, %c1_i32 : i32
    cf.br ^bb10(%155 : i32)
  ^bb12(%156: i32):  // 2 preds: ^bb10, ^bb13
    %157 = arith.addi %c0_i32_20, %c80_i32 {constant = 80 : i32} : i32
    cgra.cond_br<ge> [%156 : i32, %157 : i32], ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    %158 = cgra.lwi %c148_i32 : i32->i32
    %159 = arith.shli %158, %c5_i32 : i32
    %160 = arith.shrsi %158, %c27_i32 : i32
    %161 = arith.ori %159, %160 : i32
    %162 = cgra.lwi %c144_i32 : i32->i32
    %163 = cgra.lwi %c140_i32 : i32->i32
    %164 = arith.xori %162, %163 : i32
    %165 = cgra.lwi %c136_i32 : i32->i32
    %166 = arith.xori %164, %165 : i32
    %167 = arith.addi %161, %166 : i32
    %168 = cgra.lwi %c132_i32 : i32->i32
    %169 = arith.addi %167, %168 : i32
    %170 = arith.muli %156, %c4_i32 : i32
    %171 = arith.addi %170, %c236_i32 : i32
    %172 = cgra.lwi %171 : i32->i32
    %173 = arith.addi %169, %172 : i32
    %174 = arith.addi %173, %23 : i32
    cgra.swi %174, %c128_i32 : i32, i32
    %175 = cgra.lwi %c136_i32 : i32->i32
    cgra.swi %175, %c132_i32 : i32, i32
    %176 = cgra.lwi %c140_i32 : i32->i32
    cgra.swi %176, %c136_i32 : i32, i32
    %177 = cgra.lwi %c144_i32 : i32->i32
    %178 = arith.shli %177, %c30_i32 : i32
    %179 = arith.shrsi %177, %c2_i32 : i32
    %180 = arith.ori %178, %179 : i32
    cgra.swi %180, %c140_i32 : i32, i32
    %181 = cgra.lwi %c148_i32 : i32->i32
    cgra.swi %181, %c144_i32 : i32, i32
    %182 = cgra.lwi %c128_i32 : i32->i32
    cgra.swi %182, %c148_i32 : i32, i32
    %183 = arith.addi %156, %c1_i32 : i32
    cf.br ^bb12(%183 : i32)
  ^bb14:  // pred: ^bb12
    %184 = cgra.lwi %c148_i32 : i32->i32
    %185 = cgra.lwi %c152_i32 : i32->i32
    %186 = arith.addi %185, %184 : i32
    cgra.swi %186, %c152_i32 : i32, i32
    %187 = cgra.lwi %c144_i32 : i32->i32
    %c156_i32_32 = arith.constant 156 : i32
    %188 = cgra.lwi %c156_i32_32 : i32->i32
    %189 = arith.addi %188, %187 : i32
    %c156_i32_33 = arith.constant 156 : i32
    cgra.swi %189, %c156_i32_33 : i32, i32
    %190 = cgra.lwi %c140_i32 : i32->i32
    %c160_i32_34 = arith.constant 160 : i32
    %191 = cgra.lwi %c160_i32_34 : i32->i32
    %192 = arith.addi %191, %190 : i32
    %c160_i32_35 = arith.constant 160 : i32
    cgra.swi %192, %c160_i32_35 : i32, i32
    %193 = cgra.lwi %c136_i32 : i32->i32
    %c164_i32_36 = arith.constant 164 : i32
    %194 = cgra.lwi %c164_i32_36 : i32->i32
    %195 = arith.addi %194, %193 : i32
    %c164_i32_37 = arith.constant 164 : i32
    cgra.swi %195, %c164_i32_37 : i32, i32
    %196 = cgra.lwi %c132_i32 : i32->i32
    %c168_i32_38 = arith.constant 168 : i32
    %197 = cgra.lwi %c168_i32_38 : i32->i32
    %198 = arith.addi %197, %196 : i32
    %c168_i32_39 = arith.constant 168 : i32
    cgra.swi %198, %c168_i32_39 : i32, i32
    return
  }
}

