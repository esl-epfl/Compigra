#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.mlir.global private unnamed_addr constant @__const.main.input(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<10xf64>) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.array<10 x f64>
  llvm.mlir.global private unnamed_addr constant @".str.1"("output[%d] = %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str("Filtered output:\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func local_unnamed_addr @FIR(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %3 = llvm.load %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %4 = llvm.icmp "sgt" %2, %0 : i32
    llvm.cond_br %4, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %5 = llvm.getelementptr inbounds %arg3[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.br ^bb3(%0 : i32)
  ^bb2:  // 2 preds: ^bb0, ^bb5
    llvm.return
  ^bb3(%6: i32):  // 2 preds: ^bb1, ^bb5
    %7 = llvm.icmp "slt" %3, %6 : i32
    llvm.cond_br %7, ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %8 = llvm.getelementptr inbounds %arg4[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %9 = llvm.load %8 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
    %10 = llvm.sub %3, %6  : i32
    %11 = llvm.getelementptr inbounds %arg2[%10] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %12 = llvm.load %11 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
    %13 = llvm.load %5 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
    %14 = llvm.intr.fmuladd(%9, %12, %13)  : (f32, f32, f32) -> f32
    llvm.store %14, %5 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : f32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %15 = llvm.add %6, %1  : i32
    %16 = llvm.icmp "eq" %15, %2 : i32
    llvm.cond_br %16, ^bb2, ^bb3(%15 : i32) {loop_annotation = #loop_annotation}
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {passthrough = ["nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(40 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1.000000e-01 : f64) : f64
    %4 = llvm.mlir.constant(1.500000e-01 : f64) : f64
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %7 = llvm.mlir.constant(3 : i32) : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(80 : i64) : i64
    %10 = llvm.mlir.constant(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<10xf64>) : !llvm.array<10 x f64>
    %11 = llvm.mlir.addressof @__const.main.input : !llvm.ptr
    %12 = llvm.mlir.constant(80 : i32) : i32
    %13 = llvm.mlir.constant(false) : i1
    %14 = llvm.mlir.addressof @fir_filter : !llvm.ptr
    %15 = llvm.mlir.constant(5 : i32) : i32
    %16 = llvm.mlir.constant(10 : i32) : i32
    %17 = llvm.mlir.constant("Filtered output:\00") : !llvm.array<17 x i8>
    %18 = llvm.mlir.addressof @str : !llvm.ptr
    %19 = llvm.mlir.constant("output[%d] = %f\0A\00") : !llvm.array<17 x i8>
    %20 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %21 = llvm.alloca %0 x !llvm.array<5 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.alloca %0 x !llvm.array<10 x f64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %23 = llvm.bitcast %21 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 40, %23 : !llvm.ptr
    %24 = llvm.getelementptr inbounds %21[%2, %2] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<5 x f64>
    llvm.store %3, %24 {alignment = 8 : i64} : f64, !llvm.ptr
    %25 = llvm.getelementptr inbounds %21[%2, %0] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<5 x f64>
    llvm.store %4, %25 {alignment = 8 : i64} : f64, !llvm.ptr
    %26 = llvm.getelementptr inbounds %21[%2, %5] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<5 x f64>
    llvm.store %6, %26 {alignment = 8 : i64} : f64, !llvm.ptr
    %27 = llvm.getelementptr inbounds %21[%2, %7] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<5 x f64>
    llvm.store %4, %27 {alignment = 8 : i64} : f64, !llvm.ptr
    %28 = llvm.getelementptr inbounds %21[%2, %8] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<5 x f64>
    llvm.store %3, %28 {alignment = 8 : i64} : f64, !llvm.ptr
    %29 = llvm.bitcast %22 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 80, %29 : !llvm.ptr
    "llvm.intr.memcpy"(%29, %11, %12) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %30 = llvm.call @malloc(%12) : (i32) -> !llvm.ptr
    %31 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    %32 = llvm.getelementptr inbounds %22[%2, %2] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<10 x f64>
    %33 = llvm.call %14(%32, %31, %24, %15, %16) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> i32
    %34 = llvm.call @puts(%18) : (!llvm.ptr) -> i32
    llvm.br ^bb2(%2 : i32)
  ^bb1:  // pred: ^bb2
    llvm.call @free(%30) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 80, %29 : !llvm.ptr
    llvm.intr.lifetime.end 40, %23 : !llvm.ptr
    llvm.return %2 : i32
  ^bb2(%35: i32):  // 2 preds: ^bb0, ^bb2
    %36 = llvm.getelementptr inbounds %31[%35] : (!llvm.ptr, i32) -> !llvm.ptr, f64
    %37 = llvm.load %36 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> f64
    %38 = llvm.call @printf(%20, %35, %37) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64) -> i32
    %39 = llvm.add %35, %0  : i32
    %40 = llvm.icmp "eq" %39, %16 : i32
    llvm.cond_br %40, ^bb1, ^bb2(%39 : i32) {loop_annotation = #loop_annotation}
  }
  llvm.func local_unnamed_addr @malloc(i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @fir_filter(...) -> i32 attributes {passthrough = [["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}
}
