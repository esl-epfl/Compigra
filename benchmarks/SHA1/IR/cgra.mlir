#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @SHA1(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, ...) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = [], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(512 : i32) {stage = "init"} : i32
    %1 = llvm.mlir.constant(512 : i32) {stage = "init"} : i32
    %2 = llvm.mlir.constant(512 : i32) {stage = "init"} : i32
    %3 = llvm.mlir.constant(512 : i32) {stage = "init"} : i32
    %4 = llvm.mlir.constant(512 : i32) {arg0 = "base", stage = "init"} : i32
    %5 = llvm.mlir.constant(16 : i32) {stage = "init"} : i32
    %6 = llvm.mlir.constant(-3 : i32) {stage = "init"} : i32
    %7 = llvm.mlir.constant(-8 : i32) {stage = "init"} : i32
    %8 = llvm.mlir.constant(-14 : i32) {stage = "init"} : i32
    %9 = llvm.mlir.constant(-16 : i32) {stage = "init"} : i32
    %10 = llvm.mlir.constant(1 : i32) {stage = "init"} : i32
    %11 = llvm.mlir.constant(80 : i32) {stage = "init"} : i32
    %12 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %13 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %14 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %15 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %16 = llvm.mlir.constant(4 : i32) {stage = "init"} : i32
    %17 = merge %39, %5 {stage = "loop"} : i32
    %18 = llvm.add %17, %6 {stage = "loop"} : i32
    %19 = llvm.mul %12, %18 {stage = "loop"} : i32
    %20 = llvm.add %3, %19 {stage = "loop"} : i32
    %21 = lwi %20 {stage = "loop"} : i32
    %22 = llvm.add %17, %7 {stage = "loop"} : i32
    %23 = llvm.mul %13, %22 {stage = "loop"} : i32
    %24 = llvm.add %2, %23 {stage = "loop"} : i32
    %25 = lwi %24 {stage = "loop"} : i32
    %26 = llvm.xor %25, %21  {stage = "loop"} : i32
    %27 = llvm.add %17, %8 {stage = "loop"} : i32
    %28 = llvm.mul %14, %27 {stage = "loop"} : i32
    %29 = llvm.add %1, %28 {stage = "loop"} : i32
    %30 = lwi %29 {stage = "loop"} : i32
    %31 = llvm.xor %26, %30  {stage = "loop"} : i32
    %32 = llvm.add %17, %9 {stage = "loop"} : i32
    %33 = llvm.mul %15, %32 {stage = "loop"} : i32
    %34 = llvm.add %0, %33 {stage = "loop"} : i32
    %35 = lwi %34 {stage = "loop"} : i32
    %36 = llvm.xor %31, %35  {stage = "loop"} : i32
    %37 = llvm.mul %16, %17 {stage = "loop"} : i32
    %38 = llvm.add %4, %37 {stage = "loop"} : i32
    swi %36, %38 {stage = "loop"}
    %39 = llvm.add %17, %10 {stage = "loop"} : i32
    %40 = beq [%39, %11] %17 {stage = "loop"} : i32
    llvm.return {stage = "fini"}
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(320 : i64) : i64
    %2 = llvm.mlir.constant(16 : i32) : i32
    %3 = llvm.mlir.constant(-3 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(-8 : i32) : i32
    %6 = llvm.mlir.constant(-14 : i32) : i32
    %7 = llvm.mlir.constant(-16 : i32) : i32
    %8 = llvm.mlir.constant(80 : i32) : i32
    %9 = llvm.alloca %0 x !llvm.array<80 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.bitcast %9 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 320, %10 : !llvm.ptr
    llvm.br ^bb1(%2 : i32)
  ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb1
    %12 = llvm.add %11, %3 : i32
    %13 = llvm.getelementptr inbounds %9[%4, %12] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<80 x i32>
    %14 = llvm.load %13 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %15 = llvm.add %11, %5 : i32
    %16 = llvm.getelementptr inbounds %9[%4, %15] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<80 x i32>
    %17 = llvm.load %16 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %18 = llvm.xor %17, %14  : i32
    %19 = llvm.add %11, %6 : i32
    %20 = llvm.getelementptr inbounds %9[%4, %19] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<80 x i32>
    %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %22 = llvm.xor %18, %21  : i32
    %23 = llvm.add %11, %7 : i32
    %24 = llvm.getelementptr inbounds %9[%4, %23] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<80 x i32>
    %25 = llvm.load %24 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %26 = llvm.xor %22, %25  : i32
    %27 = llvm.getelementptr inbounds %9[%4, %11] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<80 x i32>
    llvm.store %26, %27 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %28 = llvm.add %11, %0 : i32
    %29 = llvm.icmp "eq" %28, %8 : i32
    llvm.cond_br %29, ^bb2, ^bb1(%28 : i32) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.intr.lifetime.end 320, %10 : !llvm.ptr
    llvm.return %4 : i32
  }
}

