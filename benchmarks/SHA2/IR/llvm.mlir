#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func local_unnamed_addr @SHA2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(27 : i32) : i32
    %3 = llvm.mlir.constant(-1 : i32) : i32
    %4 = llvm.mlir.constant(1518500249 : i32) : i32
    %5 = llvm.mlir.constant(30 : i32) : i32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(20 : i32) : i32
    llvm.br ^bb2(%0, %0, %0, %0, %0, %0 : i32, i32, i32, i32, i32, i32)
  ^bb1:  // pred: ^bb2
    %9 = llvm.shl %32, %7  : i32
    %10 = llvm.add %9, %19  : i32
    %11 = llvm.add %10, %35  : i32
    %12 = llvm.add %11, %17  : i32
    %13 = llvm.add %12, %16  : i32
    llvm.return %13 : i32
  ^bb2(%14: i32, %15: i32, %16: i32, %17: i32, %18: i32, %19: i32):  // 2 preds: ^bb0, ^bb2
    %20 = llvm.shl %19, %1  : i32
    %21 = llvm.ashr %19, %2  : i32
    %22 = llvm.or %20, %21  : i32
    %23 = llvm.and %17, %18  : i32
    %24 = llvm.xor %18, %3  : i32
    %25 = llvm.and %16, %24  : i32
    %26 = llvm.getelementptr inbounds %arg0[%14] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %27 = llvm.load %26 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %28 = llvm.add %22, %4  : i32
    %29 = llvm.add %28, %23  : i32
    %30 = llvm.add %29, %25  : i32
    %31 = llvm.add %30, %15  : i32
    %32 = llvm.add %31, %27  : i32
    %33 = llvm.shl %18, %5  : i32
    %34 = llvm.ashr %18, %6  : i32
    %35 = llvm.or %33, %34  : i32
    %36 = llvm.add %14, %7  : i32
    %37 = llvm.icmp "eq" %36, %8 : i32
    llvm.cond_br %37, ^bb1, ^bb2(%36, %16, %17, %35, %19, %32 : i32, i32, i32, i32, i32, i32) {loop_annotation = #loop_annotation}
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
