#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @isqrt(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> i32 attributes {passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(16384 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.load %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %5 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.br ^bb1(%4, %0 : i32, i32)
  ^bb1(%6: i32, %7: i32):  // 2 preds: ^bb0, ^bb1
    %8 = llvm.or %6, %7  : i32
    %9 = llvm.mul %8, %8  : i32
    %10 = llvm.icmp "sgt" %9, %5 : i32
    %11 = llvm.select %10, %6, %8 : i1, i32
    %12 = llvm.lshr %7, %1  : i32
    %13 = llvm.icmp "ult" %7, %2 : i32
    llvm.cond_br %13, ^bb2, ^bb1(%11, %12 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.store %11, %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    llvm.return %3 : i32
  }
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}) -> i32 attributes {passthrough = ["nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.call @time(%0) : (!llvm.ptr) -> i32
    llvm.call @srand(%2) : (i32) -> ()
    %3 = llvm.call @rand() : () -> i32
    llvm.return %1 : i32
  }
  llvm.func local_unnamed_addr @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @time(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @rand() -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}
