#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  cgra.func @isqrt(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i16 {llvm.zeroext}) attributes {CConv = #llvm.cconv<ccc>, argNames = ["in0"], linkage = #llvm.linkage<external>, memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]], resNames = ["out0"], unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.constant(0 : i16) {stage = "init"} : i16
    %1 = llvm.mlir.constant(16384 : i16) {stage = "init"} : i16
    %2 = llvm.mlir.constant(1 : i16) {stage = "init"} : i16
    %3 = llvm.mlir.constant(2 : i16) {stage = "init"} : i16
    %4 = llvm.load %arg0 {alignment = 4 : i64, stage = "init", tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %5 = merge %11, %0 {stage = "loop"} : i16
    %6 = merge %12, %1 {stage = "loop"} : i16
    %7 = llvm.or %5, %6  {stage = "loop"} : i16
    %8 = llvm.zext %7 {stage = "loop"} : i16 to i32
    %9 = llvm.mul %8, %8 {stage = "loop"} : i32
    %10 = llvm.sub %4, %9 {stage = "loop"} : i32
    %11 = bsfa %10 [%5, %7]  {stage = "loop"} : i32
    %12 = llvm.lshr %6, %2  {stage = "loop"} : i16
    %13 = blt [%6, %3] %5 {stage = "loop"} : i16
    llvm.return {stage = "fini"} %11 : i16
  }
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}) -> i32 attributes {passthrough = ["nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.call @time(%0) : (!llvm.ptr) -> i32
    llvm.call @srand(%2) : (i32) -> ()
    llvm.return %1 : i32
  }
  llvm.func local_unnamed_addr @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @time(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

