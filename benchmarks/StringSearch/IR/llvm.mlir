#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.mlir.global external local_unnamed_addr @lowervec() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<745xi8>) : !llvm.array<745 x i8>
    %2 = llvm.mlir.constant("\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF") : !llvm.array<256 x i8>
    %3 = llvm.mlir.undef : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    llvm.return %5 : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
  }
  llvm.func local_unnamed_addr @StringSearch(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(dense<0> : tensor<745xi8>) : !llvm.array<745 x i8>
    %4 = llvm.mlir.constant("\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF") : !llvm.array<256 x i8>
    %5 = llvm.mlir.undef : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %8 = llvm.mlir.addressof @lowervec : !llvm.ptr
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.add %arg0, %0  : i32
    %11 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %11, ^bb1, ^bb3(%arg1 : i32)
  ^bb1:  // pred: ^bb0
    %12 = llvm.getelementptr inbounds %arg2[%10] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %13 = llvm.load %12 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %14 = llvm.sext %13 : i8 to i32
    %15 = llvm.getelementptr inbounds %8[%9, %14] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<1001 x i8>
    %16 = llvm.load %15 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    llvm.br ^bb2(%9, %arg1 : i32, i32)
  ^bb2(%17: i32, %18: i32):  // 2 preds: ^bb1, ^bb2
    %19 = llvm.getelementptr inbounds %arg2[%17] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %20 = llvm.load %19 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %21 = llvm.sext %20 : i8 to i32
    %22 = llvm.getelementptr inbounds %8[%9, %21] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<1001 x i8>
    %23 = llvm.load %22 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %24 = llvm.icmp "eq" %23, %16 : i8
    %25 = llvm.xor %17, %0  : i32
    %26 = llvm.add %25, %arg0  : i32
    %27 = llvm.select %24, %26, %18 : i1, i32
    %28 = llvm.add %17, %1  : i32
    %29 = llvm.icmp "eq" %28, %10 : i32
    llvm.cond_br %29, ^bb3(%27 : i32), ^bb2(%28, %27 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb3(%30: i32):  // 2 preds: ^bb0, ^bb2
    llvm.return %30 : i32
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
