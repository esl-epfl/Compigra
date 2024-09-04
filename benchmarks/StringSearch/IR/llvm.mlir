#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.mlir.global external local_unnamed_addr @lowervec() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<745xi8>) : !llvm.array<745 x i8>
    %2 = llvm.mlir.constant("\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF") : !llvm.array<256 x i8>
    %3 = llvm.mlir.undef : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    llvm.return %5 : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
  }
  llvm.func local_unnamed_addr @StringSearch(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %4 = llvm.load %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %5 = llvm.add %3, %0  : i32
    %6 = llvm.icmp "sgt" %3, %1 : i32
    llvm.cond_br %6, ^bb1, ^bb3(%4 : i32)
  ^bb1:  // pred: ^bb0
    %7 = llvm.getelementptr inbounds %arg2[%5] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %8 = llvm.load %7 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %9 = llvm.sext %8 : i8 to i32
    %10 = llvm.getelementptr inbounds %arg3[%9] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %11 = llvm.load %10 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    llvm.br ^bb2(%2, %4 : i32, i32)
  ^bb2(%12: i32, %13: i32):  // 2 preds: ^bb1, ^bb2
    %14 = llvm.getelementptr inbounds %arg2[%12] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %15 = llvm.load %14 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %16 = llvm.sext %15 : i8 to i32
    %17 = llvm.getelementptr inbounds %arg3[%16] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %18 = llvm.load %17 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %19 = llvm.icmp "eq" %18, %11 : i8
    %20 = llvm.xor %12, %0  : i32
    %21 = llvm.add %3, %20  : i32
    %22 = llvm.select %19, %21, %13 : i1, i32
    %23 = llvm.add %12, %1  : i32
    %24 = llvm.icmp "eq" %23, %5 : i32
    llvm.cond_br %24, ^bb3(%22 : i32), ^bb2(%23, %22 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb3(%25: i32):  // 2 preds: ^bb0, ^bb2
    llvm.store %25, %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @main() -> i32 attributes {passthrough = ["nofree", "nosync", "nounwind", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "pentium4"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(20 : i64) : i64
    %3 = llvm.mlir.constant(20 : i32) : i32
    %4 = llvm.inttoptr %3 : i32 to !llvm.ptr
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.inttoptr %5 : i32 to !llvm.ptr
    %7 = llvm.mlir.constant(-1 : i32) : i32
    %8 = llvm.mlir.constant(0 : i8) : i8
    %9 = llvm.mlir.constant(dense<0> : tensor<745xi8>) : !llvm.array<745 x i8>
    %10 = llvm.mlir.constant("\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF") : !llvm.array<256 x i8>
    %11 = llvm.mlir.undef : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %13 = llvm.insertvalue %9, %12[1] : !llvm.struct<packed (array<256 x i8>, array<745 x i8>)> 
    %14 = llvm.mlir.addressof @lowervec : !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.array<20 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.getelementptr inbounds %15[%1, %1] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<20 x i8>
    llvm.intr.lifetime.start 20, %16 : !llvm.ptr
    %17 = llvm.load %4 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %18 = llvm.load %6 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %19 = llvm.add %17, %7  : i32
    %20 = llvm.icmp "sgt" %17, %0 : i32
    llvm.cond_br %20, ^bb1, ^bb3(%18 : i32)
  ^bb1:  // pred: ^bb0
    %21 = llvm.load %14 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %22 = llvm.add %17, %7  : i32
    %23 = llvm.icmp "eq" %19, %0 : i32
    llvm.cond_br %23, ^bb3(%22 : i32), ^bb2(%0, %22 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb2(%24: i32, %25: i32):  // 2 preds: ^bb1, ^bb2
    %26 = llvm.getelementptr inbounds %15[%1, %24] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<20 x i8>
    %27 = llvm.load %26 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %28 = llvm.sext %27 : i8 to i32
    %29 = llvm.getelementptr inbounds %14[%1, 0, %28] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<packed (array<256 x i8>, array<745 x i8>)>
    %30 = llvm.load %29 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %31 = llvm.icmp "eq" %30, %21 : i8
    %32 = llvm.xor %24, %7  : i32
    %33 = llvm.add %17, %32  : i32
    %34 = llvm.select %31, %33, %25 : i1, i32
    %35 = llvm.add %24, %0  : i32
    %36 = llvm.icmp "eq" %35, %19 : i32
    llvm.cond_br %36, ^bb3(%34 : i32), ^bb2(%35, %34 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb3(%37: i32):  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.store %37, %6 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : i32, !llvm.ptr
    llvm.intr.lifetime.end 20, %16 : !llvm.ptr
    llvm.return %1 : i32
  }
}
