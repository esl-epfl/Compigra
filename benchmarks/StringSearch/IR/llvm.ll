; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/StringSearch/StringSearch.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/StringSearch/StringSearch.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@lowervec = dso_local local_unnamed_addr global <{ [256 x i8], [745 x i8] }> <{ [256 x i8] c"\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF", [745 x i8] zeroinitializer }>, align 1

; Function Attrs: nofree norecurse nosync nounwind readonly uwtable
define dso_local i32 @StringSearch(i32 noundef %0, i32 noundef %1, i8* nocapture noundef readonly %2) local_unnamed_addr #0 {
  %4 = add i32 %0, -1
  %5 = icmp sgt i32 %0, 1
  br i1 %5, label %6, label %26

6:                                                ; preds = %3
  %7 = getelementptr inbounds i8, i8* %2, i32 %4
  %8 = load i8, i8* %7, align 1, !tbaa !4
  %9 = sext i8 %8 to i32
  %10 = getelementptr inbounds [1001 x i8], [1001 x i8]* bitcast (<{ [256 x i8], [745 x i8] }>* @lowervec to [1001 x i8]*), i32 0, i32 %9
  %11 = load i8, i8* %10, align 1, !tbaa !4
  br label %12

12:                                               ; preds = %6, %12
  %13 = phi i32 [ 0, %6 ], [ %24, %12 ]
  %14 = phi i32 [ %1, %6 ], [ %23, %12 ]
  %15 = getelementptr inbounds i8, i8* %2, i32 %13
  %16 = load i8, i8* %15, align 1, !tbaa !4
  %17 = sext i8 %16 to i32
  %18 = getelementptr inbounds [1001 x i8], [1001 x i8]* bitcast (<{ [256 x i8], [745 x i8] }>* @lowervec to [1001 x i8]*), i32 0, i32 %17
  %19 = load i8, i8* %18, align 1, !tbaa !4
  %20 = icmp eq i8 %19, %11
  %21 = xor i32 %13, -1
  %22 = add i32 %21, %0
  %23 = select i1 %20, i32 %22, i32 %14
  %24 = add nuw nsw i32 %13, 1
  %25 = icmp eq i32 %24, %4
  br i1 %25, label %26, label %12, !llvm.loop !7

26:                                               ; preds = %12, %3
  %27 = phi i32 [ %1, %3 ], [ %23, %12 ]
  ret i32 %27
}

; Function Attrs: nofree nosync nounwind readonly uwtable
define dso_local i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind readonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{!"clang version 14.0.6 (https://github.com/llvm/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!"llvm.loop.unroll.disable"}
