; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//StringSearch/StringSearch.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//StringSearch/StringSearch.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@lowervec = dso_local local_unnamed_addr global <{ [256 x i8], [745 x i8] }> <{ [256 x i8] c"\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\7F\80\81\82\83\84\85\86\87\88\89\8A\8B\8C\8D\8E\8F\90\91\92\93\94\95\96\97\98\99\9A\9B\9C\9D\9E\9F\A0\A1\A2\A3\A4\A5\A6\A7\A8\A9\AA\AB\AC\AD\AE\AF\B0\B1\B2\B3\B4\B5\B6\B7\B8\B9\BA\BB\BC\BD\BE\BF\C0\C1\C2\C3\C4\C5\C6\C7\C8\C9\CA\CB\CC\CD\CE\CF\D0\D1\D2\D3\D4\D5\D6\D7\D8\D9\DA\DB\DC\DD\DE\DF\E0\E1\E2\E3\E4\E5\E6\E7\E8\E9\EA\EB\EC\ED\EE\EF\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\FA\FB\FC\FD\FE\FF", [745 x i8] zeroinitializer }>, align 1

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @StringSearch(i32* nocapture noundef readonly %0, i32* nocapture noundef %1, i8* nocapture noundef readonly %2, i8* nocapture noundef readonly %3) local_unnamed_addr #0 {
  %5 = load i32, i32* %0, align 4, !tbaa !4
  %6 = load i32, i32* %1, align 4, !tbaa !4
  %7 = add i32 %5, -1
  %8 = icmp sgt i32 %5, 1
  br i1 %8, label %9, label %29

9:                                                ; preds = %4
  %10 = getelementptr inbounds i8, i8* %2, i32 %7
  %11 = load i8, i8* %10, align 1, !tbaa !8
  %12 = sext i8 %11 to i32
  %13 = getelementptr inbounds i8, i8* %3, i32 %12
  %14 = load i8, i8* %13, align 1, !tbaa !8
  br label %15

15:                                               ; preds = %9, %15
  %16 = phi i32 [ 0, %9 ], [ %27, %15 ]
  %17 = phi i32 [ %6, %9 ], [ %26, %15 ]
  %18 = getelementptr inbounds i8, i8* %2, i32 %16
  %19 = load i8, i8* %18, align 1, !tbaa !8
  %20 = sext i8 %19 to i32
  %21 = getelementptr inbounds i8, i8* %3, i32 %20
  %22 = load i8, i8* %21, align 1, !tbaa !8
  %23 = icmp eq i8 %22, %14
  %24 = xor i32 %16, -1
  %25 = add i32 %5, %24
  %26 = select i1 %23, i32 %25, i32 %17
  %27 = add nuw nsw i32 %16, 1
  %28 = icmp eq i32 %27, %7
  br i1 %28, label %29, label %15, !llvm.loop !9

29:                                               ; preds = %15, %4
  %30 = phi i32 [ %6, %4 ], [ %26, %15 ]
  store i32 %30, i32* %1, align 4, !tbaa !4
  ret void
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nosync nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
  %1 = alloca [20 x i8], align 1
  %2 = getelementptr inbounds [20 x i8], [20 x i8]* %1, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 20, i8* nonnull %2) #3
  %3 = load i32, i32* inttoptr (i32 20 to i32*), align 4, !tbaa !4
  %4 = load i32, i32* inttoptr (i32 2 to i32*), align 4, !tbaa !4
  %5 = add i32 %3, -1
  %6 = icmp sgt i32 %3, 1
  br i1 %6, label %7, label %25

7:                                                ; preds = %0
  %8 = load i8, i8* getelementptr inbounds (<{ [256 x i8], [745 x i8] }>, <{ [256 x i8], [745 x i8] }>* @lowervec, i32 0, i32 0, i32 0), align 1, !tbaa !8
  %9 = add i32 %3, -1
  %10 = icmp eq i32 %5, 1
  br i1 %10, label %25, label %11, !llvm.loop !9

11:                                               ; preds = %7, %11
  %12 = phi i32 [ %23, %11 ], [ 1, %7 ]
  %13 = phi i32 [ %22, %11 ], [ %9, %7 ]
  %14 = getelementptr inbounds [20 x i8], [20 x i8]* %1, i32 0, i32 %12
  %15 = load i8, i8* %14, align 1, !tbaa !8
  %16 = sext i8 %15 to i32
  %17 = getelementptr inbounds <{ [256 x i8], [745 x i8] }>, <{ [256 x i8], [745 x i8] }>* @lowervec, i32 0, i32 0, i32 %16
  %18 = load i8, i8* %17, align 1, !tbaa !8
  %19 = icmp eq i8 %18, %8
  %20 = xor i32 %12, -1
  %21 = add i32 %3, %20
  %22 = select i1 %19, i32 %21, i32 %13
  %23 = add nuw nsw i32 %12, 1
  %24 = icmp eq i32 %23, %5
  br i1 %24, label %25, label %11, !llvm.loop !9

25:                                               ; preds = %11, %7, %0
  %26 = phi i32 [ %4, %0 ], [ %9, %7 ], [ %22, %11 ]
  store i32 %26, i32* inttoptr (i32 2 to i32*), align 4, !tbaa !4
  call void @llvm.lifetime.end.p0i8(i64 20, i8* nonnull %2) #3
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { nofree nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{!"clang version 14.0.6 (https://github.com/llvm/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!6, !6, i64 0}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
