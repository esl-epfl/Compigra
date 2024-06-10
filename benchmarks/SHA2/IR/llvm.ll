; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/SHA2/SHA2.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/SHA2/SHA2.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind readonly uwtable
define dso_local i32 @SHA2(i32* nocapture noundef readonly %0) local_unnamed_addr #0 {
  br label %8

2:                                                ; preds = %8
  %3 = shl nsw i32 %27, 1
  %4 = add nsw i32 %3, %14
  %5 = add nsw i32 %4, %30
  %6 = add nsw i32 %5, %12
  %7 = add nsw i32 %6, %11
  ret i32 %7

8:                                                ; preds = %1, %8
  %9 = phi i32 [ 0, %1 ], [ %31, %8 ]
  %10 = phi i32 [ 0, %1 ], [ %11, %8 ]
  %11 = phi i32 [ 0, %1 ], [ %12, %8 ]
  %12 = phi i32 [ 0, %1 ], [ %30, %8 ]
  %13 = phi i32 [ 0, %1 ], [ %14, %8 ]
  %14 = phi i32 [ 0, %1 ], [ %27, %8 ]
  %15 = shl i32 %14, 5
  %16 = ashr i32 %14, 27
  %17 = or i32 %15, %16
  %18 = and i32 %12, %13
  %19 = xor i32 %13, -1
  %20 = and i32 %11, %19
  %21 = getelementptr inbounds i32, i32* %0, i32 %9
  %22 = load i32, i32* %21, align 4, !tbaa !4
  %23 = add i32 %17, 1518500249
  %24 = add i32 %23, %18
  %25 = add i32 %24, %20
  %26 = add i32 %25, %10
  %27 = add i32 %26, %22
  %28 = shl i32 %13, 30
  %29 = ashr i32 %13, 2
  %30 = or i32 %28, %29
  %31 = add nuw nsw i32 %9, 1
  %32 = icmp eq i32 %31, 20
  br i1 %32, label %2, label %8, !llvm.loop !8
}

; Function Attrs: nofree nosync nounwind readnone uwtable
define dso_local i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind readonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
