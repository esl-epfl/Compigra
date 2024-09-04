; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//SHA2/SHA2.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//SHA2/SHA2.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local i32 @SHA2(i32* nocapture noundef readonly %0, i32* nocapture noundef writeonly %1) local_unnamed_addr #0 {
  br label %9

3:                                                ; preds = %9
  %4 = shl nsw i32 %28, 1
  %5 = add nsw i32 %4, %15
  %6 = add nsw i32 %5, %31
  %7 = add nsw i32 %6, %13
  %8 = add nsw i32 %7, %12
  store i32 %8, i32* %1, align 4, !tbaa !4
  ret i32 0

9:                                                ; preds = %2, %9
  %10 = phi i32 [ 0, %2 ], [ %32, %9 ]
  %11 = phi i32 [ 0, %2 ], [ %12, %9 ]
  %12 = phi i32 [ 0, %2 ], [ %13, %9 ]
  %13 = phi i32 [ 0, %2 ], [ %31, %9 ]
  %14 = phi i32 [ 0, %2 ], [ %15, %9 ]
  %15 = phi i32 [ 0, %2 ], [ %28, %9 ]
  %16 = shl i32 %15, 5
  %17 = ashr i32 %15, 27
  %18 = or i32 %16, %17
  %19 = and i32 %13, %14
  %20 = xor i32 %14, -1
  %21 = and i32 %12, %20
  %22 = getelementptr inbounds i32, i32* %0, i32 %10
  %23 = load i32, i32* %22, align 4, !tbaa !4
  %24 = add i32 %18, 1518500249
  %25 = add i32 %24, %19
  %26 = add i32 %25, %21
  %27 = add i32 %26, %11
  %28 = add i32 %27, %23
  %29 = shl i32 %14, 30
  %30 = ashr i32 %14, 2
  %31 = or i32 %29, %30
  %32 = add nuw nsw i32 %10, 1
  %33 = icmp eq i32 %32, 20
  br i1 %33, label %3, label %9, !llvm.loop !8
}

; Function Attrs: nofree nosync nounwind readnone uwtable
define dso_local i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
