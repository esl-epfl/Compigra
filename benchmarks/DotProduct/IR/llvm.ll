target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local i32 @DotProduct(i32* nocapture noundef readonly %0, i32* nocapture noundef readonly %1, i32* nocapture noundef readonly %2, i32* nocapture noundef writeonly %3) local_unnamed_addr #0 {
  %5 = load i32, i32* %2, align 4, !tbaa !4
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %9, label %7

7:                                                ; preds = %9, %4
  %8 = phi i32 [ 0, %4 ], [ %17, %9 ]
  store i32 %8, i32* %3, align 4, !tbaa !4
  ret i32 %8

9:                                                ; preds = %4, %9
  %10 = phi i32 [ %18, %9 ], [ 0, %4 ]
  %11 = phi i32 [ %17, %9 ], [ 0, %4 ]
  %12 = getelementptr inbounds i32, i32* %0, i32 %10
  %13 = load i32, i32* %12, align 4, !tbaa !4
  %14 = getelementptr inbounds i32, i32* %1, i32 %10
  %15 = load i32, i32* %14, align 4, !tbaa !4
  %16 = mul nsw i32 %15, %13
  %17 = add nsw i32 %16, %11
  %18 = add nuw nsw i32 %10, 1
  %19 = icmp eq i32 %18, %5
  br i1 %19, label %7, label %9, !llvm.loop !8
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
