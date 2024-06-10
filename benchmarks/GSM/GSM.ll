; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/GSM/GSM.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/GSM/GSM.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind readonly uwtable
define dso_local i32 @GSM(i32 noundef %0, i32 %1, i32* nocapture noundef readonly %2) local_unnamed_addr #0 {
  br label %5

4:                                                ; preds = %5
  ret i32 %16

5:                                                ; preds = %3, %5
  %6 = phi i32 [ 0, %3 ], [ %17, %5 ]
  %7 = phi i32 [ %0, %3 ], [ %16, %5 ]
  %8 = getelementptr inbounds i32, i32* %2, i32 %6
  %9 = load i32, i32* %8, align 4, !tbaa !4
  %10 = icmp slt i32 %9, 0
  %11 = icmp eq i32 %9, -32768
  %12 = sub nsw i32 0, %9
  %13 = select i1 %11, i32 32767, i32 %12
  %14 = select i1 %10, i32 %13, i32 %9
  %15 = icmp sgt i32 %14, %7
  %16 = select i1 %15, i32 %14, i32 %7
  %17 = add nuw nsw i32 %6, 1
  %18 = icmp eq i32 %17, 40
  br i1 %18, label %4, label %5, !llvm.loop !8
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
