; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/isqrt/isqrt.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/isqrt/isqrt.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind readonly uwtable
define dso_local i32 @isqrt(i32* nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = load i32, i32* %0, align 4, !tbaa !4
  br label %3

3:                                                ; preds = %1, %3
  %4 = phi i32 [ 0, %1 ], [ %9, %3 ]
  %5 = phi i32 [ 16384, %1 ], [ %10, %3 ]
  %6 = or i32 %4, %5
  %7 = mul i32 %6, %6
  %8 = icmp ugt i32 %7, %2
  %9 = select i1 %8, i32 %4, i32 %6
  %10 = lshr i32 %5, 1
  %11 = icmp ult i32 %5, 2
  br i1 %11, label %12, label %3, !llvm.loop !8

12:                                               ; preds = %3
  ret i32 %9
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 noundef %0, i8** nocapture noundef readnone %1) local_unnamed_addr #1 {
  %3 = tail call i32 @time(i32* noundef null) #3
  tail call void @srand(i32 noundef %3) #3
  ret i32 1
}

; Function Attrs: nounwind
declare dso_local void @srand(i32 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i32 @time(i32* noundef) local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind readonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
