; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/ReverseBits/ReverseBits.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/ReverseBits/ReverseBits.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind readnone uwtable
define dso_local i32 @ReverseBits(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %14, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %10, %4 ], [ 0, %2 ]
  %6 = phi i32 [ %12, %4 ], [ 0, %2 ]
  %7 = phi i32 [ %11, %4 ], [ %0, %2 ]
  %8 = shl i32 %5, 1
  %9 = and i32 %7, 1
  %10 = or i32 %8, %9
  %11 = lshr i32 %7, 1
  %12 = add nuw i32 %6, 1
  %13 = icmp eq i32 %12, %1
  br i1 %13, label %14, label %4, !llvm.loop !4

14:                                               ; preds = %4, %2
  %15 = phi i32 [ 0, %2 ], [ %10, %4 ]
  ret i32 %15
}

; Function Attrs: nofree norecurse nosync nounwind readnone uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind readnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{!"clang version 14.0.6 (https://github.com/llvm/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!4 = distinct !{!4, !5, !6}
!5 = !{!"llvm.loop.mustprogress"}
!6 = !{!"llvm.loop.unroll.disable"}
