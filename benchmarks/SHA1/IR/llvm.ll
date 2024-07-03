; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//SHA1/SHA1.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//SHA1/SHA1.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @SHA1(i32* nocapture noundef %0) local_unnamed_addr #0 {
  br label %3

2:                                                ; preds = %3
  ret void

3:                                                ; preds = %1, %3
  %4 = phi i32 [ 16, %1 ], [ %21, %3 ]
  %5 = add nsw i32 %4, -3
  %6 = getelementptr inbounds i32, i32* %0, i32 %5
  %7 = load i32, i32* %6, align 4, !tbaa !4
  %8 = add nsw i32 %4, -8
  %9 = getelementptr inbounds i32, i32* %0, i32 %8
  %10 = load i32, i32* %9, align 4, !tbaa !4
  %11 = xor i32 %10, %7
  %12 = add nsw i32 %4, -14
  %13 = getelementptr inbounds i32, i32* %0, i32 %12
  %14 = load i32, i32* %13, align 4, !tbaa !4
  %15 = xor i32 %11, %14
  %16 = add nsw i32 %4, -16
  %17 = getelementptr inbounds i32, i32* %0, i32 %16
  %18 = load i32, i32* %17, align 4, !tbaa !4
  %19 = xor i32 %15, %18
  %20 = getelementptr inbounds i32, i32* %0, i32 %4
  store i32 %19, i32* %20, align 4, !tbaa !4
  %21 = add nuw nsw i32 %4, 1
  %22 = icmp eq i32 %21, 80
  br i1 %22, label %2, label %3, !llvm.loop !8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
