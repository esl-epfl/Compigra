; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//ReverseBits/ReverseBits.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//ReverseBits/ReverseBits.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local i32 @ReverseBits(i32* nocapture noundef %0, i32* nocapture noundef %1, i32* nocapture noundef writeonly %2) local_unnamed_addr #0 {
  %4 = load i32, i32* %0, align 4, !tbaa !4
  %5 = load i32, i32* %1, align 4, !tbaa !4
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %17, label %7

7:                                                ; preds = %3, %7
  %8 = phi i32 [ %13, %7 ], [ 0, %3 ]
  %9 = phi i32 [ %15, %7 ], [ 0, %3 ]
  %10 = phi i32 [ %14, %7 ], [ %4, %3 ]
  %11 = shl i32 %8, 1
  %12 = and i32 %10, 1
  %13 = or i32 %11, %12
  %14 = lshr i32 %10, 1
  %15 = add nuw i32 %9, 1
  %16 = icmp eq i32 %15, %5
  br i1 %16, label %17, label %7, !llvm.loop !8

17:                                               ; preds = %7, %3
  %18 = phi i32 [ %4, %3 ], [ %14, %7 ]
  %19 = phi i32 [ 0, %3 ], [ %13, %7 ]
  store i32 %18, i32* %0, align 4, !tbaa !4
  store i32 %5, i32* %1, align 4, !tbaa !4
  store i32 %19, i32* %2, align 4, !tbaa !4
  ret i32 0
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
  %1 = load i32, i32* inttoptr (i32 32 to i32*), align 32, !tbaa !4
  %2 = icmp ne i32 %1, 0
  tail call void @llvm.assume(i1 %2)
  br label %3

3:                                                ; preds = %3, %0
  %4 = phi i32 [ %5, %3 ], [ 0, %0 ]
  %5 = add nuw i32 %4, 1
  %6 = icmp ne i32 %5, %1
  tail call void @llvm.assume(i1 %6)
  br label %3
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }

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
