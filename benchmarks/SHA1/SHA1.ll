; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks/SHA1/SHA1.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks/SHA1/SHA1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @SHA1(i32* nocapture noundef %0) local_unnamed_addr #0 {
  br label %3

2:                                                ; preds = %3
  ret void

3:                                                ; preds = %1, %3
  %4 = phi i64 [ 16, %1 ], [ %21, %3 ]
  %5 = add nsw i64 %4, -3
  %6 = getelementptr inbounds i32, i32* %0, i64 %5
  %7 = load i32, i32* %6, align 4, !tbaa !3
  %8 = add nsw i64 %4, -8
  %9 = getelementptr inbounds i32, i32* %0, i64 %8
  %10 = load i32, i32* %9, align 4, !tbaa !3
  %11 = xor i32 %10, %7
  %12 = add nsw i64 %4, -14
  %13 = getelementptr inbounds i32, i32* %0, i64 %12
  %14 = load i32, i32* %13, align 4, !tbaa !3
  %15 = xor i32 %11, %14
  %16 = add nsw i64 %4, -16
  %17 = getelementptr inbounds i32, i32* %0, i64 %16
  %18 = load i32, i32* %17, align 4, !tbaa !3
  %19 = xor i32 %15, %18
  %20 = getelementptr inbounds i32, i32* %0, i64 %4
  store i32 %19, i32* %20, align 4, !tbaa !3
  %21 = add nuw nsw i64 %4, 1
  %22 = icmp eq i64 %21, 80
  br i1 %22, label %2, label %3, !llvm.loop !7
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nosync nounwind readnone uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
  %1 = alloca [80 x i32], align 16
  %2 = bitcast [80 x i32]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 320, i8* nonnull %2) #3
  br label %3

3:                                                ; preds = %3, %0
  %4 = phi i64 [ 16, %0 ], [ %21, %3 ]
  %5 = add nsw i64 %4, -3
  %6 = getelementptr inbounds [80 x i32], [80 x i32]* %1, i64 0, i64 %5
  %7 = load i32, i32* %6, align 4, !tbaa !3
  %8 = add nsw i64 %4, -8
  %9 = getelementptr inbounds [80 x i32], [80 x i32]* %1, i64 0, i64 %8
  %10 = load i32, i32* %9, align 4, !tbaa !3
  %11 = xor i32 %10, %7
  %12 = add nsw i64 %4, -14
  %13 = getelementptr inbounds [80 x i32], [80 x i32]* %1, i64 0, i64 %12
  %14 = load i32, i32* %13, align 4, !tbaa !3
  %15 = xor i32 %11, %14
  %16 = add nsw i64 %4, -16
  %17 = getelementptr inbounds [80 x i32], [80 x i32]* %1, i64 0, i64 %16
  %18 = load i32, i32* %17, align 4, !tbaa !3
  %19 = xor i32 %15, %18
  %20 = getelementptr inbounds [80 x i32], [80 x i32]* %1, i64 0, i64 %4
  store i32 %19, i32* %20, align 4, !tbaa !3
  %21 = add nuw nsw i64 %4, 1
  %22 = icmp eq i64 %21, 80
  br i1 %22, label %23, label %3, !llvm.loop !7

23:                                               ; preds = %3
  call void @llvm.lifetime.end.p0i8(i64 320, i8* nonnull %2) #3
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { nofree nosync nounwind readnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{!"clang version 14.0.6 (https://github.com/llvm/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!"llvm.loop.unroll.disable"}
