; ModuleID = '/home/yuxuan/Projects/24S/Compigra/benchmarks//FIR/FIR.c'
source_filename = "/home/yuxuan/Projects/24S/Compigra/benchmarks//FIR/FIR.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@__const.main.input = private unnamed_addr constant [10 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00, double 5.000000e+00, double 4.000000e+00, double 3.000000e+00, double 2.000000e+00, double 1.000000e+00, double 0.000000e+00], align 8
@.str.1 = private unnamed_addr constant [17 x i8] c"output[%d] = %f\0A\00", align 1
@str = private unnamed_addr constant [17 x i8] c"Filtered output:\00", align 1

; Function Attrs: nofree nosync nounwind uwtable
define dso_local void @FIR(i32* nocapture noundef readonly %0, i32* nocapture noundef readonly %1, float* nocapture noundef readonly %2, float* nocapture noundef %3, float* nocapture noundef readonly %4) local_unnamed_addr #0 {
  %6 = load i32, i32* %0, align 4, !tbaa !4
  %7 = load i32, i32* %1, align 4, !tbaa !4
  %8 = icmp sgt i32 %6, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %5
  %10 = getelementptr inbounds float, float* %3, i32 %7
  br label %12

11:                                               ; preds = %23, %5
  ret void

12:                                               ; preds = %9, %23
  %13 = phi i32 [ 0, %9 ], [ %24, %23 ]
  %14 = icmp slt i32 %7, %13
  br i1 %14, label %23, label %15

15:                                               ; preds = %12
  %16 = getelementptr inbounds float, float* %4, i32 %13
  %17 = load float, float* %16, align 4, !tbaa !8
  %18 = sub nsw i32 %7, %13
  %19 = getelementptr inbounds float, float* %2, i32 %18
  %20 = load float, float* %19, align 4, !tbaa !8
  %21 = load float, float* %10, align 4, !tbaa !8
  %22 = tail call float @llvm.fmuladd.f32(float %17, float %20, float %21)
  store float %22, float* %10, align 4, !tbaa !8
  br label %23

23:                                               ; preds = %12, %15
  %24 = add nuw nsw i32 %13, 1
  %25 = icmp eq i32 %24, %6
  br i1 %25, label %11, label %12, !llvm.loop !10
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #3 {
  %1 = alloca [5 x double], align 8
  %2 = alloca [10 x double], align 8
  %3 = bitcast [5 x double]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %3) #10
  %4 = getelementptr inbounds [5 x double], [5 x double]* %1, i32 0, i32 0
  store double 1.000000e-01, double* %4, align 8
  %5 = getelementptr inbounds [5 x double], [5 x double]* %1, i32 0, i32 1
  store double 1.500000e-01, double* %5, align 8
  %6 = getelementptr inbounds [5 x double], [5 x double]* %1, i32 0, i32 2
  store double 5.000000e-01, double* %6, align 8
  %7 = getelementptr inbounds [5 x double], [5 x double]* %1, i32 0, i32 3
  store double 1.500000e-01, double* %7, align 8
  %8 = getelementptr inbounds [5 x double], [5 x double]* %1, i32 0, i32 4
  store double 1.000000e-01, double* %8, align 8
  %9 = bitcast [10 x double]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %9) #10
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* noundef nonnull align 8 dereferenceable(80) %9, i8* noundef nonnull align 8 dereferenceable(80) bitcast ([10 x double]* @__const.main.input to i8*), i32 80, i1 false)
  %10 = tail call noalias dereferenceable_or_null(80) i8* @malloc(i32 noundef 80) #10
  %11 = bitcast i8* %10 to double*
  %12 = getelementptr inbounds [10 x double], [10 x double]* %2, i32 0, i32 0
  %13 = call i32 bitcast (i32 (...)* @fir_filter to i32 (double*, double*, double*, i32, i32)*)(double* noundef nonnull %12, double* noundef %11, double* noundef nonnull %4, i32 noundef 5, i32 noundef 10) #10
  %14 = call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([17 x i8], [17 x i8]* @str, i32 0, i32 0))
  br label %16

15:                                               ; preds = %16
  call void @free(i8* noundef nonnull %10) #10
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %9) #10
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %3) #10
  ret i32 0

16:                                               ; preds = %0, %16
  %17 = phi i32 [ 0, %0 ], [ %21, %16 ]
  %18 = getelementptr inbounds double, double* %11, i32 %17
  %19 = load double, double* %18, align 4, !tbaa !13
  %20 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([17 x i8], [17 x i8]* @.str.1, i32 0, i32 0), i32 noundef %17, double noundef %19)
  %21 = add nuw nsw i32 %17, 1
  %22 = icmp eq i32 %21, 10
  br i1 %22, label %15, label %16, !llvm.loop !15
}

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg) #4

; Function Attrs: inaccessiblememonly mustprogress nofree nounwind willreturn
declare dso_local noalias noundef i8* @malloc(i32 noundef) local_unnamed_addr #5

declare dso_local i32 @fir_filter(...) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #7

; Function Attrs: inaccessiblemem_or_argmemonly mustprogress nounwind willreturn
declare dso_local void @free(i8* nocapture noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @puts(i8* nocapture noundef readonly) local_unnamed_addr #9

attributes #0 = { nofree nosync nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly mustprogress nofree nounwind willreturn }
attributes #5 = { inaccessiblememonly mustprogress nofree nounwind willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { inaccessiblemem_or_argmemonly mustprogress nounwind willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nofree nounwind }
attributes #10 = { nounwind }

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
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !6, i64 0}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !6, i64 0}
!15 = distinct !{!15, !11, !12}
