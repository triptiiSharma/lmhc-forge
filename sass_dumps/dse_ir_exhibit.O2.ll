; ModuleID = 'src/dse_ir_exhibit.c'
source_filename = "src/dse_ir_exhibit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@sink = dso_local global i32 0, align 4

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @fill_read_scrub_naive() local_unnamed_addr #0 {
  %1 = alloca [1024 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 4096, ptr nonnull %1) #2
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 0, %0 ], [ %12, %2 ]
  %4 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %0 ], [ %13, %2 ]
  %5 = mul <4 x i32> %4, splat (i32 1234567)
  %6 = mul <4 x i32> %4, splat (i32 1234567)
  %7 = add <4 x i32> %6, splat (i32 4938268)
  %8 = xor <4 x i32> %5, splat (i32 -559038737)
  %9 = xor <4 x i32> %7, splat (i32 -559038737)
  %10 = getelementptr inbounds nuw [1024 x i32], ptr %1, i64 0, i64 %3
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store <4 x i32> %8, ptr %10, align 16, !tbaa !5
  store <4 x i32> %9, ptr %11, align 16, !tbaa !5
  %12 = add nuw i64 %3, 8
  %13 = add <4 x i32> %4, splat (i32 8)
  %14 = icmp eq i64 %12, 1024
  br i1 %14, label %15, label %2, !llvm.loop !9

15:                                               ; preds = %2
  %16 = load i32, ptr %1, align 16, !tbaa !5
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 4092
  %18 = load i32, ptr %17, align 4, !tbaa !5
  %19 = xor i32 %18, %16
  store volatile i32 %19, ptr @sink, align 4, !tbaa !5
  call void @llvm.lifetime.end.p0(i64 4096, ptr nonnull %1) #2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @fill_read_scrub_hardened() local_unnamed_addr #0 {
  %1 = alloca [1024 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 4096, ptr nonnull %1) #2
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 0, %0 ], [ %12, %2 ]
  %4 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %0 ], [ %13, %2 ]
  %5 = mul <4 x i32> %4, splat (i32 1234567)
  %6 = mul <4 x i32> %4, splat (i32 1234567)
  %7 = add <4 x i32> %6, splat (i32 4938268)
  %8 = xor <4 x i32> %5, splat (i32 -559038737)
  %9 = xor <4 x i32> %7, splat (i32 -559038737)
  %10 = getelementptr inbounds nuw [1024 x i32], ptr %1, i64 0, i64 %3
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store <4 x i32> %8, ptr %10, align 16, !tbaa !5
  store <4 x i32> %9, ptr %11, align 16, !tbaa !5
  %12 = add nuw i64 %3, 8
  %13 = add <4 x i32> %4, splat (i32 8)
  %14 = icmp eq i64 %12, 1024
  br i1 %14, label %15, label %2, !llvm.loop !13

15:                                               ; preds = %2
  %16 = load i32, ptr %1, align 16, !tbaa !5
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 4092
  %18 = load i32, ptr %17, align 4, !tbaa !5
  %19 = xor i32 %18, %16
  store volatile i32 %19, ptr @sink, align 4, !tbaa !5
  %20 = getelementptr inbounds i8, ptr %1, i64 4
  %21 = getelementptr inbounds i8, ptr %1, i64 8
  %22 = getelementptr inbounds i8, ptr %1, i64 12
  %23 = getelementptr inbounds i8, ptr %1, i64 16
  %24 = getelementptr inbounds i8, ptr %1, i64 20
  %25 = getelementptr inbounds i8, ptr %1, i64 24
  %26 = getelementptr inbounds i8, ptr %1, i64 28
  br label %27

27:                                               ; preds = %27, %15
  %28 = phi i64 [ 0, %15 ], [ %37, %27 ]
  %29 = getelementptr inbounds nuw i32, ptr %1, i64 %28
  store volatile i32 0, ptr %29, align 16, !tbaa !5
  %30 = getelementptr inbounds i32, ptr %20, i64 %28
  store volatile i32 0, ptr %30, align 4, !tbaa !5
  %31 = getelementptr inbounds i32, ptr %21, i64 %28
  store volatile i32 0, ptr %31, align 8, !tbaa !5
  %32 = getelementptr inbounds i32, ptr %22, i64 %28
  store volatile i32 0, ptr %32, align 4, !tbaa !5
  %33 = getelementptr inbounds i32, ptr %23, i64 %28
  store volatile i32 0, ptr %33, align 16, !tbaa !5
  %34 = getelementptr inbounds i32, ptr %24, i64 %28
  store volatile i32 0, ptr %34, align 4, !tbaa !5
  %35 = getelementptr inbounds i32, ptr %25, i64 %28
  store volatile i32 0, ptr %35, align 8, !tbaa !5
  %36 = getelementptr inbounds i32, ptr %26, i64 %28
  store volatile i32 0, ptr %36, align 4, !tbaa !5
  %37 = add nuw nsw i64 %28, 8
  %38 = icmp eq i64 %37, 1024
  br i1 %38, label %39, label %27, !llvm.loop !14

39:                                               ; preds = %27
  call void @llvm.lifetime.end.p0(i64 4096, ptr nonnull %1) #2
  ret void
}

attributes #0 = { nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Debian clang version 21.1.8 (7+b1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10}
