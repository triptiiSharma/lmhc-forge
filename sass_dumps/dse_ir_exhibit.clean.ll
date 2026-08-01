; ModuleID = 'sass_dumps/dse_ir_exhibit.O0.ll'
source_filename = "src/dse_ir_exhibit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@sink = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fill_read_scrub_naive() #0 {
  %1 = alloca [1024 x i32], align 16
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %13, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 1024
  br i1 %5, label %6, label %16

6:                                                ; preds = %3
  %7 = load i32, ptr %2, align 4
  %8 = mul i32 %7, 1234567
  %9 = xor i32 %8, -559038737
  %10 = load i32, ptr %2, align 4
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 %11
  store i32 %9, ptr %12, align 4
  br label %13

13:                                               ; preds = %6
  %14 = load i32, ptr %2, align 4
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr %2, align 4
  br label %3, !llvm.loop !6

16:                                               ; preds = %3
  %17 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 0
  %18 = load i32, ptr %17, align 16
  %19 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 1023
  %20 = load i32, ptr %19, align 4
  %21 = xor i32 %18, %20
  store volatile i32 %21, ptr @sink, align 4
  store i32 0, ptr %2, align 4
  br label %22

22:                                               ; preds = %29, %16
  %23 = load i32, ptr %2, align 4
  %24 = icmp slt i32 %23, 1024
  br i1 %24, label %25, label %32

25:                                               ; preds = %22
  %26 = load i32, ptr %2, align 4
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 %27
  store i32 0, ptr %28, align 4
  br label %29

29:                                               ; preds = %25
  %30 = load i32, ptr %2, align 4
  %31 = add nsw i32 %30, 1
  store i32 %31, ptr %2, align 4
  br label %22, !llvm.loop !8

32:                                               ; preds = %22
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fill_read_scrub_hardened() #0 {
  %1 = alloca [1024 x i32], align 16
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 0
  store ptr %4, ptr %2, align 8
  store i32 0, ptr %3, align 4
  br label %5

5:                                                ; preds = %15, %0
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %6, 1024
  br i1 %7, label %8, label %18

8:                                                ; preds = %5
  %9 = load i32, ptr %3, align 4
  %10 = mul i32 %9, 1234567
  %11 = xor i32 %10, -559038737
  %12 = load i32, ptr %3, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 %13
  store i32 %11, ptr %14, align 4
  br label %15

15:                                               ; preds = %8
  %16 = load i32, ptr %3, align 4
  %17 = add nsw i32 %16, 1
  store i32 %17, ptr %3, align 4
  br label %5, !llvm.loop !9

18:                                               ; preds = %5
  %19 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 0
  %20 = load i32, ptr %19, align 16
  %21 = getelementptr inbounds [1024 x i32], ptr %1, i64 0, i64 1023
  %22 = load i32, ptr %21, align 4
  %23 = xor i32 %20, %22
  store volatile i32 %23, ptr @sink, align 4
  store i32 0, ptr %3, align 4
  br label %24

24:                                               ; preds = %32, %18
  %25 = load i32, ptr %3, align 4
  %26 = icmp slt i32 %25, 1024
  br i1 %26, label %27, label %35

27:                                               ; preds = %24
  %28 = load ptr, ptr %2, align 8
  %29 = load i32, ptr %3, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds i32, ptr %28, i64 %30
  store volatile i32 0, ptr %31, align 4
  br label %32

32:                                               ; preds = %27
  %33 = load i32, ptr %3, align 4
  %34 = add nsw i32 %33, 1
  store i32 %34, ptr %3, align 4
  br label %24, !llvm.loop !10

35:                                               ; preds = %24
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Debian clang version 21.1.8 (7+b1)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
