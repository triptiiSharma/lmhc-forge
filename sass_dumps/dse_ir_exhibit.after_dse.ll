; ModuleID = 'sass_dumps/dse_ir_exhibit.O0.ll'
source_filename = "src/dse_ir_exhibit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@sink = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fill_read_scrub_naive(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %15, %1
  %5 = load i32, ptr %3, align 4
  %6 = icmp slt i32 %5, 1024
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = mul i32 %8, 1234567
  %10 = xor i32 %9, -559038737
  %11 = load ptr, ptr %2, align 8
  %12 = load i32, ptr %3, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i32, ptr %11, i64 %13
  store i32 %10, ptr %14, align 4
  br label %15

15:                                               ; preds = %7
  %16 = load i32, ptr %3, align 4
  %17 = add nsw i32 %16, 1
  store i32 %17, ptr %3, align 4
  br label %4, !llvm.loop !6

18:                                               ; preds = %4
  %19 = load ptr, ptr %2, align 8
  %20 = getelementptr inbounds i32, ptr %19, i64 0
  %21 = load i32, ptr %20, align 4
  %22 = load ptr, ptr %2, align 8
  %23 = getelementptr inbounds i32, ptr %22, i64 1023
  %24 = load i32, ptr %23, align 4
  %25 = xor i32 %21, %24
  store volatile i32 %25, ptr @sink, align 4
  store i32 0, ptr %3, align 4
  br label %26

26:                                               ; preds = %34, %18
  %27 = load i32, ptr %3, align 4
  %28 = icmp slt i32 %27, 1024
  br i1 %28, label %29, label %37

29:                                               ; preds = %26
  %30 = load ptr, ptr %2, align 8
  %31 = load i32, ptr %3, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr %30, i64 %32
  store i32 0, ptr %33, align 4
  br label %34

34:                                               ; preds = %29
  %35 = load i32, ptr %3, align 4
  %36 = add nsw i32 %35, 1
  store i32 %36, ptr %3, align 4
  br label %26, !llvm.loop !8

37:                                               ; preds = %26
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fill_read_scrub_hardened(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  store ptr %5, ptr %3, align 8
  store i32 0, ptr %4, align 4
  br label %6

6:                                                ; preds = %17, %1
  %7 = load i32, ptr %4, align 4
  %8 = icmp slt i32 %7, 1024
  br i1 %8, label %9, label %20

9:                                                ; preds = %6
  %10 = load i32, ptr %4, align 4
  %11 = mul i32 %10, 1234567
  %12 = xor i32 %11, -559038737
  %13 = load ptr, ptr %2, align 8
  %14 = load i32, ptr %4, align 4
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds i32, ptr %13, i64 %15
  store i32 %12, ptr %16, align 4
  br label %17

17:                                               ; preds = %9
  %18 = load i32, ptr %4, align 4
  %19 = add nsw i32 %18, 1
  store i32 %19, ptr %4, align 4
  br label %6, !llvm.loop !9

20:                                               ; preds = %6
  %21 = load ptr, ptr %2, align 8
  %22 = getelementptr inbounds i32, ptr %21, i64 0
  %23 = load i32, ptr %22, align 4
  %24 = load ptr, ptr %2, align 8
  %25 = getelementptr inbounds i32, ptr %24, i64 1023
  %26 = load i32, ptr %25, align 4
  %27 = xor i32 %23, %26
  store volatile i32 %27, ptr @sink, align 4
  store i32 0, ptr %4, align 4
  br label %28

28:                                               ; preds = %36, %20
  %29 = load i32, ptr %4, align 4
  %30 = icmp slt i32 %29, 1024
  br i1 %30, label %31, label %39

31:                                               ; preds = %28
  %32 = load ptr, ptr %3, align 8
  %33 = load i32, ptr %4, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds i32, ptr %32, i64 %34
  store volatile i32 0, ptr %35, align 4
  br label %36

36:                                               ; preds = %31
  %37 = load i32, ptr %4, align 4
  %38 = add nsw i32 %37, 1
  store i32 %38, ptr %4, align 4
  br label %28, !llvm.loop !10

39:                                               ; preds = %28
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
