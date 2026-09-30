from pathlib import Path
import shutil
import csv

import cv2
from ultralytics import YOLO


# =========================================================
# 1. 경로
# =========================================================

MODEL_PATH = Path(
    r"C:\Users\AISW_203_115\Desktop\완료된 프젝 결과물\번호판탐지\nextframe\best.pt"
)

IMAGE_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\새 학습 차번호판탐지 yolo학습\test\images"
)

LABEL_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\새 학습 차번호판탐지 yolo학습\test\labels"
)

OUTPUT_DIR = Path(
    r"C:\Users\AISW_203_115\Desktop\번호판_오류분석"
)

# validation 결과의 F1 최적 confidence
CONF_THRESHOLD = 0.409

# IoU 기준
IOU_THRESHOLD = 0.50


# =========================================================
# 2. 폴더 생성
# =========================================================

FN_DIR = OUTPUT_DIR / "01_FN_놓친번호판"
FP_DIR = OUTPUT_DIR / "02_FP_잘못검출"
LOC_DIR = OUTPUT_DIR / "03_위치정밀도_낮은검출"

for folder in [FN_DIR, FP_DIR, LOC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. IoU 계산
# =========================================================

def calculate_iou(box1, box2):
    """
    box = [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# =========================================================
# 4. YOLO 라벨 읽기
# =========================================================

def read_labels(label_path, image_width, image_height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, cx, cy, w, h = map(float, values)

            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height

            boxes.append([
                x1, y1, x2, y2
            ])

    return boxes


# =========================================================
# 5. 모델
# =========================================================

print("모델 로딩 중...")

model = YOLO(str(MODEL_PATH))

print("모델:", model.names)
print("테스트 시작")
print()


# =========================================================
# 6. CSV
# =========================================================

csv_path = OUTPUT_DIR / "error_analysis.csv"

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
)

writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "gt_count",
    "pred_count",
    "matched_count",
    "FN",
    "FP",
    "best_iou",
    "low_localization"
])


# =========================================================
# 7. 통계
# =========================================================

total_images = 0
total_gt = 0
total_pred = 0

total_fn = 0
total_fp = 0

total_matched = 0
total_low_localization = 0


# =========================================================
# 8. 이미지 처리
# =========================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print(f"이미지 수: {len(image_files)}")
print()


for index, image_path in enumerate(image_files, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print("읽기 실패:", image_path.name)
        continue

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # GT
    # -----------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_labels(
        label_path,
        width,
        height
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    pred_boxes = []

    if result.boxes is not None:

        for box, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            pred_boxes.append({
                "box": box.tolist(),
                "conf": float(conf)
            })

    # -----------------------------------------------------
    # Matching
    # -----------------------------------------------------

    matched_gt = set()
    matched_pred = set()

    matches = []

    for pred_idx, pred in enumerate(pred_boxes):

        best_iou = 0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_boxes):

            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(
                pred["box"],
                gt
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

            matches.append(
                (
                    pred_idx,
                    best_gt_idx,
                    best_iou
                )
            )

    # -----------------------------------------------------
    # FN / FP
    # -----------------------------------------------------

    fn_count = len(gt_boxes) - len(matched_gt)
    fp_count = len(pred_boxes) - len(matched_pred)

    # IoU 0.5 이상이지만 0.95 미만인 검출
    low_loc_count = sum(
        1
        for _, _, iou in matches
        if iou < 0.95
    )

    # -----------------------------------------------------
    # 이미지 표시
    # -----------------------------------------------------

    annotated = image.copy()

    # GT = 초록색
    for gt in gt_boxes:

        x1, y1, x2, y2 = map(int, gt)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # Prediction = 파란색
    for pred_idx, pred in enumerate(pred_boxes):

        x1, y1, x2, y2 = map(
            int,
            pred["box"]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated,
            f"{pred['conf']:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # -----------------------------------------------------
    # 오류가 있는 이미지 저장
    # -----------------------------------------------------

    if fn_count > 0:

        output_path = FN_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated
        )

    if fp_count > 0:

        output_path = FP_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated
        )

    if low_loc_count > 0:

        output_path = LOC_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated
        )

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    best_iou = max(
        [x[2] for x in matches],
        default=0
    )

    writer.writerow([
        image_path.name,
        len(gt_boxes),
        len(pred_boxes),
        len(matches),
        fn_count,
        fp_count,
        round(best_iou, 4),
        low_loc_count
    ])

    # -----------------------------------------------------
    # 통계
    # -----------------------------------------------------

    total_images += 1
    total_gt += len(gt_boxes)
    total_pred += len(pred_boxes)

    total_fn += fn_count
    total_fp += fp_count

    total_matched += len(matches)
    total_low_localization += low_loc_count

    if index % 50 == 0:
        print(f"{index}/{len(image_files)} 완료")


csv_file.close()


# =========================================================
# 9. 결과
# =========================================================

print()
print("=" * 60)
print("오류 분석 완료")
print("=" * 60)

print(f"이미지: {total_images}")
print(f"GT 객체: {total_gt}")
print(f"예측 객체: {total_pred}")
print(f"매칭: {total_matched}")
print(f"FN (놓친 번호판): {total_fn}")
print(f"FP (잘못 검출): {total_fp}")
print(f"위치 정밀도 낮은 검출(IoU < 0.95): {total_low_localization}")

print()
print("결과 폴더:")
print(OUTPUT_DIR)

print()
print("CSV:")
print(csv_path)