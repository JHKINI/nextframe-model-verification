from pathlib import Path

images_dir = Path(r"C:\Users\AISW_203_115\Desktop\새 학습 차번호판탐지 yolo학습\test\images")
labels_dir = Path(r"C:\Users\AISW_203_115\Desktop\새 학습 차번호판탐지 yolo학습\test\labels")

# 현재 존재하는 이미지 이름
image_names = {
    f.stem
    for f in images_dir.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
}

# 이미지가 없는 라벨 삭제
deleted = 0

for label in labels_dir.glob("*.txt"):
    if label.stem not in image_names:
        label.unlink()
        deleted += 1

print(f"삭제된 라벨: {deleted}개")
print(f"현재 이미지: {len(image_names)}개")
print(f"현재 라벨: {len(list(labels_dir.glob('*.txt')))}개")