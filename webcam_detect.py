from pathlib import Path
from ultralytics import YOLO
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่พบกล้องเว็บแคม กรุณาตรวจสอบการเชื่อมต่อ")
    raise SystemExit

print("กด Q เพื่อออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    results = model.predict(frame, conf=0.25, verbose=False)
    cv2.imshow("YOLO Key Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
