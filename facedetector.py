import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import joblib

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(image_size=160, margin=20, device=device, post_process=True)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model, in_encoder = joblib.load('facenet_svm.pkl')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from webcam.")
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = detector.detect(img)

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop and prepare face
            face_crop = img.crop((x1, y1, x2, y2))
            aligned_face = detector(Image.fromarray(np.array(face_crop)))

            if aligned_face is not None:
                try:
                    aligned_face = aligned_face.unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = embedder(aligned_face)

                    embedding_np = embedding.squeeze().cpu().numpy().reshape(1, -1)
                    embedding_np = in_encoder.transform(embedding_np)
                    prediction = model.predict(embedding_np)
                    proba = model.predict_proba(embedding_np)
                    confidence = np.max(proba) * 100

                    color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
                    label = f"{prediction[0]} ({confidence:.1f}%)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except Exception as e:
                    print("[ERROR] Face processing failed:", e)
                    continue
            else:
                print("[WARNING] MTCNN failed to align face.")
    else:
        print("[INFO] No face detected.")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
