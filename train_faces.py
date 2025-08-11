import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from facenet_pytorch import MTCNN,InceptionResnetV1
import torch,joblib

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector=MTCNN(image_size=160,margin=20,device=device)
embedder=InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')  # Convert BEFORE passing to detector
    except Exception as e:
        print(f"[ERROR] Failed to open image: {img_path}, error: {e}")
        return None

    face = detector(img)  # This returns a tensor or None
    if face is None:
        print(f"[WARNING] No face detected in {img_path}")
        return None

    with torch.no_grad():
        emb = embedder(face.unsqueeze(0).to(device))  # Add batch dimension
    return emb.squeeze().cpu().numpy()

def load_dataset(folder):
    X, y = [], []
    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            emb = get_embedding(img_path)
            if emb is not None:
                X.append(emb)
                y.append(person)
            else:
                print(f"[WARNING] No face detected in {img_path}")
    print(f"[INFO] Loaded {len(X)} samples.")
    return np.array(X), np.array(y)

X,y=load_dataset('faces')
X=Normalizer(norm='l2').fit_transform(X)
model=SVC(kernel='linear',probability=True, class_weight='balanced').fit(X,y)
joblib.dump((model, Normalizer(norm='l2')),'facenet_svm.pkl')
print(f"[INFO] TRAINING COMPLETE. {len(X)} samples used")