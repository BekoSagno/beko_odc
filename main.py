from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import face_recognition
import numpy as np
import cv2
import os
from typing import List

app = FastAPI(title="Face Recognition API")

# --- CORS (pour autoriser un front-end JS, React, etc.) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou restreindre à ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement des visages connus ---
KNOWN_FACES_DIR = "known_faces"
known_encodings: List[np.ndarray] = []
known_names: List[str] = []

print("🔄 Chargement des visages connus...")
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
print(f"✅ {len(known_names)} visages connus chargés.")


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de reconnaissance faciale !"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    Reçoit une image (UploadFile) et renvoie les visages détectés + leurs noms.
    """
    try:
        # Lire l'image uploadée en mémoire
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Conversion en RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Inconnu"
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            results.append({
                "name": name,
                "box": {"top": top, "right": right, "bottom": bottom, "left": left}
            })

        return JSONResponse({"faces": results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    # Lancement local : uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
