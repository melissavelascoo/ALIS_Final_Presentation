import csv
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from paths import MLP_IMAGES_DIR, MLP_RAW_CSV

# CONFIGURACION DE CAPTURA
label = input("Ingresa la letra: ").strip().upper()
session = input("Ingresa la sesion: ").strip()

if not label or not session:
    raise ValueError("La letra y la sesion no pueden estar vacias.")

offset = 20
imgSize = 224

raw_img_folder = MLP_IMAGES_DIR / label
raw_csv_path = MLP_RAW_CSV  # Ya apunta a landmarks_raw.csv

raw_img_folder.mkdir(parents=True, exist_ok=True)
raw_csv_path.parent.mkdir(parents=True, exist_ok=True)

# Crear encabezado si el CSV no existe o esta vacio
if not raw_csv_path.exists() or raw_csv_path.stat().st_size == 0:
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    header = ["label", "session", "image_file", "handedness"] + coord_cols
    with open(raw_csv_path, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.5)
counter = 0

print(f"Carpeta de imagenes: {raw_img_folder}")
print(f"Archivo CSV: {raw_csv_path}")
print("Haz click en la ventana 'Image' y presiona 's' para capturar | 'q' para salir")

while True:
    success, img = cap.read()
    if not success:
        print("ERROR: No se pudo leer fotograma de la camara.")
        break

    imgClean = img.copy()  # Copia limpia sin esqueleto dibujado
    hands, img = detector.findHands(img, draw=True)

    imgWhite = None
    lm_raw = None
    handedness = "Unknown"

    if hands:
        hand = hands[0]
        handedness = hand.get("type", "Unknown")
        x, y, w, h = hand["bbox"]
        H, W, _ = imgClean.shape

        y1, y2 = max(0, y - offset), min(H, y + h + offset)
        x1, x2 = max(0, x - offset), min(W, x + w + offset)
        imgCrop = imgClean[y1:y2, x1:x2]

        if imgCrop.size > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            cropH, cropW = imgCrop.shape[:2]
            aspectRatio = cropH / cropW

            if aspectRatio > 1:
                k = imgSize / cropH
                wCal = min(imgSize, int(k * cropW))
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / cropW
                hCal = min(imgSize, int(k * cropH))
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Adaptar lmList sin importar si cvzone devuelve (21, 3) o (21, 2)
            lm_array = np.array(hand["lmList"], dtype=np.float32)
            if lm_array.shape == (21, 3):
                lm_raw = lm_array.flatten().tolist()
            elif lm_array.shape == (21, 2):
                zeros = np.zeros((21, 1), dtype=np.float32)
                lm_raw = np.hstack([lm_array, zeros]).flatten().tolist()

            cv2.imshow("ImageWhite (Limpia)", imgWhite)

    cv2.putText(
        img,
        f"Letra: {label} | Capturas: {counter}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("s"), ord("S")):
        if imgWhite is None or lm_raw is None:
            print("AVISO: Presionaste 's', pero no se detecto una mano valida en pantalla.")
        else:
            counter += 1
            timestamp = int(time.time() * 1000)
            img_file = raw_img_folder / f"{session}_{handedness}_{timestamp}.jpg"

            ok = cv2.imwrite(str(img_file), imgWhite)
            if not ok:
                print(f"ERROR: cv2.imwrite no pudo guardar en {img_file}")
                counter -= 1
                continue

            with open(raw_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([label, session, img_file.name, handedness] + lm_raw)

            print(f"[{label} - {session}] Guardado #{counter} -> {img_file.name}")

    elif key in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()