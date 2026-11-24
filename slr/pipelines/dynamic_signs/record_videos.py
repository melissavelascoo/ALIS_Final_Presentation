import json
import shutil
import time
from datetime import datetime

import cv2

from paths import LSTM_VIDEOS_DIR


# ============================================================
# 1. CONFIGURACION
# ============================================================

SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 30
COUNTDOWN_SECONDS = 3

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30


# ============================================================
# 2. UTILIDADES DE ARCHIVOS Y VISUALIZACION
# ============================================================

def get_next_sequence(save_dir):
    """Obtiene el siguiente ID numerico disponible sin sobrescribir videos previos."""
    existing = [
        int(folder.name)
        for folder in save_dir.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ]
    return 0 if not existing else max(existing) + 1


def draw_framing_guide(display):
    """Dibuja una guia sutil de encuadre (solo en pantalla, nunca en el video)."""
    height, width = display.shape[:2]
    margin_x = int(width * 0.12)
    margin_top = int(height * 0.08)
    margin_bottom = int(height * 0.96)

    cv2.rectangle(
        display,
        (margin_x, margin_top),
        (width - margin_x, margin_bottom),
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 3. CONFIGURACION DE CAPTURA
# ============================================================

action = input("Ingresa la sena: ").strip().upper()
if not action:
    raise ValueError("Debes ingresar un nombre de sena valido.")

session = input("Ingresa el identificador de la sesion: ").strip()
if not session:
    raise ValueError("Debes ingresar un identificador de sesion valido.")

save_dir = LSTM_VIDEOS_DIR / action
save_dir.mkdir(parents=True, exist_ok=True)

start_sequence = get_next_sequence(save_dir)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara web.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Codec MJPG: conserva nitidez en movimientos rapidos de dedos
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
frame_interval = 1.0 / TARGET_FPS


# ============================================================
# 4. BUCLE PRINCIPAL DE GRABACION
# ============================================================

stop_capture = False
completed_videos = 0
last_saved_dir = None
seq_idx = 0

try:
    while seq_idx < NUM_SEQUENCES and not stop_capture:
        sequence_id = start_sequence + seq_idx
        seq_dir = save_dir / str(sequence_id)
        seq_dir.mkdir(parents=True, exist_ok=True)

        video_path = seq_dir / "video.avi"
        metadata_path = seq_dir / "metadata.json"

        # ----------------------------------------
        # CUENTA REGRESIVA
        # ----------------------------------------
        countdown_start = time.time()
        repeat_previous = False

        while True:
            ret, frame = cap.read()
            if not ret:
                stop_capture = True
                break

            remaining = COUNTDOWN_SECONDS - int(time.time() - countdown_start)
            display = frame.copy()
            draw_framing_guide(display)

            cv2.putText(
                display,
                f"PREPARATE: {max(0, remaining)}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Sena: {action} | Sesion: {session} | Video: {seq_idx + 1}/{NUM_SEQUENCES}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            footer_text = "Q: Salir"
            if last_saved_dir is not None and last_saved_dir.exists():
                footer_text += "  |  R: Repetir video anterior"

            cv2.putText(
                display,
                footer_text,
                (15, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ALIS - Grabacion de Videos", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_capture = True
                break
            elif (
                key == ord("r")
                and last_saved_dir is not None
                and last_saved_dir.exists()
            ):
                shutil.rmtree(seq_dir, ignore_errors=True)
                shutil.rmtree(last_saved_dir, ignore_errors=True)
                last_saved_dir = None
                seq_idx = max(0, seq_idx - 1)
                completed_videos = max(0, completed_videos - 1)
                repeat_previous = True
                print("[INFO] Ultimo video eliminado. Repitiendo secuencia...")
                break

            if remaining <= 0:
                break

        if stop_capture:
            shutil.rmtree(seq_dir, ignore_errors=True)
            break

        if repeat_previous:
            continue

        # ----------------------------------------
        # GRABACION DE 30 FRAMES LIMPIOS
        # ----------------------------------------
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            TARGET_FPS,
            (width, height),
        )

        if not writer.isOpened():
            shutil.rmtree(seq_dir, ignore_errors=True)
            raise RuntimeError(f"No se pudo inicializar VideoWriter en: {video_path}")

        frames_written = 0
        aborted = False

        for frame_num in range(SEQUENCE_LENGTH):
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                stop_capture = True
                aborted = True
                break

            # Escribir el frame ORIGINAL intacto (sin dibujos ni texto)
            writer.write(frame)
            frames_written += 1

            display = frame.copy()
            cv2.circle(display, (width - 28, 28), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                f"GRABANDO | {action} | Video: {seq_idx + 1}/{NUM_SEQUENCES} | Frame: {frame_num + 1}/{SEQUENCE_LENGTH}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ALIS - Grabacion de Videos", display)

            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_interval - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                stop_capture = True
                aborted = True
                break

        writer.release()

        # Verificar que el video se escribio completo y no esta vacio
        if (
            aborted
            or frames_written != SEQUENCE_LENGTH
            or not video_path.is_file()
            or video_path.stat().st_size == 0
        ):
            shutil.rmtree(seq_dir, ignore_errors=True)
            if stop_capture:
                break
            print(f"[AVISO] Video {sequence_id} incompleto, reintentando...")
            continue

        # Guardar metadata
        metadata = {
            "action": action,
            "session": session,
            "sequence_id": sequence_id,
            "sequence_length": SEQUENCE_LENGTH,
            "fps": TARGET_FPS,
            "resolution": [width, height],
            "capture_date": datetime.now().isoformat(timespec="seconds"),
            "complete": True,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        last_saved_dir = seq_dir
        completed_videos += 1
        print(f"Video {sequence_id} guardado correctamente ({video_path}).")
        seq_idx += 1

finally:
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 5. RESUMEN FINAL
# ============================================================

print("\n========== GRABACION FINALIZADA ==========")
print(f"Sena: {action}")
print(f"Sesion: {session}")
print(f"Videos validos guardados: {completed_videos}")
print(f"Frames por video: {SEQUENCE_LENGTH} ({SEQUENCE_LENGTH / TARGET_FPS:.1f} s)")
print(f"Directorio: {save_dir}")