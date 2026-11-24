import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from paths import (
    LSTM_RAW_DIR,
    LSTM_SPLITS_DIR,
    LSTM_TRAIN_NPZ,
    LSTM_VAL_NPZ,
    LSTM_TEST_NPZ,
)


# ============================================================
# CONFIGURACION
# ============================================================

SEQUENCE_LENGTH = 30
STATIC_FEATURES = 227
NUM_FEATURES = STATIC_FEATURES * 2  # 454 (227 estaticas + 227 velocidad)

# Indices de presencia dentro del vector estatico:
# [223]: pose, [224]: cara, [225]: mano izquierda, [226]: mano derecha
LEFT_HAND_PRESENCE_IDX = 225
RIGHT_HAND_PRESENCE_IDX = 226
MIN_VALID_HAND_RATIO = 0.35

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Directorio de salida
Path(LSTM_SPLITS_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA Y VALIDACION DE SECUENCIAS
# ============================================================

def load_sequence(sequence_dir):
    """
    Carga y valida una secuencia completa de 30 frames x 454 caracteristicas.

    Retorna:
        (sequence, session_id) si es valida.
        (None, None) si esta incompleta, corrupta o sin manos detectadas.
    """
    session_id = "unknown"
    metadata_path = sequence_dir / "metadata.json"

    # 1. Validar metadata si existe
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, None

        if not metadata.get("complete", False):
            return None, None

        meta_features = metadata.get("num_features")
        if meta_features is not None and meta_features != NUM_FEATURES:
            return None, None

        meta_len = metadata.get("sequence_length")
        if meta_len is not None and meta_len != SEQUENCE_LENGTH:
            return None, None

        session_id = str(metadata.get("session", "unknown")).strip() or "unknown"

    # 2. Cargar estrictamente los frames 0.npy .. 29.npy
    frames = []

    for frame_num in range(SEQUENCE_LENGTH):
        frame_path = sequence_dir / f"{frame_num}.npy"

        if not frame_path.is_file():
            return None, None

        try:
            keypoints = np.load(frame_path, allow_pickle=False)
        except (OSError, ValueError):
            return None, None

        if keypoints.shape != (NUM_FEATURES,):
            return None, None

        if not np.all(np.isfinite(keypoints)):
            return None, None

        frames.append(keypoints.astype(np.float32))

    sequence = np.stack(frames, axis=0)

    if sequence.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
        return None, None

    # 3. Verificar presencia minima de manos en la secuencia
    has_left = sequence[:, LEFT_HAND_PRESENCE_IDX] > 0.5
    has_right = sequence[:, RIGHT_HAND_PRESENCE_IDX] > 0.5
    hand_ratio = float(np.mean(has_left | has_right))

    if hand_ratio < MIN_VALID_HAND_RATIO:
        return None, None

    return sequence, session_id


# ============================================================
# SEPARACION SIN FUGA DE DATOS POR CLASE
# ============================================================

def split_class_sequences(sequences, sessions, rng):
    """
    Divide las secuencias de una clase en Train, Val y Test:
      - Si existen >= 3 sesiones distintas: separa por sesiones completas
        (evita que una misma sesion aparezca en train y test).
      - Si existen < 3 sesiones: separa en bloques cronologicos contiguos.
    """
    total = len(sequences)

    if total < 3:
        raise ValueError(
            "Cada clase necesita al menos 3 secuencias validas "
            "para generar Train, Val y Test."
        )

    n_test_target = max(1, int(np.ceil(total * TEST_SIZE)))
    n_val_target = max(1, int(np.ceil(total * VAL_SIZE)))

    # Agrupar manteniendo el orden de aparicion de las sesiones
    session_groups = defaultdict(list)
    for seq, sess in zip(sequences, sessions):
        session_groups[sess].append((seq, sess))

    unique_sessions = list(session_groups.keys())

    # Estrategia 1: Separacion por sesiones completas (>= 3 sesiones distintas)
    if len(unique_sessions) >= 3:
        shuffled_sessions = unique_sessions.copy()
        rng.shuffle(shuffled_sessions)

        test_sessions = []
        val_sessions = []
        remaining = shuffled_sessions.copy()

        # Asignar sesiones a Test dejando al menos 2 sesiones para Val y Train
        test_count = 0
        while remaining and len(remaining) > 2 and (
            not test_sessions or test_count < n_test_target
        ):
            sess = remaining.pop()
            test_sessions.append(sess)
            test_count += len(session_groups[sess])

        # Asignar sesiones a Val dejando al menos 1 sesion para Train
        val_count = 0
        while remaining and len(remaining) > 1 and (
            not val_sessions or val_count < n_val_target
        ):
            sess = remaining.pop()
            val_sessions.append(sess)
            val_count += len(session_groups[sess])

        train_sessions = remaining

        train_items = [item for s in train_sessions for item in session_groups[s]]
        val_items = [item for s in val_sessions for item in session_groups[s]]
        test_items = [item for s in test_sessions for item in session_groups[s]]

        return train_items, val_items, test_items, "por_sesion"

    # Estrategia 2: Bloques cronologicos contiguos DENTRO de cada sesion (< 3 sesiones)
    train_items, val_items, test_items = [], [], []

    for sess, items in session_groups.items():
        n = len(items)
        if n < 3:
            # Si una sesion auxiliar tiene menos de 3 videos, va a Train
            train_items.extend(items)
            continue

        n_test_s = max(1, int(np.ceil(n * TEST_SIZE)))
        n_val_s = max(1, int(np.ceil(n * VAL_SIZE)))
        n_train_s = n - n_val_s - n_test_s

        if n_train_s < 1:
            train_items.extend(items)
            continue

        train_items.extend(items[:n_train_s])
        val_items.extend(items[n_train_s:n_train_s + n_val_s])
        test_items.extend(items[n_train_s + n_val_s:])

    if not train_items or not val_items or not test_items:
        raise ValueError(
            "No hay suficientes secuencias por sesion para generar Train, Val y Test."
        )

    return train_items, val_items, test_items, "cronologico_por_sesion"


# ============================================================
# PROCESAMIENTO DEL DATASET
# ============================================================

def create_splits():
    raw_dir = Path(LSTM_RAW_DIR)

    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {raw_dir}")

    rng = np.random.default_rng(SEED)

    action_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir()
    ])

    if len(action_dirs) < 2:
        raise RuntimeError("Se necesitan al menos dos carpetas de clases.")

    X_train, Y_train, S_train = [], [], []
    X_val, Y_val, S_val = [], [], []
    X_test, Y_test, S_test = [], [], []

    active_actions = []
    total_valid = 0
    total_invalid = 0
    unique_session_ids = set()

    for action_dir in action_dirs:
        action = action_dir.name

        seq_folders = sorted(
            [
                p for p in action_dir.iterdir()
                if p.is_dir() and p.name.isdigit()
            ],
            key=lambda p: int(p.name),
        )

        class_sequences = []
        class_sessions = []
        invalid_count = 0

        print(f"\nProcesando clase: {action}")

        for seq_folder in seq_folders:
            sequence, session_id = load_sequence(seq_folder)

            if sequence is None:
                invalid_count += 1
                print(f"  [DESCARTADA] {action}/{seq_folder.name}")
                continue

            class_sequences.append(sequence)
            class_sessions.append(session_id)
            if session_id != "unknown":
                unique_session_ids.add(session_id)

        total_invalid += invalid_count

        if len(class_sequences) == 0:
            print(
                f"  [OMITIDA] La clase '{action}' no tiene secuencias validas "
                f"(descartadas: {invalid_count})."
            )
            continue

        if len(class_sequences) < 3:
            raise RuntimeError(
                f"La clase '{action}' tiene solamente {len(class_sequences)} secuencias validas. "
                "Se necesitan al menos 3 para dividir en Train, Val y Test."
            )

        label_idx = len(active_actions)
        active_actions.append(action)

        train_items, val_items, test_items, split_mode = split_class_sequences(
            class_sequences,
            class_sessions,
            rng,
        )

        for seq, sess in train_items:
            X_train.append(seq)
            Y_train.append(label_idx)
            S_train.append(sess)

        for seq, sess in val_items:
            X_val.append(seq)
            Y_val.append(label_idx)
            S_val.append(sess)

        for seq, sess in test_items:
            X_test.append(seq)
            Y_test.append(label_idx)
            S_test.append(sess)

        total_valid += len(class_sequences)

        print(
            f"  Validas: {len(class_sequences)} | Descartadas: {invalid_count} | "
            f"Modo: {split_mode}"
        )
        print(
            f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}"
        )

    if len(active_actions) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos clases con secuencias validas para entrenar."
        )

    # ========================================================
    # CONVERSION A NUMPY
    # ========================================================

    X_train_arr = np.asarray(X_train, dtype=np.float32)
    Y_train_arr = np.asarray(Y_train, dtype=np.int32)
    S_train_arr = np.asarray(S_train)

    X_val_arr = np.asarray(X_val, dtype=np.float32)
    Y_val_arr = np.asarray(Y_val, dtype=np.int32)
    S_val_arr = np.asarray(S_val)

    X_test_arr = np.asarray(X_test, dtype=np.float32)
    Y_test_arr = np.asarray(Y_test, dtype=np.int32)
    S_test_arr = np.asarray(S_test)

    labels_arr = np.asarray(active_actions)

    # ========================================================
    # GUARDAR PARTICIONES
    # ========================================================

    np.savez_compressed(
        LSTM_TRAIN_NPZ,
        x=X_train_arr,
        y=Y_train_arr,
        labels=labels_arr,
        sessions=S_train_arr,
    )

    np.savez_compressed(
        LSTM_VAL_NPZ,
        x=X_val_arr,
        y=Y_val_arr,
        labels=labels_arr,
        sessions=S_val_arr,
    )

    np.savez_compressed(
        LSTM_TEST_NPZ,
        x=X_test_arr,
        y=Y_test_arr,
        labels=labels_arr,
        sessions=S_test_arr,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n" + "=" * 55)
    print("RESUMEN DE LA PARTICION")
    print("=" * 55)

    print(f"Clases activas ({len(active_actions)}): {active_actions}")
    print(f"Secuencias validas:     {total_valid}")
    print(f"Secuencias descartadas: {total_invalid}")
    if len(unique_session_ids) < 2:
        print(
            "\n[ADVERTENCIA] Las secuencias disponibles pertenecen a una sola "
            "sesion o no tienen identificador. La precision del conjunto "
            "Hold-out no demuestra generalizacion entre sesiones distintas."
        )

    print("\nDIMENSIONES:")
    print(f"Train: X={X_train_arr.shape}, y={Y_train_arr.shape}")
    print(f"Val:   X={X_val_arr.shape}, y={Y_val_arr.shape}")
    print(f"Test:  X={X_test_arr.shape}, y={Y_test_arr.shape}")

    print("\nARCHIVOS GUARDADOS:")
    print(f"- {LSTM_TRAIN_NPZ}")
    print(f"- {LSTM_VAL_NPZ}")
    print(f"- {LSTM_TEST_NPZ}")


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    create_splits()