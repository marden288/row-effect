import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

# Modelo oficial de Google para el HandLandmarker de la Tasks API.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


def _ensure_model_downloaded() -> str:
    """Descarga el modelo .task la primera vez que se ejecuta el proyecto."""
    if not os.path.exists(_MODEL_PATH):
        print("Descargando modelo hand_landmarker.task (solo la primera vez)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"Modelo guardado en: {_MODEL_PATH}")
    return _MODEL_PATH


def create_hand_landmarker(
    num_hands: int = 2,
    min_detection_confidence: float = 0.6,
    min_tracking_confidence: float = 0.6,
    min_presence_confidence: float = 0.6,
) -> mp_tasks_vision.HandLandmarker:
    """Crea el detector de manos usando la nueva MediaPipe Tasks API.

    Sustituye a mp.solutions.hands.Hands(...), que ya no existe en
    MediaPipe >= 0.10. Se usa el modo VIDEO para poder alimentar el
    detector con timestamps crecientes, frame a frame, tal como se
    hacia antes con hands.process(rgb).
    """
    model_path = _ensure_model_downloaded()
    base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
    options = mp_tasks_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
    )
    return mp_tasks_vision.HandLandmarker.create_from_options(options)


def frame_to_mp_image(rgb_frame: np.ndarray) -> mp.Image:
    """Envuelve un frame RGB (numpy.ndarray) en un mediapipe.Image."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def get_left_right_hands(result):
    """Separa el resultado del HandLandmarker en mano izquierda/derecha.

    Devuelve listas de landmarks (cada una indexable como antes, p.ej.
    lm[INDEX_TIP].x / .y), preservando la misma logica de "espejo" que
    tenia el codigo original: como el frame se voltea para vista selfie,
    la etiqueta que reporta MediaPipe se invierte para que coincida con
    la mano real del usuario.
    """
    left_hand = None
    right_hand = None

    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness in zip(
            result.hand_landmarks, result.handedness
        ):
            raw_label = handedness[0].category_name
            label = "Right" if raw_label == "Left" else "Left"

            if label == "Left":
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

    return left_hand, right_hand


def _dist(lm, i, j, w, h):
    a = np.array([lm[i].x * w, lm[i].y * h])
    b = np.array([lm[j].x * w, lm[j].y * h])
    return np.linalg.norm(a - b)


def get_extended_fingers(hand_landmarks, w, h):
    """hand_landmarks es ahora una lista de NormalizedLandmark (Tasks API),
    en lugar del objeto con atributo .landmark de la API legacy."""
    lm = hand_landmarks

    def is_extended(tip, mcp):
        return _dist(lm, tip, WRIST, w, h) > _dist(lm, mcp, WRIST, w, h) * 1.3

    return {
        "thumb": is_extended(THUMB_TIP, THUMB_MCP),
        "index": is_extended(INDEX_TIP, INDEX_MCP),
        "middle": is_extended(MIDDLE_TIP, MIDDLE_MCP),
        "ring": is_extended(RING_TIP, RING_MCP),
        "pinky": is_extended(PINKY_TIP, PINKY_MCP),
    }
