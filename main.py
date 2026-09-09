import time

import cv2

from hand_tracking import (
    INDEX_TIP,
    THUMB_TIP,
    create_hand_landmarker,
    frame_to_mp_image,
    get_left_right_hands,
)
from geometry import render_portal, portal_width, ClosingGestureDetector
from filters import FILTROS


def main():
    landmarker = create_hand_landmarker(
        num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa el indice de camara o los permisos."
        )

    filtro_index = 0
    closing_detector = ClosingGestureDetector()

    # El modo VIDEO del HandLandmarker requiere timestamps (ms) estrictamente
    # crecientes, uno por frame, en lugar del antiguo hands.process(rgb).
    start_time = time.time()
    last_timestamp_ms = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = frame_to_mp_image(rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand = get_left_right_hands(result)

        if left_hand is not None and right_hand is not None:
            lm_left = left_hand
            lm_right = right_hand

            p1 = (lm_left[INDEX_TIP].x * w, lm_left[INDEX_TIP].y * h)
            p2 = (lm_left[THUMB_TIP].x * w, lm_left[THUMB_TIP].y * h)
            p3 = (lm_right[INDEX_TIP].x * w, lm_right[INDEX_TIP].y * h)
            p4 = (lm_right[THUMB_TIP].x * w, lm_right[THUMB_TIP].y * h)

            width = portal_width(p1, p2, p3, p4)

            if closing_detector.update(width, w):
                filtro_index = (filtro_index + 1) % len(FILTROS)

            frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filtro_index])

        cv2.imshow(" ", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
