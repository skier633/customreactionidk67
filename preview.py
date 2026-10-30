"""Preview-only guides. Never draw these on the virtual-camera frame."""

import cv2
import numpy as np


CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def skeleton(frame, hands):
    for hand in hands:
        color = (190, 229, 110) if hand["label"] == "Left" else (235, 170, 146)
        pts = np.rint(hand["pts"]).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for i, point in enumerate(pts):
            cv2.circle(frame, tuple(point), 4 if i in (4, 8, 12, 16, 20) else 2,
                       (245, 245, 245), -1, cv2.LINE_AA)


def draw(frame, hands, fps, pen_enabled, virtual_cam, poses):
    """Draw a compact status strip and a deliberate-gesture hold meter in-place."""
    h, w = frame.shape[:2]
    scale = min(w / 960, h / 540)
    if scale < 0.35:
        return
    unit = lambda n: max(1, int(n * scale))
    margin, panel_h = unit(16), unit(66)
    x1, y1, x2, y2 = margin, margin, w - margin, margin + panel_h
    panel = frame[y1:y2, x1:x2]
    panel[:] = (panel.astype(np.float32) * 0.22 + np.array([25, 23, 20]) * 0.78).astype(np.uint8)

    def text(value, x, y, size=0.43, color=(205, 205, 205)):
        cv2.putText(frame, value, (unit(x), unit(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    size * scale, color, max(1, unit(1)), cv2.LINE_AA)

    text("GESTURE / CAM", 30, 41, 0.55, (210, 240, 164))
    source = "CALL OUTPUT ON" if virtual_cam else "LOCAL PREVIEW"
    text(f"{source}   |   {len(hands)} HAND{'S' if len(hands) != 1 else ''}   |   {fps:.0f} FPS", 260, 41)
    text(f"P  PEN {'ON' if pen_enabled else 'OFF'}    C  CLEAR    D  LANDMARKS    H  HIDE    Q  QUIT", 30, 65, 0.36)

    # One meter follows the strongest candidate; never imply an effect fired if it is unconfigured.
    pose = max(poses, key=lambda p: p["progress"], default=None)
    if pose:
        label = pose["name"].replace("_x2", " / both hands").replace("_", " ").upper()
        message = label if pose.get("motion") else label + ("  /  RELEASE TO REPEAT" if pose["done"] else "  /  HOLD")
        progress = float(np.clip(pose["progress"], 0, 1))
    else:
        message = "HOLD A GESTURE TO TRIGGER" if hands else "BRING YOUR HAND INTO VIEW"
        progress = 0
    width = min(unit(430), w - 2 * margin)
    left, bottom = (w - width) // 2, h - unit(26)
    layer = frame[bottom - unit(40):bottom + unit(5), left:left + width]
    layer[:] = (layer.astype(np.float32) * .22 + np.array([25, 23, 20]) * .78).astype(np.uint8)
    cv2.putText(frame, message, (left + unit(14), bottom - unit(14)), cv2.FONT_HERSHEY_SIMPLEX,
                .37 * scale, (235, 235, 235), max(1, unit(1)), cv2.LINE_AA)
    cv2.line(frame, (left, bottom), (left + width, bottom), (65, 65, 65), unit(3))
    if progress:
        cv2.line(frame, (left, bottom), (left + int(width * progress), bottom), (190, 229, 110), unit(3))
