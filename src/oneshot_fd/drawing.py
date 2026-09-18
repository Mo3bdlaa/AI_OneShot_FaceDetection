"""Live overlay rendering: face boxes, body brackets, labels and the HUD."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DrawConfig, RecognitionConfig
from .tracking import Track
from .utils import color_for_label

FONT = cv2.FONT_HERSHEY_SIMPLEX
UNKNOWN_COLOR = (130, 130, 130)


def _text_color_for(background: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Black on light boxes, white on dark ones, so labels stay readable."""
    b, g, r = background
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def draw_label(frame: np.ndarray, text: str, anchor: Tuple[int, int],
               color: Tuple[int, int, int], font_scale: float = 0.55,
               thickness: int = 1, above: bool = True) -> None:
    """Draw a filled caption pinned to ``anchor``, kept inside the frame."""
    height, width = frame.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = 5
    box_w = text_w + pad * 2
    box_h = text_h + baseline + pad

    x = max(0, min(anchor[0], width - box_w))
    y = anchor[1] - box_h if above else anchor[1]
    if y < 0:
        y = anchor[1]                      # no room above, flip below
    y = max(0, min(y, height - box_h))

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, cv2.FILLED)
    cv2.putText(frame, text, (x + pad, y + box_h - baseline - 2), FONT,
                font_scale, _text_color_for(color), thickness, cv2.LINE_AA)


def draw_corner_box(frame: np.ndarray, box: Sequence[int], color: Tuple[int, int, int],
                    thickness: int = 2, corner_ratio: float = 0.18) -> None:
    """Corner brackets instead of a full rectangle.

    Used for the body so it reads as a softer, secondary highlight and does not
    fight with the face box for attention.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    corner_w = max(8, int((x2 - x1) * corner_ratio))
    corner_h = max(8, int((y2 - y1) * corner_ratio))

    for cx, cy, dx, dy in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * corner_w, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_h), color, thickness, cv2.LINE_AA)


def blur_region(frame: np.ndarray, box: Sequence[int]) -> None:
    """Pixelate a region in place - used to protect unknown bystanders."""
    x1, y1, x2, y2 = (int(v) for v in box)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    height, width = region.shape[:2]
    small = cv2.resize(region, (max(1, width // 12), max(1, height // 12)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


class Renderer:
    """Turns tracks into the annotated frame the user actually sees."""

    def __init__(self, draw: Optional[DrawConfig] = None,
                 recognition: Optional[RecognitionConfig] = None) -> None:
        self.config = draw or DrawConfig()
        self.recognition = recognition or RecognitionConfig()

    def color_for(self, track: Track) -> Tuple[int, int, int]:
        if track.label == self.recognition.unknown_label:
            return UNKNOWN_COLOR
        return color_for_label(track.label)

    def render(self, frame: np.ndarray, tracks: List[Track], fps: Optional[float] = None,
               source_name: str = "", frame_index: int = 0) -> np.ndarray:
        """Draw every track onto a copy of the frame and return it."""
        canvas = frame.copy()
        config = self.config

        # Farthest first, so the nearest person's label ends up on top.
        ordered = sorted(tracks, key=lambda t: (t.box[3] - t.box[1]) * (t.box[2] - t.box[0]))

        for track in ordered:
            color = self.color_for(track)
            is_unknown = track.label == self.recognition.unknown_label

            if is_unknown and config.blur_unknown:
                blur_region(canvas, track.box)

            if config.show_body and track.body_box is not None:
                draw_corner_box(canvas, track.body_box, color,
                                max(1, config.box_thickness))

            if config.show_face:
                x1, y1, x2, y2 = (int(v) for v in track.box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, config.box_thickness,
                              cv2.LINE_AA)

            if config.show_landmarks and track.landmarks is not None:
                for point in np.asarray(track.landmarks).reshape(-1, 2):
                    cv2.circle(canvas, (int(point[0]), int(point[1])), 2, color, -1, cv2.LINE_AA)

            label = track.label
            if config.show_score and not is_unknown:
                label = f"{label} {track.label_score:.2f}"
            elif config.show_score and is_unknown and track.label_score > 0:
                label = f"{label} ({track.label_score:.2f})"
            label = f"#{track.track_id} {label}"
            if config.show_attributes:
                years = track.estimated_age
                attributes = " ".join(
                    part for part in (track.estimated_gender, f"~{years}" if years else None)
                    if part
                )
                if attributes:
                    label = f"{label} | {attributes}"

            draw_label(canvas, label, (int(track.box[0]), int(track.box[1]) - 2),
                       color, config.font_scale, 1, above=True)

        if config.show_fps or config.show_roster:
            self._draw_hud(canvas, ordered, fps, source_name, frame_index)
        return canvas

    def _draw_hud(self, canvas: np.ndarray, tracks: List[Track], fps: Optional[float],
                  source_name: str, frame_index: int) -> None:
        """Translucent status strip: speed, source and who is on screen."""
        lines: List[Tuple[str, Tuple[int, int, int]]] = []

        if self.config.show_fps:
            status = f"{fps:5.1f} FPS" if fps else "-- FPS"
            head = f"{status} | frame {frame_index}"
            if source_name:
                head += f" | {source_name}"
            lines.append((head, (255, 255, 255)))

        if self.config.show_roster:
            known = [t for t in tracks if t.label != self.recognition.unknown_label]
            unknown_count = len(tracks) - len(known)
            seen = sorted({t.label for t in known})
            summary = f"On screen: {len(tracks)}"
            if unknown_count:
                summary += f" ({unknown_count} unknown)"
            lines.append((summary, (255, 255, 255)))
            for name in seen[:6]:
                lines.append((f"  {name}", color_for_label(name)))
            if len(seen) > 6:
                lines.append((f"  +{len(seen) - 6} more", (200, 200, 200)))

        if not lines:
            return

        pad = 8
        line_height = 20
        width = max(cv2.getTextSize(text, FONT, 0.5, 1)[0][0] for text, _ in lines) + pad * 2
        height = line_height * len(lines) + pad

        panel = canvas[0:height, 0:width]
        if panel.size:
            canvas[0:height, 0:width] = cv2.addWeighted(
                panel, 0.35, np.zeros_like(panel), 0.65, 0
            )

        y = pad + 12
        for text, color in lines:
            cv2.putText(canvas, text, (pad, y), FONT, 0.5, color, 1, cv2.LINE_AA)
            y += line_height
