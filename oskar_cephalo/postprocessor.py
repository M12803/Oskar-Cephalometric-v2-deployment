"""Postprocessing, coordinate extraction, and visualization for Oskar Cephalo."""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# 29 Cephalometric Landmarks Reference
LANDMARKS_29 = [
    {"index": 0, "symbol": "A", "name": "A-point", "desc": "Subspinale - deepest anterior point of maxilla"},
    {"index": 1, "symbol": "ANS", "name": "Anterior Nasal Spine", "desc": "Anterior tip of sharp bony process of maxilla"},
    {"index": 2, "symbol": "B", "name": "B-point", "desc": "Supramentale - deepest anterior point of mandible"},
    {"index": 3, "symbol": "Me", "name": "Menton", "desc": "Most inferior point on mandibular symphysis"},
    {"index": 4, "symbol": "N", "name": "Nasion", "desc": "Most anterior point of nasofrontal suture"},
    {"index": 5, "symbol": "Or", "name": "Orbitale", "desc": "Lowest point on inferior margin of orbit"},
    {"index": 6, "symbol": "Pog", "name": "Pogonion", "desc": "Most anterior point of bony chin"},
    {"index": 7, "symbol": "PNS", "name": "Posterior Nasal Spine", "desc": "Posterior spine of palatine bone"},
    {"index": 8, "symbol": "Pn", "name": "Pronasale", "desc": "Most prominent point of nose tip"},
    {"index": 9, "symbol": "R", "name": "Ramus", "desc": "Posterior border of mandibular ramus"},
    {"index": 10, "symbol": "S", "name": "Sella", "desc": "Geometric center of pituitary fossa"},
    {"index": 11, "symbol": "Ar", "name": "Articulare", "desc": "Intersection of condyle posterior border and cranial base"},
    {"index": 12, "symbol": "Co", "name": "Condylion", "desc": "Most superior point of mandibular condyle"},
    {"index": 13, "symbol": "Gn", "name": "Gnathion", "desc": "Most anteroinferior point of bony chin"},
    {"index": 14, "symbol": "Go", "name": "Gonion", "desc": "Most posteroinferior point on angle of mandible"},
    {"index": 15, "symbol": "Po", "name": "Porion", "desc": "Superior point on margin of external acoustic meatus"},
    {"index": 16, "symbol": "LPM", "name": "Lower 2nd PM Cusp Tip", "desc": "Cusp tip of mandibular 2nd premolar"},
    {"index": 17, "symbol": "LIT", "name": "Lower Incisor Tip", "desc": "Incisal edge of mandibular central incisor"},
    {"index": 18, "symbol": "LMT", "name": "Lower Molar Cusp Tip", "desc": "Mesiobuccal cusp tip of mandibular 1st molar"},
    {"index": 19, "symbol": "UPM", "name": "Upper 2nd PM Cusp Tip", "desc": "Cusp tip of maxillary 2nd premolar"},
    {"index": 20, "symbol": "UIA", "name": "Upper Incisor Apex", "desc": "Root apex of maxillary central incisor"},
    {"index": 21, "symbol": "UIT", "name": "Upper Incisor Tip", "desc": "Incisal edge of maxillary central incisor"},
    {"index": 22, "symbol": "UMT", "name": "Upper Molar Cusp Tip", "desc": "Mesiobuccal cusp tip of maxillary 1st molar"},
    {"index": 23, "symbol": "LIA", "name": "Lower Incisor Apex", "desc": "Root apex of mandibular central incisor"},
    {"index": 24, "symbol": "Li", "name": "Labrale inferius", "desc": "Most prominent point of lower lip"},
    {"index": 25, "symbol": "Ls", "name": "Labrale superius", "desc": "Most prominent point of upper lip"},
    {"index": 26, "symbol": "N'", "name": "Soft Tissue Nasion", "desc": "Concavity between forehead and soft tissue nose"},
    {"index": 27, "symbol": "Pog'", "name": "Soft Tissue Pogonion", "desc": "Most prominent point of soft tissue chin"},
    {"index": 28, "symbol": "Sn", "name": "Subnasale", "desc": "Junction of columella and upper lip"},
]

CVM_STAGES_INFO = {
    1: {"stage": 1, "code": "CVM-S1", "phase": "Pre-peak", "desc": "Lower borders of C2-C4 are flat. Mandibular peak growth in ~2 years."},
    2: {"stage": 2, "code": "CVM-S2", "phase": "Pre-peak", "desc": "Concavity present at lower border of C2. Mandibular peak growth in ~1 year."},
    3: {"stage": 3, "code": "CVM-S3", "phase": "Peak", "desc": "Concavities at C2 and C3. Peak mandibular growth occurs within this stage."},
    4: {"stage": 4, "code": "CVM-S4", "phase": "Post-peak", "desc": "Concavities at C2, C3, and C4. Peak growth passed ~1 year earlier."},
    5: {"stage": 5, "code": "CVM-S5", "phase": "Post-peak", "desc": "At least one vertebral body is square. Mandibular growth has substantially ended."},
    6: {"stage": 6, "code": "CVM-S6", "phase": "Completion", "desc": "Vertebral body height exceeds width. Mandibular growth completed."}
}

# Standard cephalometric planes to draw
TRACING_PLANES = [
    (10, 4, "S-N", (255, 69, 0)),        # Sella-Nasion (Cranial base plane)
    (4, 0, "N-A", (30, 144, 255)),       # Nasion-A plane
    (4, 2, "N-B", (50, 205, 50)),        # Nasion-B plane
    (0, 2, "A-B", (255, 215, 0)),        # A-B plane
    (7, 1, "Palatal", (138, 43, 226)),   # PNS-ANS (Palatal plane)
    (14, 3, "Mandibular", (255, 20, 147)),# Go-Me (Mandibular plane)
    (15, 5, "Frankfort", (0, 206, 209)), # Po-Or (Frankfort Horizontal)
    (4, 6, "Facial", (255, 105, 180)),   # N-Pog (Facial plane)
]

COLORS_29 = [
    (255, 59, 48), (255, 149, 0), (255, 204, 0), (52, 199, 89), (0, 199, 190),
    (48, 176, 254), (0, 122, 255), (88, 86, 214), (175, 82, 222), (255, 45, 85),
    (162, 132, 94), (76, 217, 100), (255, 56, 107), (0, 222, 255), (255, 128, 0),
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216), (245, 130, 49),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (188, 246, 12), (250, 190, 212),
    (0, 128, 128), (230, 190, 255), (154, 99, 36), (255, 250, 200)
]


class CephaloPostprocessor:
    """Extracts coordinates and generates visual cephalometric overlays."""

    HEATMAP_WIDTH = 192
    HEATMAP_HEIGHT = 216

    @classmethod
    def extract_coordinates(
        cls,
        heatmaps: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> List[Dict[str, Any]]:
        """
        Extract coordinates from heatmaps (1, 29, 216, 192) back into original image space.
        """
        scale_x = orig_w / float(cls.HEATMAP_WIDTH)
        scale_y = orig_h / float(cls.HEATMAP_HEIGHT)

        results = []
        for i, info in enumerate(LANDMARKS_29):
            hm = heatmaps[0, i]
            max_idx = int(np.argmax(hm))
            confidence = float(np.max(hm))

            y_hm = max_idx // cls.HEATMAP_WIDTH
            x_hm = max_idx % cls.HEATMAP_WIDTH

            px = float(x_hm * scale_x)
            py = float(y_hm * scale_y)

            results.append({
                "index": i + 1,
                "symbol": info["symbol"],
                "name": info["name"],
                "x": round(px, 1),
                "y": round(py, 1),
                "confidence": round(confidence, 4),
                "description": info["desc"]
            })

        return results

    @classmethod
    def parse_cvm(cls, cvm_logits: np.ndarray) -> Dict[str, Any]:
        """Convert CVM logits (1, 6) into clinical stage details."""
        stage_idx = int(np.argmax(cvm_logits[0])) + 1
        probabilities = np.exp(cvm_logits[0] - np.max(cvm_logits[0]))
        probabilities = probabilities / np.sum(probabilities)

        info = CVM_STAGES_INFO.get(stage_idx, CVM_STAGES_INFO[1])
        return {
            "stage": stage_idx,
            "code": info["code"],
            "phase": info["phase"],
            "description": info["desc"],
            "confidence": round(float(probabilities[stage_idx - 1]) * 100, 2)
        }

    @classmethod
    def render_overlay(
        cls,
        canvas: np.ndarray,
        landmarks: List[Dict[str, Any]],
        point_size: int = 6,
        show_labels: bool = True,
        draw_lines: bool = True
    ) -> np.ndarray:
        """Render anatomical landmarks and cephalometric planes onto the image."""
        img = canvas.copy()

        # Draw cephalometric planes
        if draw_lines:
            for p1_idx, p2_idx, name, color in TRACING_PLANES:
                if p1_idx < len(landmarks) and p2_idx < len(landmarks):
                    pt1 = (int(landmarks[p1_idx]["x"]), int(landmarks[p1_idx]["y"]))
                    pt2 = (int(landmarks[p2_idx]["x"]), int(landmarks[p2_idx]["y"]))
                    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw landmark points and labels
        for i, lm in enumerate(landmarks):
            pt = (int(lm["x"]), int(lm["y"]))
            color = COLORS_29[i % len(COLORS_29)]

            # High-contrast circular landmark
            cv2.circle(img, pt, point_size + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pt, point_size, color, -1, cv2.LINE_AA)

            if show_labels:
                label = lm["symbol"]
                font_scale = max(0.4, point_size / 9.0)
                cv2.putText(img, label, (pt[0] + point_size + 3, pt[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, (pt[0] + point_size + 3, pt[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return img
