"""Geometry helpers for OCR detection regions.

Coordinate convention used throughout this project
--------------------------------------------------
* Origin is the **top-left** pixel of the image as loaded by OpenCV.
* ``x`` increases to the right, ``y`` increases downward.
* Units are pixels in the coordinate space of the image that was *fed to
  the detector*. If preprocessing rescaled the image, coordinates are in
  the rescaled space and the scale factor is recorded alongside them.
* A polygon is a list of ``[x, y]`` vertices in the order the detector
  returned them (PP-OCRv5 detection returns 4 vertices, clockwise from
  the top-left of the rotated box). Polygons are **not** reordered.
* An axis-aligned bounding box (AABB) is ``[x_min, y_min, x_max, y_max]``
  with ``x_max``/``y_max`` exclusive-ish in the sense of pixel extents;
  they are stored as returned, not rounded outward.

These are *image* coordinates. They are not projector coordinates and
not page coordinates. Mapping to a projector requires camera-to-page and
page-to-projector calibration, which is a separate future task.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Polygon = List[List[float]]
BBox = Tuple[float, float, float, float]


class GeometryError(ValueError):
    """Raised for malformed or out-of-range geometry."""


def normalise_polygon(raw: Iterable[Sequence[float]]) -> Polygon:
    """Coerce a detector polygon into a plain list of ``[x, y]`` floats.

    Accepts numpy arrays, tuples or lists. The vertex order is preserved
    exactly as given -- callers rely on that to keep the detector's own
    notion of which corner is "first".
    """
    poly: Polygon = []
    for vertex in raw:
        pair = list(vertex)
        if len(pair) != 2:
            raise GeometryError(
                f"Polygon vertex must have exactly 2 coordinates, got {len(pair)}"
            )
        poly.append([float(pair[0]), float(pair[1])])
    if len(poly) < 3:
        raise GeometryError(f"Polygon needs at least 3 vertices, got {len(poly)}")
    return poly


def polygon_to_bbox(polygon: Iterable[Sequence[float]]) -> BBox:
    """Axis-aligned bounding box enclosing ``polygon``."""
    poly = normalise_polygon(polygon)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_to_polygon(bbox: Sequence[float]) -> Polygon:
    """Convert an AABB to a 4-vertex polygon, clockwise from top-left."""
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def _validated_bbox(bbox: Sequence[float]) -> BBox:
    values = list(bbox)
    if len(values) != 4:
        raise GeometryError(f"BBox must have 4 values, got {len(values)}")
    x_min, y_min, x_max, y_max = (float(v) for v in values)
    if x_max < x_min or y_max < y_min:
        raise GeometryError(
            f"Degenerate bbox: max must be >= min, got {(x_min, y_min, x_max, y_max)}"
        )
    return (x_min, y_min, x_max, y_max)


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> BBox:
    """Clip an AABB to the image rectangle ``[0, width] x [0, height]``."""
    if width <= 0 or height <= 0:
        raise GeometryError(f"Image size must be positive, got {width}x{height}")
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return (
        min(max(x_min, 0.0), float(width)),
        min(max(y_min, 0.0), float(height)),
        min(max(x_max, 0.0), float(width)),
        min(max(y_max, 0.0), float(height)),
    )


def clip_polygon(
    polygon: Iterable[Sequence[float]], width: int, height: int
) -> Polygon:
    """Clamp every vertex into the image rectangle.

    This clamps vertices individually; it is not a true polygon-rectangle
    intersection, so a polygon crossing a corner keeps its vertex count.
    """
    if width <= 0 or height <= 0:
        raise GeometryError(f"Image size must be positive, got {width}x{height}")
    return [
        [min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))]
        for x, y in normalise_polygon(polygon)
    ]


def bbox_area(bbox: Sequence[float]) -> float:
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return (x_max - x_min) * (y_max - y_min)


def polygon_area(polygon: Iterable[Sequence[float]]) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    poly = normalise_polygon(polygon)
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def bbox_centre(bbox: Sequence[float]) -> Point:
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two AABBs. Returns 0.0 when disjoint."""
    ax0, ay0, ax1, ay1 = _validated_bbox(a)
    bx0, by0, bx1, by1 = _validated_bbox(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = ix1 - ix0, iy1 - iy0
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def scale_polygon(polygon: Iterable[Sequence[float]], factor: float) -> Polygon:
    """Scale a polygon about the image origin by ``factor``.

    Used to map coordinates back from a preprocessed (upscaled) image
    into the original image space: pass ``1 / upscale``.
    """
    if factor <= 0:
        raise GeometryError(f"Scale factor must be > 0, got {factor}")
    return [[x * factor, y * factor] for x, y in normalise_polygon(polygon)]


def scale_bbox(bbox: Sequence[float], factor: float) -> BBox:
    if factor <= 0:
        raise GeometryError(f"Scale factor must be > 0, got {factor}")
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return (x_min * factor, y_min * factor, x_max * factor, y_max * factor)


def bbox_is_within(bbox: Sequence[float], width: int, height: int) -> bool:
    """Whether an AABB lies fully inside the image rectangle."""
    if width <= 0 or height <= 0:
        raise GeometryError(f"Image size must be positive, got {width}x{height}")
    x_min, y_min, x_max, y_max = _validated_bbox(bbox)
    return x_min >= 0 and y_min >= 0 and x_max <= width and y_max <= height


def horizontal_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap along x as a fraction of the narrower box's width."""
    ax0, _, ax1, _ = _validated_bbox(a)
    bx0, _, bx1, _ = _validated_bbox(b)
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return 0.0
    narrower = min(ax1 - ax0, bx1 - bx0)
    return overlap / narrower if narrower > 0 else 0.0


def vertical_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap along y as a fraction of the shorter box's height.

    Used to group detection regions into reading lines.
    """
    _, ay0, _, ay1 = _validated_bbox(a)
    _, by0, _, by1 = _validated_bbox(b)
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap <= 0:
        return 0.0
    shorter = min(ay1 - ay0, by1 - by0)
    return overlap / shorter if shorter > 0 else 0.0
