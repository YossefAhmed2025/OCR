"""Geometry conversion and bounds tests.

These exercise pure functions only -- no model, no GPU, no images. They
validate the *code*, not OCR accuracy.
"""

import pytest

from phonics_ocr import geometry as geo


class TestPolygonNormalisation:
    def test_accepts_nested_lists(self):
        poly = geo.normalise_polygon([[1, 2], [3, 2], [3, 5], [1, 5]])
        assert poly == [[1.0, 2.0], [3.0, 2.0], [3.0, 5.0], [1.0, 5.0]]
        assert all(isinstance(v, float) for point in poly for v in point)

    def test_accepts_tuples(self):
        assert geo.normalise_polygon([(0, 0), (4, 0), (4, 4)]) == [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
        ]

    def test_preserves_vertex_order(self):
        # Vertex order carries the detector's notion of "first corner";
        # reordering it would silently rotate every box.
        given = [[9, 1], [1, 1], [1, 9], [9, 9]]
        assert geo.normalise_polygon(given) == [[float(x), float(y)] for x, y in given]

    def test_rejects_3d_vertex(self):
        with pytest.raises(geo.GeometryError, match="exactly 2 coordinates"):
            geo.normalise_polygon([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_rejects_too_few_vertices(self):
        with pytest.raises(geo.GeometryError, match="at least 3 vertices"):
            geo.normalise_polygon([[0, 0], [1, 1]])


class TestPolygonBBoxRoundTrip:
    def test_polygon_to_bbox(self):
        assert geo.polygon_to_bbox([[10, 20], [50, 22], [48, 60], [8, 58]]) == (
            8.0,
            20.0,
            50.0,
            60.0,
        )

    def test_bbox_to_polygon_is_clockwise_from_top_left(self):
        assert geo.bbox_to_polygon([1, 2, 5, 8]) == [
            [1.0, 2.0],
            [5.0, 2.0],
            [5.0, 8.0],
            [1.0, 8.0],
        ]

    def test_round_trip_preserves_axis_aligned_boxes(self):
        bbox = (3.0, 7.0, 19.0, 25.0)
        assert geo.polygon_to_bbox(geo.bbox_to_polygon(bbox)) == bbox

    def test_rotated_polygon_bbox_is_larger_than_polygon(self):
        diamond = [[10, 0], [20, 10], [10, 20], [0, 10]]
        assert geo.polygon_area(diamond) == pytest.approx(200.0)
        assert geo.bbox_area(geo.polygon_to_bbox(diamond)) == pytest.approx(400.0)

    def test_rejects_wrong_length_bbox(self):
        with pytest.raises(geo.GeometryError, match="4 values"):
            geo.bbox_to_polygon([1, 2, 3])

    def test_rejects_inverted_bbox(self):
        with pytest.raises(geo.GeometryError, match="Degenerate"):
            geo.bbox_to_polygon([10, 10, 5, 20])


class TestClipping:
    def test_clip_bbox_inside_is_unchanged(self):
        assert geo.clip_bbox((10, 10, 90, 90), 100, 100) == (10.0, 10.0, 90.0, 90.0)

    def test_clip_bbox_clamps_overflow(self):
        assert geo.clip_bbox((-20, -5, 150, 130), 100, 100) == (0.0, 0.0, 100.0, 100.0)

    def test_clip_bbox_fully_outside_collapses_to_edge(self):
        assert geo.clip_bbox((200, 200, 300, 300), 100, 100) == (
            100.0,
            100.0,
            100.0,
            100.0,
        )

    def test_clip_polygon_clamps_each_vertex(self):
        clipped = geo.clip_polygon([[-10, 50], [110, 50], [110, 150], [-10, 150]], 100, 100)
        assert clipped == [[0.0, 50.0], [100.0, 50.0], [100.0, 100.0], [0.0, 100.0]]

    def test_clip_polygon_keeps_vertex_count(self):
        poly = [[-5, -5], [105, -5], [105, 105], [-5, 105]]
        assert len(geo.clip_polygon(poly, 100, 100)) == len(poly)

    @pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 10)])
    def test_rejects_non_positive_image_size(self, width, height):
        with pytest.raises(geo.GeometryError, match="must be positive"):
            geo.clip_bbox((0, 0, 1, 1), width, height)


class TestBoundsPredicate:
    def test_inside(self):
        assert geo.bbox_is_within((0, 0, 100, 100), 100, 100)

    @pytest.mark.parametrize(
        "bbox", [(-1, 0, 50, 50), (0, -1, 50, 50), (0, 0, 101, 50), (0, 0, 50, 101)]
    )
    def test_outside(self, bbox):
        assert not geo.bbox_is_within(bbox, 100, 100)


class TestAreaAndCentre:
    def test_bbox_area(self):
        assert geo.bbox_area((0, 0, 10, 4)) == 40.0

    def test_zero_area(self):
        assert geo.bbox_area((5, 5, 5, 12)) == 0.0

    def test_centre(self):
        assert geo.bbox_centre((0, 0, 10, 20)) == (5.0, 10.0)

    def test_shoelace_is_orientation_independent(self):
        clockwise = [[0, 0], [4, 0], [4, 3], [0, 3]]
        counter = list(reversed(clockwise))
        assert geo.polygon_area(clockwise) == geo.polygon_area(counter) == 12.0


class TestIoU:
    def test_identical_boxes(self):
        assert geo.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert geo.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_edge_touching_counts_as_disjoint(self):
        assert geo.bbox_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Intersection 50, union 150.
        assert geo.bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_contained_box(self):
        assert geo.bbox_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(4 / 100)

    def test_is_symmetric(self):
        a, b = (1, 1, 9, 9), (4, 4, 14, 14)
        assert geo.bbox_iou(a, b) == pytest.approx(geo.bbox_iou(b, a))


class TestScaling:
    def test_scale_polygon_down(self):
        assert geo.scale_polygon([[10, 20], [30, 20], [30, 40]], 0.5) == [
            [5.0, 10.0],
            [15.0, 10.0],
            [15.0, 20.0],
        ]

    def test_upscale_then_scale_back_is_identity(self):
        # This is exactly the path engine.process_image takes when
        # --upscale is used, so it must round-trip.
        original = [[11.0, 23.0], [57.0, 23.0], [57.0, 48.0], [11.0, 48.0]]
        up = geo.scale_polygon(original, 2.0)
        assert geo.scale_polygon(up, 1 / 2.0) == original

    def test_scale_bbox(self):
        assert geo.scale_bbox((10, 10, 20, 20), 0.25) == (2.5, 2.5, 5.0, 5.0)

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_rejects_non_positive_factor(self, factor):
        with pytest.raises(geo.GeometryError, match="must be > 0"):
            geo.scale_bbox((0, 0, 1, 1), factor)


class TestOverlapRatios:
    def test_vertical_overlap_same_line(self):
        # Two words on one text line overlap almost fully in y.
        assert geo.vertical_overlap_ratio((0, 10, 50, 40), (60, 12, 110, 42)) > 0.9

    def test_vertical_overlap_different_lines(self):
        assert geo.vertical_overlap_ratio((0, 0, 50, 20), (0, 40, 50, 60)) == 0.0

    def test_horizontal_overlap_uses_narrower_box(self):
        # The narrow box is fully inside the wide one, so the ratio is 1.
        assert geo.horizontal_overlap_ratio((0, 0, 100, 10), (40, 0, 60, 10)) == 1.0

    def test_horizontal_overlap_disjoint(self):
        assert geo.horizontal_overlap_ratio((0, 0, 10, 10), (50, 0, 60, 10)) == 0.0
