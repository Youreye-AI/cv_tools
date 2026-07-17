from main import compute_resize_scale


def test_compute_resize_scale_no_scaling_when_already_fits():
    scale = compute_resize_scale(height=480, width=640, max_width=1280, max_height=720)

    assert scale == 1.0


def test_compute_resize_scale_no_scaling_when_exactly_fits():
    scale = compute_resize_scale(height=720, width=1280, max_width=1280, max_height=720)

    assert scale == 1.0


def test_compute_resize_scale_scales_down_when_width_overflows():
    scale = compute_resize_scale(height=720, width=2560, max_width=1280, max_height=720)

    assert scale == 0.5


def test_compute_resize_scale_scales_down_when_height_overflows():
    scale = compute_resize_scale(height=1440, width=1280, max_width=1280, max_height=720)

    assert scale == 0.5


def test_compute_resize_scale_uses_smaller_ratio_when_both_overflow():
    scale = compute_resize_scale(height=2160, width=3840, max_width=1280, max_height=720)

    assert scale == 1280 / 3840
