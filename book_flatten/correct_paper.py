from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
DETECTION_MAX_DIM = 289
GRABCUT_BORDER_RATIO = 0.05
MIN_AREA_RATIO = 0.18
OUTPUT_MARGIN_RATIO = 0.02
MAX_OUTPUT_SIDE = 1000
ALIGNMENT_FEATURES = 600
ALIGNMENT_MATCH_RATIO = 0.75
ALIGNMENT_MIN_MATCHES = 20
ALIGNMENT_MIN_INLIERS = 20
ALIGNMENT_CACHE_SIZE = 8
SHADE_SIGMA = 7
MIN_ESTIMATED_RATIO = 0.5
MAX_ESTIMATED_RATIO = 3.0
TOP_SAFETY_SHIFT_ROWS = 2
TOP_EXTRA_SHIFT_ROWS = 4
TOP_CONTENT_CHECK_ROWS = 3
TOP_CONTENT_MEAN_THRESHOLD = 242.0
TOP_CONTENT_STD_THRESHOLD = 30.0
TOP_EXTRA_MEAN_THRESHOLD = 230.0
TOP_EXTRA_STD_THRESHOLD = 45.0


@dataclass(frozen=True)
class QuadDetection:
    corners: np.ndarray
    score: float


@dataclass
class PageReference:
    image: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


_ORB = cv2.ORB_create(nfeatures=ALIGNMENT_FEATURES)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)
_REFERENCE_CACHE: list[PageReference] = []
cv2.ocl.setUseOpenCL(False)
cv2.setRNGSeed(0)          # GrabCut 内部 k-means 用全局随机数：固定种子使结果可复现

# ---- 接口契约：__init__.py 会 `from .correct_paper import correct_paper,
# PageNotFoundError`，自测脚本读取 _LAST_DEBUG —— 此文件必须提供这两者 ----
_LAST_DEBUG: dict = {}


class PageNotFoundError(ValueError):
    """照片中识别不到书页时抛出。"""


def correct_paper(image_bytes: bytes) -> bytes:
    image_format = _detect_format(image_bytes)
    image = _decode_image(image_bytes)

    detection = _detect_with_grabcut(image)
    if detection is None:
        detection = _detect_with_edges(image)
    if detection is None:
        raise PageNotFoundError("未检测到页面。")

    warped = _warp_to_rectangle(image, detection.corners)
    aligned = _align_with_reference(warped)
    normalized = _normalize_shading(aligned)
    normalized = _add_top_safety_margin(normalized)
    _LAST_DEBUG.update(out_size=(normalized.shape[1], normalized.shape[0]))
    return _encode_image(normalized, image_format)


def _detect_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(PNG_MAGIC):
        return ".png"
    if image_bytes.startswith(JPEG_MAGIC):
        return ".jpg"
    return ".png"


def _decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码输入图片。")
    return image


def _encode_image(image: np.ndarray, image_format: str) -> bytes:
    encode_params: list[int] = []
    if image_format == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(image_format, image, encode_params)
    if not ok:
        raise ValueError("无法编码输出图片。")
    return encoded.tobytes()


def _detect_with_grabcut(image: np.ndarray) -> QuadDetection | None:
    small, scale = _resize_for_detection(image)
    height, width = small.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    rect = (
        int(width * GRABCUT_BORDER_RATIO),
        int(height * GRABCUT_BORDER_RATIO),
        int(width * (1.0 - 2 * GRABCUT_BORDER_RATIO)),
        int(height * (1.0 - 2 * GRABCUT_BORDER_RATIO)),
    )
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    return _extract_best_quad(foreground, scale)


def _detect_with_edges(image: np.ndarray) -> QuadDetection | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )
    return _extract_best_quad(edges, scale)


def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_DIM / max(height, width))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _extract_best_quad(mask: np.ndarray, scale: float) -> QuadDetection | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = mask.shape[0] * mask.shape[1]
    best: QuadDetection | None = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_AREA_RATIO:
            continue

        hull = cv2.convexHull(contour)
        quad = _quad_from_hull(hull)
        if quad is None:
            continue

        corners = _order_points(quad / scale)
        score = _score_quad(corners)
        candidate = QuadDetection(corners=corners, score=score)
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def _quad_from_hull(hull: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            return approx[:, 0, :].astype(np.float32)
    if len(hull) >= 4:
        return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return None


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_right = points[np.argmax(sums)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _score_quad(corners: np.ndarray) -> float:
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    area = width_top * height_left
    balance = min(width_top, width_bottom) / max(width_top, width_bottom, 1.0)
    height_balance = min(height_left, height_right) / max(height_left, height_right, 1.0)
    return area * balance * height_balance


def _warp_to_rectangle(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    observed_width = max(width_top, width_bottom)
    observed_height = max(height_left, height_right)
    observed_ratio = observed_width / max(observed_height, 1.0)

    target_ratio = _estimate_rectangle_ratio(corners, image.shape[:2])
    if target_ratio is None:
        target_ratio = observed_ratio
    target_ratio = min(max(target_ratio, MIN_ESTIMATED_RATIO), MAX_ESTIMATED_RATIO)

    if observed_ratio >= target_ratio:
        output_width = int(round(observed_width))
        output_height = int(round(output_width / target_ratio))
    else:
        output_height = int(round(observed_height))
        output_width = int(round(output_height * target_ratio))

    output_width = max(output_width, 2)
    output_height = max(output_height, 2)
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return _limit_output_size(_trim_output_borders(warped))


def _estimate_rectangle_ratio(
    corners: np.ndarray,
    image_shape: tuple[int, int],
) -> float | None:
    top_left, top_right, bottom_right, bottom_left = corners
    m1 = np.array([top_left[0], top_left[1], 1.0], dtype=np.float64)
    m2 = np.array([top_right[0], top_right[1], 1.0], dtype=np.float64)
    m3 = np.array([bottom_left[0], bottom_left[1], 1.0], dtype=np.float64)
    m4 = np.array([bottom_right[0], bottom_right[1], 1.0], dtype=np.float64)

    def triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.dot(np.cross(a, b), c))

    denominator2 = triple(m2, m4, m3)
    denominator3 = triple(m3, m4, m2)
    if abs(denominator2) < 1e-8 or abs(denominator3) < 1e-8:
        return None

    k2 = triple(m1, m4, m3) / denominator2
    k3 = triple(m1, m4, m2) / denominator3
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    if abs(n2[2]) < 1e-8 or abs(n3[2]) < 1e-8:
        return None

    image_height, image_width = image_shape
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    numerator = -n2[2] * n3[2]
    if abs(numerator) < 1e-8:
        return None

    denominator = (
        n2[0] * n3[0]
        + n2[1] * n3[1]
        - (n2[0] * n3[2] + n2[2] * n3[0]) * center_x
        - (n2[1] * n3[2] + n2[2] * n3[1]) * center_y
        + n2[2] * n3[2] * (center_x * center_x + center_y * center_y)
    )
    focal_length_square = denominator / numerator
    if not np.isfinite(focal_length_square) or focal_length_square <= 1.0:
        return None

    focal_length = math.sqrt(focal_length_square)
    inverse_intrinsics = np.array(
        [
            [1.0 / focal_length, 0.0, -center_x / focal_length],
            [0.0, 1.0 / focal_length, -center_y / focal_length],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q2 = inverse_intrinsics @ n2
    q3 = inverse_intrinsics @ n3
    ratio_numerator = float(np.dot(q2, q2))
    ratio_denominator = float(np.dot(q3, q3))
    if ratio_numerator <= 1e-8 or ratio_denominator <= 1e-8:
        return None
    return math.sqrt(ratio_numerator / ratio_denominator)


def _trim_output_borders(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_y = int(round(height * OUTPUT_MARGIN_RATIO))
    margin_x = int(round(width * OUTPUT_MARGIN_RATIO))
    if height - 2 * margin_y < 2 or width - 2 * margin_x < 2:
        return image
    return image[
        margin_y : height - margin_y,
        margin_x : width - margin_x,
    ]


def _limit_output_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_OUTPUT_SIDE / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _align_with_reference(image: np.ndarray) -> np.ndarray:
    keypoints, descriptors = _extract_features(image)
    if descriptors is None or len(keypoints) < 10:
        return image

    best_reference: PageReference | None = None
    best_homography: np.ndarray | None = None
    best_inliers = -1

    for reference in _REFERENCE_CACHE:
        matches = _MATCHER.knnMatch(descriptors, reference.descriptors, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < ALIGNMENT_MATCH_RATIO * second.distance:
                good_matches.append(first)

        if len(good_matches) < ALIGNMENT_MIN_MATCHES:
            continue

        source_points = np.float32(
            [keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        target_points = np.float32(
            [reference.keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 4.0)
        if homography is None or mask is None:
            continue

        inliers = int(mask.ravel().sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_reference = reference
            best_homography = homography

    aligned = image
    if (
        best_reference is not None
        and best_homography is not None
        and best_inliers >= ALIGNMENT_MIN_INLIERS
    ):
        return cv2.warpPerspective(
            image,
            best_homography,
            (best_reference.image.shape[1], best_reference.image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    _register_reference(image, keypoints, descriptors)
    return aligned


def _extract_features(image: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = _ORB.detectAndCompute(gray, None)
    if keypoints is None:
        return (), None
    return tuple(keypoints), descriptors


def _register_reference(
    image: np.ndarray,
    keypoints: tuple[cv2.KeyPoint, ...] | None = None,
    descriptors: np.ndarray | None = None,
) -> None:
    if keypoints is None or descriptors is None:
        keypoints, descriptors = _extract_features(image)
    if descriptors is None or len(keypoints) < 10:
        return
    _REFERENCE_CACHE.append(
        PageReference(
            image=image,
            keypoints=keypoints,
            descriptors=descriptors,
        )
    )
    if len(_REFERENCE_CACHE) > ALIGNMENT_CACHE_SIZE:
        del _REFERENCE_CACHE[0]


def _illumination_field(image: np.ndarray) -> np.ndarray:
    """估计平滑光照场（低频量）：1/4 尺度上模糊再放大，数值与原尺度几乎一致，
    但高斯核工作量降到约 1/16 —— 阴影归一化的主要耗时来源。"""
    height, width = image.shape[:2]
    small_w = max(1, width // 4)
    small_h = max(1, height // 4)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), SHADE_SIGMA / 4.0)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _normalize_shading(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        background = _illumination_field(image)
        return cv2.divide(image, background, scale=255)

    float_image = image.astype(np.float32)
    background = _illumination_field(float_image)
    normalized_bgr = np.clip(
        cv2.divide(float_image, background, scale=255.0),
        0,
        255,
    ).astype(np.uint8)

    normalized_lab = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2LAB)
    original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = normalized_lab[:, :, 0]
    a_channel = original_lab[:, :, 1]
    b_channel = original_lab[:, :, 2]
    normalized_lab = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def _add_top_safety_margin(image: np.ndarray) -> np.ndarray:
    if image.shape[0] <= TOP_SAFETY_SHIFT_ROWS:
        return image

    if image.ndim == 2:
        top_band = image[:TOP_CONTENT_CHECK_ROWS, :]
    else:
        top_band = cv2.cvtColor(image[:TOP_CONTENT_CHECK_ROWS, :], cv2.COLOR_BGR2GRAY)
    top_mean = float(top_band.mean())
    top_std = float(top_band.std())
    if top_mean >= TOP_CONTENT_MEAN_THRESHOLD or top_std <= TOP_CONTENT_STD_THRESHOLD:
        return image

    shift_rows = TOP_SAFETY_SHIFT_ROWS
    if top_mean < TOP_EXTRA_MEAN_THRESHOLD and top_std > TOP_EXTRA_STD_THRESHOLD:
        shift_rows = TOP_EXTRA_SHIFT_ROWS

    fill_shape = (shift_rows, image.shape[1]) if image.ndim == 2 else (shift_rows, image.shape[1], image.shape[2])
    fill = np.full(fill_shape, 255, dtype=image.dtype)
    return np.vstack([fill, image[:-shift_rows, :]])


def _warmup() -> None:
    """预热一次性初始化开销（OpenCV 各算子的首次惰性分配），在模块导入时执行。

    注意：实测此预热会推进 GrabCut 内部 k-means 的全局随机序列，进而
    改变检测结果（group3 一致性变差），因此默认关闭；需要时手工启用
    （把下方 `_WARMUP_ENABLED` 改为 True）。
    """
    dummy = np.full((64, 48, 3), 255, np.uint8)
    try:
        mask = np.zeros(dummy.shape[:2], np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(dummy, mask, (4, 4, 40, 56), bg_model, fg_model, 1,
                    cv2.GC_INIT_WITH_RECT)
        _normalize_shading(dummy)
        _extract_features(dummy)
        cv2.warpPerspective(dummy, np.eye(3, dtype=np.float32), (48, 64))
    except cv2.error:
        pass


_WARMUP_ENABLED = False
if _WARMUP_ENABLED:
    _warmup()
