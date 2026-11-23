"""批次提取連續影格的額頭與雙頰色彩及變化率。

The extractor intentionally produces interpretable handcrafted features.  It is
not a diagnostic system; model training belongs in a separate stage.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cv2
import mediapipe as mp #偵測臉部關鍵點
import numpy as np     #計算中位數、標準差、色差、角度


# 建立一個專門代表 特徵擷取失敗 的error
class FeatureExtractionError(RuntimeError):
    """Raised when an image cannot yield trustworthy facial ROIs."""

    def __init__(
        self,
        message: str,
        *,
        region: str = "",
        valid_pixel_ratio: float | None = None,
        dark_pixel_ratio: float | None = None,
        bright_pixel_ratio: float | None = None,
        preview: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.region = region
        self.valid_pixel_ratio = valid_pixel_ratio
        self.dark_pixel_ratio = dark_pixel_ratio
        self.bright_pixel_ratio = bright_pixel_ratio
        self.preview = preview


@dataclass(frozen=True)
#專門存放ROI資訊的data structure
class CheekRegion:
    name: str
    landmark_ids: tuple[int, ...]
    color: tuple[int, int, int]


# MediaPipe 中左右對稱的臉頰區域點
# 後續會依實際 x 座標重新判斷圖片左側與圖片右側
# 因此不直接依賴解剖學上的左右名稱
CHEEK_REGIONS = (
    CheekRegion("cheek_a", (205,), (46, 204, 113)),
    CheekRegion("cheek_b", (425,), (46, 204, 113)),
)

# ROI 尺寸相對於雙眼距離的比例
CHEEK_WIDTH_RATIO = 0.50
CHEEK_HEIGHT_RATIO = 0.34

# 負值代表將 landmark 中心稍微往上移
CHEEK_Y_OFFSET_RATIO = -0.15

# MediaPipe 兩側眼睛輪廓點，用來旋正人臉及計算雙眼距離
EYE_A = (33, 133, 159, 145)
EYE_B = (362, 263, 386, 374)

# 額頭採幾何定位初始值，尚未做頭髮分割，需以原始圖片人工確認
# 使用臉部上緣點與雙眼中點之間的位置，避免直接在臉部上緣取樣
FOREHEAD_TOP_LANDMARK = 10
FOREHEAD_CENTER_FRACTION = 0.45  # 0 靠近上緣，1 靠近雙眼
FOREHEAD_WIDTH_RATIO = 0.50      # 寬度相對於雙眼距離
FOREHEAD_HEIGHT_RATIO = 0.28     # 高度相對於雙眼距離
FOREHEAD_MAX_GAP_RATIO = 0.45    # 高度不超過上緣至雙眼垂直距離的此比例
FOREHEAD_COLOR = (255, 180, 0)   # OpenCV BGR，與雙頰綠框區分


# 計算色相(Hue)的圓形平均值
# 因為色相是一個圓，0度和360度本質上是一樣的，如果用相加/2會得到180，明顯錯誤
def _circular_hue_stats(hue: np.ndarray, saturation: np.ndarray) -> tuple[float, float]:
    """Return saturation-weighted circular mean and circular deviation in degrees."""
    #用飽和度當權重，因為當飽和度很低時Hue的影響很小
    weights = saturation.astype(np.float64)
    valid = weights > 0.05
    if not np.any(valid):
        return float("nan"), float("nan")
    radians = np.deg2rad(hue[valid].astype(np.float64)) #先將角度轉成弧度
    weights = weights[valid]
    # 再把色相投影到單位圓，計算平均
    x = np.average(np.cos(radians), weights=weights)
    y = np.average(np.sin(radians), weights=weights)
    mean = np.rad2deg(np.arctan2(y, x)) % 360.0 #最後轉回角度
    resultant = np.clip(np.hypot(x, y), 1e-12, 1.0) #色相方向的一致程度
    deviation = np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))
    return float(mean), float(deviation)


#計算顏色特徵 (input：一塊臉頰照片，以及名字)
def colour_features(bgr_roi: np.ndarray, prefix: str) -> dict[str, float]:
    """Extract HSV/CIELAB medians and distributions using the same valid pixels.

    Per cheek: 6 existing colour features, 15 distribution features and 1 QC
    field. Hue remains circular; linear distribution statistics exclude hue.
    """
    if bgr_roi.size == 0:
        raise FeatureExtractionError(f"Empty ROI: {prefix}")

    #BRG轉RGB、HSV、CIELAB
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # 排除亮度太低或過度曝光的像素。先保留既有門檻，並輸出比例供 QC 判讀
    dark = hsv[..., 2] <= 0.03
    bright = hsv[..., 2] >= 0.98
    valid = ~(dark | bright)
    valid_ratio = float(np.mean(valid))
    dark_ratio = float(np.mean(dark))
    bright_ratio = float(np.mean(bright))
    if np.count_nonzero(valid) < max(20, int(valid.size * 0.25)): #有效像素的最低門檻:至少20個有效像素或至少占整個ROI的25%
        raise FeatureExtractionError(
            f"Too few usable pixels in {prefix} "
            f"(valid={valid_ratio:.3f}, dark={dark_ratio:.3f}, bright={bright_ratio:.3f})",
            region=prefix,
            valid_pixel_ratio=valid_ratio,
            dark_pixel_ratio=dark_ratio,
            bright_pixel_ratio=bright_ratio,
        )

    h, s = (hsv[..., channel][valid] for channel in range(2))
    l_star, a_star, b_star = (lab[..., channel][valid] for channel in range(3))
    hue_mean, _ = _circular_hue_stats(h, s)
    features = {
        f"{prefix}_hue_circular_mean": hue_mean,                 # Hue圓型平均
        f"{prefix}_s_median": float(np.median(s)),               # S中位數
        f"{prefix}_lab_l_median": float(np.median(l_star)),      # L*中位數
        f"{prefix}_lab_a_median": float(np.median(a_star)),      # a*中位數
        f"{prefix}_lab_b_median": float(np.median(b_star)),      # b*中位數
        f"qc_{prefix}_valid_pixel_ratio": float(np.mean(valid)), # 有效比例QC (Quality控制，)不會送入模型
        f"qc_{prefix}_dark_pixel_ratio": dark_ratio,
        f"qc_{prefix}_bright_pixel_ratio": bright_ratio,
    }

    # 僅新增分布特徵；沿用上述有效像素，保留既有中位數、Hue 與 QC 計算
    # 使用 float64 計算，標準差採母體標準差 (ddof=0)
    # S、V 維持 0–1；Lab 維持 OpenCV 浮點 CIELAB 尺度
    for channel_name, values in (
        ("s", s), ("lab_l", l_star),
        ("lab_a", a_star), ("lab_b", b_star),
    ):
        values = values.astype(np.float64)
        p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
        features[f"{prefix}_{channel_name}_std"] = float(np.std(values, ddof=0))
        features[f"{prefix}_{channel_name}_iqr"] = float(p75 - p25)
        features[f"{prefix}_{channel_name}_p90_p10"] = float(p90 - p10)

    return features



class FaceFeatureExtractor:
    """沿用原版定位與旋正，讓新舊影片特徵可比較。"""

    # mediapipe初始化，人臉偵測用
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, #每張照片都獨立做人臉偵測
            max_num_faces=1,        #最多只處理一張臉
            refine_landmarks=True,  #用眼睛附近當關鍵點
            min_detection_confidence=min_detection_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceFeatureExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    #取得臉部關鍵點
    def _landmarks(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #mediapipe預期RGB，但openCV讀進來是BGR
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            raise FeatureExtractionError("No face detected")
        height, width = image.shape[:2]
        #回傳的座標是正規化座標，要換算成實際像素
        return np.array(
            [(p.x * width, p.y * height) for p in result.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )

    #根據雙眼旋正人臉
    def _align(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        landmarks = self._landmarks(image)
        centre_a = landmarks[list(EYE_A)].mean(axis=0)
        centre_b = landmarks[list(EYE_B)].mean(axis=0)
        if centre_a[0] > centre_b[0]:
            centre_a, centre_b = centre_b, centre_a
        dy, dx = centre_b[1] - centre_a[1], centre_b[0] - centre_a[0]
        angle = np.degrees(np.arctan2(dy, dx))
        centre = tuple(((centre_a + centre_b) / 2.0).tolist())
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        aligned = cv2.warpAffine( #套用旋轉
            image, matrix, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        # 使用矩陣轉換landmarks
        ones = np.ones((len(landmarks), 1), dtype=np.float32)
        homogeneous = np.hstack([landmarks, ones])
        aligned_landmarks = homogeneous @ matrix.T

        return aligned, aligned_landmarks

    @staticmethod #從landmarks得到完整臉的框
    def _face_box(landmarks: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1 = np.floor(landmarks.min(axis=0)).astype(int)
        x2, y2 = np.ceil(landmarks.max(axis=0)).astype(int)
        return (
            int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1)),
            int(np.clip(x2, 1, width)), int(np.clip(y2, 1, height)),
        )

    @staticmethod
    def _cheek_boxes(landmarks: np.ndarray,shape: tuple[int, ...],) -> list[tuple[str, tuple[int, int, int, int], tuple[int, int, int]]]:
        """以臉頰 landmarks 定位中心，並以雙眼距離決定矩形大小。"""
        image_height, image_width = shape[:2]

        eye_a_center = landmarks[list(EYE_A)].mean(axis=0)
        eye_b_center = landmarks[list(EYE_B)].mean(axis=0)
        eye_distance = float(np.linalg.norm(eye_a_center - eye_b_center))

        if eye_distance < 1.0:
            raise FeatureExtractionError("Invalid eye distance")

        roi_width = max(1, round(eye_distance * CHEEK_WIDTH_RATIO))
        roi_height = max(1, round(eye_distance * CHEEK_HEIGHT_RATIO))
        y_offset = eye_distance * CHEEK_Y_OFFSET_RATIO

        candidates = []

        for region in CHEEK_REGIONS:
            center = landmarks[list(region.landmark_ids)].mean(axis=0)
            center_x = float(center[0])
            center_y = float(center[1] + y_offset)

            x1 = round(center_x - roi_width / 2)
            y1 = round(center_y - roi_height / 2)
            x2 = round(center_x + roi_width / 2)
            y2 = round(center_y + roi_height / 2)

            x1 = int(np.clip(x1, 0, image_width - 1))
            y1 = int(np.clip(y1, 0, image_height - 1))
            x2 = int(np.clip(x2, x1 + 1, image_width))
            y2 = int(np.clip(y2, y1 + 1, image_height))

            candidates.append(((center_x, center_y), (x1, y1, x2, y2), region.color))

        # 不依賴自拍鏡像或受試者解剖左右，直接按照圖片中的 x 座標排序
        candidates.sort(key=lambda item: item[0][0])

        return [
            ("image_left_cheek", candidates[0][1], candidates[0][2]),
            ("image_right_cheek", candidates[1][1], candidates[1][2]),
        ]

    @staticmethod
    def _forehead_box(landmarks: np.ndarray, shape: tuple[int, ...]) -> tuple[str, tuple[int, int, int, int], tuple[int, int, int]]:
        """在旋正後座標計算額頭矩形；超出影像時拒絕，避免悄悄位移。"""
        eye_a = landmarks[list(EYE_A)].mean(axis=0)
        eye_b = landmarks[list(EYE_B)].mean(axis=0)
        eye_mid = (eye_a + eye_b) / 2.0
        top = landmarks[FOREHEAD_TOP_LANDMARK]
        eye_distance = float(np.linalg.norm(eye_a - eye_b))
        gap = float(eye_mid[1] - top[1])
        if not np.isfinite([*eye_mid, *top, eye_distance, gap]).all() or eye_distance < 1 or gap < 4:
            raise FeatureExtractionError("Invalid forehead geometry", region="image_forehead")
        center = top + FOREHEAD_CENTER_FRACTION * (eye_mid - top)
        width = max(2, round(eye_distance * FOREHEAD_WIDTH_RATIO))
        height = max(2, round(min(eye_distance * FOREHEAD_HEIGHT_RATIO, gap * FOREHEAD_MAX_GAP_RATIO)))
        x1 = round(float(center[0]) - width / 2)
        y1 = round(float(center[1]) - height / 2)
        x2, y2 = x1 + width, y1 + height
        if x1 < 0 or y1 < 0 or x2 > shape[1] or y2 > shape[0]:
            raise FeatureExtractionError("Forehead ROI outside image", region="image_forehead")
        return "image_forehead", (x1, y1, x2, y2), FOREHEAD_COLOR

    @staticmethod
    def _draw_regions(image: np.ndarray, boxes: list) -> np.ndarray:
        """只在副本畫框，避免文字與框線污染分析影像。"""
        preview = image.copy()
        for name, (x1, y1, x2, y2), color in boxes:
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(preview, name, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return preview

    def preview_regions(self, image: np.ndarray) -> tuple[np.ndarray, list]:
        """僅確認三區定位，不套用有效像素門檻或提取特徵。"""
        if image is None or image.size == 0:
            raise FeatureExtractionError("Image is empty")
        aligned, landmarks = self._align(image)
        boxes = self._cheek_boxes(landmarks, aligned.shape)
        try:
            boxes.append(self._forehead_box(landmarks, aligned.shape))
        except FeatureExtractionError as error:
            error.preview = self._draw_regions(aligned, boxes)
            raise
        return self._draw_regions(aligned, boxes), boxes

    def extract_regions(self, image: np.ndarray) -> dict:
        """影片模式按區記錄成功與失敗，沿用原本的定位和像素門檻。"""
        aligned, landmarks = self._align(image)
        boxes = []
        result = {}
        try:
            boxes.extend(self._cheek_boxes(landmarks, aligned.shape))
        except FeatureExtractionError as error:
            for region in VIDEO_REGIONS[:2]:
                result[f"{region}_status"] = "roi_failed"
                result[f"{region}_error"] = str(error)
        try:
            boxes.append(self._forehead_box(landmarks, aligned.shape))
        except FeatureExtractionError as error:
            result["image_forehead_status"] = "roi_failed"
            result["image_forehead_error"] = str(error)
        for region, (x1, y1, x2, y2), _ in boxes:
            try:
                values = colour_features(aligned[y1:y2, x1:x2], region)
                # 低飽和度可能使 Hue 無定義；留空，其他欄位照常保留
                result.update({key: val for key, val in values.items() if math.isfinite(val)})
                result[f"{region}_status"] = "ok"
            except FeatureExtractionError as error:
                result[f"{region}_status"] = "feature_failed"
                result[f"{region}_error"] = str(error)
                for key in ("valid_pixel_ratio", "dark_pixel_ratio", "bright_pixel_ratio"):
                    result[f"qc_{region}_{key}"] = getattr(error, key)
        return result

# 連續影格輸出欄位固定；全部失敗也保留相同表頭


VIDEO_REGIONS = ("image_left_cheek", "image_right_cheek", "image_forehead")
COLOUR_KEYS = ("hue_circular_mean", "s_median", "lab_l_median",
               "lab_a_median", "lab_b_median") + tuple(
    f"{channel}_{stat}"
    for channel in ("s", "lab_l", "lab_a", "lab_b")
    for stat in ("std", "iqr", "p90_p10")
)
STATIC_KEYS = tuple(f"{region}_{key}" for region in VIDEO_REGIONS for key in COLOUR_KEYS)
# 三區各保留五條逐幀訊號；分布統計另存，不對它們做差分
SIGNALS = tuple(f"{region}_{key}" for region in VIDEO_REGIONS for key in
                ("hue_circular_mean", "s_median", "lab_l_median", "lab_a_median", "lab_b_median"))
REGION_FIELDS = tuple(
    field for region in VIDEO_REGIONS
    for field in (f"{region}_status", f"{region}_error", f"qc_{region}_valid_pixel_ratio",
                  f"qc_{region}_dark_pixel_ratio", f"qc_{region}_bright_pixel_ratio")
)
FRAME_FIELDS = (
    "video_id", "subject_id", "session_id", "source_csv_row", "frame_index",
    "relative_time_s", "image_path", "frame_status", "frame_error", "timestamp_status",
    "drop_status", "possible_blur", "previous_frame_index", "delta_time_s",
) + REGION_FIELDS + STATIC_KEYS + tuple(
    field for signal in SIGNALS for field in (f"{signal}_rate", f"{signal}_rate_status")
)


def _finite(value):
    """只接受有限數值；缺值與 0 秒必須分開。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value):
    """相容布林字串；不把非空的 False 誤當成真。"""
    return str(value).strip().lower() in {"true", "1"}


def _frame_path(csv_path, image_path, frames_dir):
    """支援 Windows 分隔符，拒絕絕對路徑及跳出資料夾的相對路徑。"""
    path = PureWindowsPath(image_path)
    if not path.parts or path.anchor or ".." in path.parts or path.suffix.lower() != ".png":
        return None
    root = (frames_dir or csv_path.parent).resolve()
    parts = (path.name,) if frames_dir else path.parts
    target = root.joinpath(*parts).resolve()
    return target if target.is_relative_to(root) else None


def _read_manifest(csv_path, frames_dir):
    """讀取單一影片清單；保留 CSV 順序，不靠檔名排序猜時間。"""
    required = {"source_video", "frame_index", "image_path", "relative_time_s",
                "timestamp_valid", "png_saved"}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("影格 CSV 有重複欄名")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"影格 CSV 缺少必要欄位：{sorted(required - set(reader.fieldnames or []))}")
        records = list(reader)
    if not records or any(None in record or any(v is None for v in record.values()) for record in records):
        raise ValueError("影格 CSV 沒有資料，或有列的欄位數與表頭不符")
    for key in ("source_video", "subject_id", "session_id"):
        values = {record.get(key, "").strip() for record in records}
        if len(values) > 1 or (key == "source_video" and not next(iter(values))):
            raise ValueError(f"同一份 CSV 須對應一支影片，欄位 {key} 不一致或缺值")
    video_id = PureWindowsPath(records[0]["source_video"]).stem
    rows, paths = [], []
    for source_row, record in enumerate(records, start=2):
        try:
            index = int(record["frame_index"])
            if index < 1:
                index = None
        except ValueError:
            index = None
        path = _frame_path(csv_path, record["image_path"], frames_dir)
        paths.append(path)
        rows.append({
            "video_id": video_id, "subject_id": record.get("subject_id", ""),
            "session_id": record.get("session_id", ""), "source_csv_row": source_row,
            "frame_index": index, "relative_time_s": _finite(record["relative_time_s"]),
            "image_path": record["image_path"], "drop_status": record.get("drop_status", ""),
            "possible_blur": record.get("possible_blur", ""),
        })
    indices = Counter(row["frame_index"] for row in rows)
    path_counts = Counter(str(path).casefold() for path in paths if path is not None)
    last_time, last_index = None, 0
    for row, record, path in zip(rows, records, paths):
        index, timestamp = row["frame_index"], row["relative_time_s"]
        if index is None:
            row["timestamp_status"] = "invalid_frame_index"
        elif indices[index] > 1:
            row["timestamp_status"] = "duplicate_frame_index"
        elif index <= last_index:
            row["timestamp_status"] = "frame_index_out_of_order"
        elif not _flag(record["timestamp_valid"]) or timestamp is None or timestamp < 0:
            row["timestamp_status"] = "invalid_timestamp"
        elif last_time is not None and timestamp <= last_time:
            row["timestamp_status"] = "non_increasing_timestamp"
        else:
            row["timestamp_status"] = "ok"
            last_time = timestamp
        if index is not None:
            last_index = max(last_index, index)
        if row["timestamp_status"] != "ok":
            row["frame_status"] = "invalid_manifest"
        elif not _flag(record["png_saved"]):
            row["frame_status"] = "png_not_saved"
        elif path is None:
            row["frame_status"] = "invalid_image_path"
        elif path_counts[str(path).casefold()] > 1:
            row["frame_status"] = "duplicate_image_path"
        elif not path.is_file():
            row["frame_status"] = "missing_image"
        else:
            row["frame_status"] = "ready"
    return rows, paths


def _read_video_image(path):
    """使用位元組解碼，讓 Windows 中文路徑也能讀圖。"""
    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not payload.size:
        raise FeatureExtractionError("Empty PNG")
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise FeatureExtractionError("Cannot decode PNG")
    return image


def _process_frames(rows, paths, extractor):
    """逐張保留狀態：讀圖、共用定位、各區取樣分開記錄。"""
    for row, path in zip(rows, paths):
        for region in VIDEO_REGIONS:
            row[f"{region}_status"] = "not_processed"
        if row["frame_status"] != "ready":
            continue
        try:
            image = _read_video_image(path)
        except (OSError, FeatureExtractionError, cv2.error) as error:
            row.update(frame_status="image_read_failed", frame_error=str(error))
            continue
        try:
            row.update(extractor.extract_regions(image))
        except (FeatureExtractionError, cv2.error) as error:
            row.update(frame_status="face_processing_failed", frame_error=str(error))
            continue
        ok_count = sum(row[f"{region}_status"] == "ok" for region in VIDEO_REGIONS)
        row["frame_status"] = {0: "all_regions_failed", 3: "ok"}.get(ok_count, "partial")


def _time_difference(previous, current):
    """只對原解碼順序緊鄰的有效時間戳計算時間差。"""
    if previous is None:
        return None, "first_frame"
    if previous["timestamp_status"] != "ok" or current["timestamp_status"] != "ok":
        return None, "invalid_timestamp_pair"
    if current["frame_index"] != previous["frame_index"] + 1:
        return None, "non_adjacent_frame_index"
    dt = current["relative_time_s"] - previous["relative_time_s"]
    return (dt, "ok") if dt > 0 else (None, "non_positive_time_delta")


def _add_rates(rows, gap_factor):
    """依 PTS 差分；缺影格、品質失敗及異常間隔都不跨算。"""
    pairs = [_time_difference(rows[i - 1] if i else None, row) for i, row in enumerate(rows)]
    typical = [dt for row, (dt, _) in zip(rows, pairs)
               if dt is not None and row["drop_status"] != "possible_gap_not_confirmed"]
    median_dt = float(np.median(typical)) if typical else None
    limit = median_dt * gap_factor if median_dt is not None else None
    for i, (row, (dt, reason)) in enumerate(zip(rows, pairs)):
        previous = rows[i - 1] if i else None
        if previous is not None:
            row["previous_frame_index"] = previous["frame_index"]
        row["delta_time_s"] = dt
        if dt is not None and (row["drop_status"] == "possible_gap_not_confirmed"
                               or (limit is not None and dt > limit)):
            reason = "large_time_gap"
        for signal in SIGNALS:
            status = reason
            if reason == "ok":
                before, after = _finite(previous.get(signal)), _finite(row.get(signal))
                if before is None or after is None:
                    status = "missing_region_value"
                else:
                    delta = after - before
                    # 色相以最短帶符號角差處理，避免 359° 到 1° 的假尖峰
                    if signal.endswith("_hue_circular_mean"):
                        delta = (delta + 180.0) % 360.0 - 180.0
                    rate = delta / dt
                    if math.isfinite(rate):
                        row[f"{signal}_rate"] = rate
                    else:
                        status = "non_finite_rate"
            row[f"{signal}_rate_status"] = status
    return median_dt, limit


def _write_table(path, fields, rows):
    """UTF-8 BOM 方便在 Excel 開啟；None 寫成空欄。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows, quality, median_dt, limit):
    """時間戳涵蓋與各區成功率分開，保留動態特徵的有效差分數。"""
    valid_times = [row["relative_time_s"] for row in rows if row["timestamp_status"] == "ok"]
    summary = {key: rows[0][key] for key in ("video_id", "subject_id", "session_id")}
    summary.update(
        frame_records=len(rows), valid_timestamp_count=len(valid_times),
        first_valid_time_s=valid_times[0] if valid_times else None,
        last_valid_time_s=valid_times[-1] if valid_times else None,
        timestamp_span_s=valid_times[-1] - valid_times[0] if valid_times else None,
        median_adjacent_dt_s=median_dt, large_gap_threshold_s=limit,
        forehead_manual_quality=quality,
    )
    for region in VIDEO_REGIONS:
        passed = [row for row in rows if row[f"{region}_status"] == "ok"]
        times = [row["relative_time_s"] for row in passed]
        summary[f"{region}_success_count"] = len(passed)
        summary[f"{region}_success_ratio"] = len(passed) / len(rows)
        summary[f"{region}_valid_time_span_s"] = times[-1] - times[0] if times else None
    for key in STATIC_KEYS:
        values = [row[key] for row in rows if key in row]
        if key.endswith("_hue_circular_mean"):
            # Hue 是圓周值，不能直接用線性中位數跨越 0/360 度。
            value, _ = _circular_hue_stats(np.asarray(values), np.ones(len(values)))
            summary[f"{key}_time_circular_mean"] = _finite(value)
        else:
            summary[f"{key}_time_median"] = float(np.median(values)) if values else None
    for signal in SIGNALS:
        values = [abs(row[f"{signal}_rate"]) for row in rows if f"{signal}_rate" in row]
        summary[f"{signal}_rate_count"] = len(values)
        summary[f"{signal}_abs_rate_median"] = float(np.median(values)) if values else None
        summary[f"{signal}_abs_rate_p90"] = float(np.percentile(values, 90)) if values else None
    return summary


def _write_curves(path, rows):
    """產生可用瀏覽器開啟的 SVG；缺值與中斷處不連線。"""
    panel_w, panel_h = 580, 190
    total_height = panel_h * len(SIGNALS)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="{total_height}" '
             f'viewBox="0 0 1160 {total_height}" font-family="sans-serif" font-size="12">',
             '<rect width="100%" height="100%" fill="white"/>']
    times = [row["relative_time_s"] for row in rows if row["timestamp_status"] == "ok"]
    t_min, t_max = (times[0], times[-1]) if times else (0.0, 1.0)
    for i, signal in enumerate(SIGNALS):
        base = "deg" if "_hue_" in signal else ("0 to 1" if signal.endswith("_s_median") else "Lab")
        for j, (column, unit) in enumerate(((signal, base), (f"{signal}_rate", f"{base} / s"))):
            left, top = j * panel_w + 76, i * panel_h + 36
            width, height = 475, 112
            values = [row[column] for row in rows if column in row]
            lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            parts.extend([
                f'<text x="{left}" y="{top - 12}">{column} ({unit})</text>',
                f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
                'fill="#f7f9fc" stroke="#bcc8d4"/>',
                f'<text x="{left - 8}" y="{top + 6}" text-anchor="end">{hi:.3g}</text>',
                f'<text x="{left - 8}" y="{top + height}" text-anchor="end">{lo:.3g}</text>',
                f'<text x="{left}" y="{top + height + 18}">{t_min:.3f} s</text>',
                f'<text x="{left + width}" y="{top + height + 18}" text-anchor="end">{t_max:.3f} s</text>',
            ])
            if not values:
                parts.append(f'<text x="{left + 16}" y="{top + 26}">No valid values</text>')
            last = None
            for row in rows:
                if column not in row:
                    last = None
                    continue
                x = left + width * (row["relative_time_s"] - t_min) / max(t_max - t_min, 1e-9)
                y = top + height * (hi - row[column]) / (hi - lo)
                wrap = not j and "_hue_" in signal and last is not None and abs(last[2] - row[column]) > 180
                if last is not None and not wrap and row[f"{signal}_rate_status"] == "ok":
                    parts.append(f'<line x1="{last[0]:.2f}" y1="{last[1]:.2f}" '
                                 f'x2="{x:.2f}" y2="{y:.2f}" stroke="#23649a"/>')
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.4" fill="#23649a"/>')
                last = (x, y, row[column])
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _save_diagnostics(folder, rows, paths, extractor, top_spikes):
    """依保存原因分開代表影格和變化率排名；排名不等於異常。"""
    folder.mkdir()
    samples_dir = folder / "roi_samples"
    changes_dir = folder / "top_changes"
    samples_dir.mkdir()
    changes_dir.mkdir()
    _write_curves(folder / "temporal_curves.svg", rows)

    def prepare_frame(i):
        # 同一影格可能同時入選抽樣和排名，只需重新定位與編碼一次
        if i in prepared:
            return prepared[i]
        row, path = rows[i], paths[i]
        result = {"original": None, "roi": None, "error": ""}
        prepared[i] = result
        if row["frame_status"] not in {"ok", "partial", "all_regions_failed", "face_processing_failed"}:
            result["error"] = row["frame_status"]
            return result
        try:
            image = _read_video_image(path)
            result["original"] = path.read_bytes()
            try:
                preview, _ = extractor.preview_regions(image)
            except FeatureExtractionError as error:
                if error.preview is None:
                    raise
                preview = error.preview
                result["error"] = str(error)
            ok, encoded = cv2.imencode(".png", preview)
            if not ok:
                raise OSError("Cannot encode ROI preview")
            result["roi"] = encoded.tobytes()
        except (OSError, FeatureExtractionError, cv2.error) as error:
            result["error"] = str(error)
        return result

    def save_frame(i, target_dir, stem, index_dir):
        data = prepare_frame(i)
        result = {"original": "", "roi": "", "error": data["error"]}
        for kind in ("original", "roi"):
            if data[kind] is not None:
                target = target_dir / f"{stem}_{kind}.png"
                try:
                    target.write_bytes(data[kind])
                    # CSV 中的預覽路徑一律相對於該 CSV 所在目錄
                    result[kind] = target.relative_to(index_dir).as_posix()
                except OSError as error:
                    result["error"] = "; ".join(filter(None, (result["error"], str(error))))
        return result

    prepared = {}
    candidates = [i for i, row in enumerate(rows) if row["timestamp_status"] == "ok"]
    representatives = []
    if candidates:
        for label, i in zip(("start", "middle", "end"),
                            (candidates[0], candidates[len(candidates) // 2], candidates[-1])):
            row = rows[i]
            stem = f"{label}_frame_{row['frame_index']:06d}"
            representatives.append(dict(sample=label, frame_index=row["frame_index"],
                                        relative_time_s=row["relative_time_s"],
                                        **save_frame(i, samples_dir, stem, samples_dir)))
    _write_table(samples_dir / "sample_index.csv",
                 ("sample", "frame_index", "relative_time_s", "original", "roi", "error"), representatives)
    spikes = []
    pair_fields = ("frame_index", "relative_time_s", "image_path", "original", "roi", "error")
    for signal in SIGNALS:
        rate = f"{signal}_rate"
        signal_dir = changes_dir / signal.removeprefix("image_").replace("_lab_", "_").removesuffix("_median")
        signal_dir.mkdir()
        ordered = sorted((i for i, row in enumerate(rows) if rate in row),
                         key=lambda i: abs(rows[i][rate]), reverse=True)
        for rank, i in enumerate(ordered[:top_spikes], start=1):
            entry = dict(signal=signal, rank=rank, rate=rows[i][rate], delta_time_s=rows[i]["delta_time_s"])
            for prefix, index in (("before", i - 1), ("after", i)):
                row = rows[index]
                metadata = {key: row[key] for key in pair_fields[:3]}
                stem = f"rank_{rank:02d}_{prefix}_frame_{row['frame_index']:06d}"
                metadata.update(save_frame(index, signal_dir, stem, changes_dir))
                entry.update({f"{prefix}_{key}": value for key, value in metadata.items()})
            spikes.append(entry)
    _write_table(changes_dir / "spikes.csv", ("signal", "rank", "rate", "delta_time_s") + tuple(
        f"{prefix}_{key}" for prefix in ("before", "after") for key in pair_fields), spikes)


def _run_video(manifest, output, gap_factor, top_spikes):
    """每支影片獨立初始化偵測器和時間序列，避免跨影片差分。"""
    frames = manifest.parent / "frames"
    if not frames.is_dir():
        raise ValueError("找不到與 frames.csv 同層的 frames 資料夾")
    rows, paths = _read_manifest(manifest, frames)
    output.mkdir(parents=True, exist_ok=False)
    has_images = any(row["frame_status"] == "ready" for row in rows)
    with (FaceFeatureExtractor() if has_images else nullcontext()) as extractor:
        _process_frames(rows, paths, extractor)
        median_dt, limit = _add_rates(rows, gap_factor)
        # 定位成功不代表沒有頭髮或手部遮擋，預設一律待人工複核
        summary = _summarize(rows, "pending", median_dt, limit)
        _write_table(output / "frame_features.csv", FRAME_FIELDS, rows)
        _write_table(output / "data_summary.csv", list(summary), [summary])
        _save_diagnostics(output / "diagnostics", rows, paths, extractor, top_spikes)
    summary["valid_feature_frame_count"] = sum(
        any(row[f"{region}_status"] == "ok" for region in VIDEO_REGIONS) for row in rows
    )
    return summary


def main():
    """遞迴掃描影片清單，並將失敗與部分成功都保留在批次總表。"""
    parser = argparse.ArgumentParser(description="批次提取影片三區 HS/Lab 與變化率")
    parser.add_argument("input_root", type=Path, help="包含各影片資料夾的根目錄")
    parser.add_argument("--output", type=Path, required=True, help="新的或空的輸出目錄")
    parser.add_argument("--gap-factor", type=float, default=2.5)
    parser.add_argument("--top-spikes", type=int, default=3)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"找不到輸入資料夾：{root}")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("請將輸入與輸出放在互不包含的資料夾")
    if not math.isfinite(args.gap_factor) or args.gap_factor <= 1 or args.top_spikes < 0:
        parser.error("gap-factor 需大於 1，top-spikes 不可為負數")
    manifests = sorted(p for p in root.rglob("frames.csv") if p.is_file())
    if not manifests:
        parser.error("找不到 frames.csv")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("請使用新的或空的輸出資料夾")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for manifest in manifests:
        # 保留來源資料夾名稱及相對層級，避免不同子目錄的同名影片互撞。
        target = output / "videos" / manifest.parent.relative_to(root)
        entry = {"source_manifest": manifest.relative_to(root).as_posix(),
                 "output_dir": target.relative_to(output).as_posix(), "error": ""}
        try:
            summary = _run_video(manifest, target, args.gap_factor, args.top_spikes)
            entry.update(summary)
            good = summary["valid_feature_frame_count"]
            entry["status"] = ("failed" if good == 0 else "partial"
                               if any(summary[f"{r}_success_count"] < summary["frame_records"]
                                      for r in VIDEO_REGIONS) else "ok")
            if not good:
                entry["error"] = "沒有影格成功取得任一 ROI 特徵"
            elif entry["status"] == "partial":
                entry["error"] = "部分影格或區域失敗，詳見 frame_features.csv"
        except Exception as error:
            # 只隔離單支影片的錯誤；Ctrl+C 等中斷仍照常傳遞
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"{entry['status']}：{entry['source_manifest']} {entry['error']}", flush=True)
        results.append(entry)
    fields = list(dict.fromkeys(("source_manifest", "output_dir", "status", "error", "video_id",
                                 "subject_id", "session_id", "frame_records", "valid_timestamp_count",
                                 "valid_feature_frame_count", "forehead_manual_quality") +
                                tuple(key for row in results for key in row)))
    _write_table(output / "batch_summary.csv", fields, results)
    print(f"批次結果：{dict(Counter(row['status'] for row in results))}", flush=True)
    return int(any(row["status"] == "failed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
