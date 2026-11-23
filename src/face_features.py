"""額頭與雙頰色彩特徵提取，支援單張圖片 ROI 預覽。

The extractor intentionally produces interpretable handcrafted features.  It is
not a diagnostic system; model training belongs in a separate stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

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


# MediaPipe 中左右對稱的臉頰區域點。
# 後續會依實際 x 座標重新判斷圖片左側與圖片右側，
# 因此不直接依賴解剖學上的左右名稱。
CHEEK_REGIONS = (
    CheekRegion("cheek_a", (205,), (46, 204, 113)),
    CheekRegion("cheek_b", (425,), (46, 204, 113)),
)

# ROI 尺寸相對於雙眼距離的比例。
CHEEK_WIDTH_RATIO = 0.50
CHEEK_HEIGHT_RATIO = 0.34

# 負值代表將 landmark 中心稍微往上移。
CHEEK_Y_OFFSET_RATIO = -0.15

# MediaPipe 兩側眼睛輪廓點，用來旋正人臉及計算雙眼距離。
EYE_A = (33, 133, 159, 145)
EYE_B = (362, 263, 386, 374)

# 額頭採幾何定位初始值，尚未做頭髮分割，需以原始圖片人工確認。
# 使用臉部上緣點與雙眼中點之間的位置，避免直接在臉部上緣取樣。
FOREHEAD_TOP_LANDMARK = 10
FOREHEAD_CENTER_FRACTION = 0.45  # 0 靠近上緣，1 靠近雙眼。
FOREHEAD_WIDTH_RATIO = 0.50      # 寬度相對於雙眼距離。
FOREHEAD_HEIGHT_RATIO = 0.28     # 高度相對於雙眼距離。
FOREHEAD_MAX_GAP_RATIO = 0.45    # 高度不超過上緣至雙眼垂直距離的此比例。
FOREHEAD_COLOR = (255, 180, 0)   # OpenCV BGR，與雙頰綠框區分。


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

    # 排除亮度太低或過度曝光的像素。先保留既有門檻，並輸出比例供 QC 判讀。
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

    h, s, v = (hsv[..., channel][valid] for channel in range(3))
    l_star, a_star, b_star = (lab[..., channel][valid] for channel in range(3))
    hue_mean, _ = _circular_hue_stats(h, s)
    features = {
        f"{prefix}_hue_circular_mean": hue_mean,                 # Hue圓型平均
        f"{prefix}_s_median": float(np.median(s)),               # S中位數
        f"{prefix}_v_median": float(np.median(v)),               # V中位數
        f"{prefix}_lab_l_median": float(np.median(l_star)),      # L*中位數
        f"{prefix}_lab_a_median": float(np.median(a_star)),      # a*中位數
        f"{prefix}_lab_b_median": float(np.median(b_star)),      # b*中位數
        f"qc_{prefix}_valid_pixel_ratio": float(np.mean(valid)), # 有效比例QC (Quality控制，)不會送入模型
    }

    # 僅新增分布特徵；沿用上述有效像素，保留既有中位數、Hue 與 QC 計算。
    # 使用 float64 計算，標準差採母體標準差 (ddof=0)。
    # S、V 維持 0–1；Lab 維持 OpenCV 浮點 CIELAB 尺度。
    for channel_name, values in (
        ("s", s), ("v", v), ("lab_l", l_star),
        ("lab_a", a_star), ("lab_b", b_star),
    ):
        values = values.astype(np.float64)
        p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
        features[f"{prefix}_{channel_name}_std"] = float(np.std(values, ddof=0))
        features[f"{prefix}_{channel_name}_iqr"] = float(p75 - p25)
        features[f"{prefix}_{channel_name}_p90_p10"] = float(p90 - p10)

    return features


#使用中位數，不容易被其他因素影響
# return 會是 [55.3, 41.2,20.7]
def _median_lab(features: Mapping[str, object], prefix: str) -> np.ndarray:
    return np.array([
        float(features[f"{prefix}_lab_l_median"]),
        float(features[f"{prefix}_lab_a_median"]),
        float(features[f"{prefix}_lab_b_median"]),
    ], dtype=np.float64)



class FaceFeatureExtractor:
    """Detect one face and extract fixed regional colour features."""

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

        # 不依賴自拍鏡像或受試者解剖左右，直接按照圖片中的 x 座標排序。
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

    # 串接所有步驟，回傳:特徵csv檔及ROI預覽圖片
    def extract(self, image: np.ndarray, image_id: str) -> tuple[dict[str, float | str], np.ndarray]:
        if image is None or image.size == 0: #檢查
            raise FeatureExtractionError("Image is empty")

        aligned, landmarks = self._align(image)
        preview = aligned.copy()
        features: dict[str, float | str] = {"image_id": image_id}
        cheek_boxes = self._cheek_boxes(landmarks, aligned.shape)

        # 保留原有雙頰設定，僅在其後加入額頭；三區共用色彩統計與 QC。
        try:
            cheek_boxes.append(self._forehead_box(landmarks, aligned.shape))
        except FeatureExtractionError as error:
            error.preview = self._draw_regions(aligned, cheek_boxes)
            raise

        # 先畫出三區 ROI；即使後續特徵失敗，也能輸出失敗預覽供人工檢查。
        for region_name, box, color in cheek_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview,
                region_name,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        for region_name, box, color in cheek_boxes:
            x1, y1, x2, y2 = box
            roi = aligned[y1:y2, x1:x2]
            try:
                features.update(colour_features(roi, region_name))
            except FeatureExtractionError as error:
                error.preview = preview
                raise

        #計算左右臉頰差異
        left_cheek = _median_lab(features, "image_left_cheek")
        right_cheek = _median_lab(features, "image_right_cheek")
        cheek_delta = left_cheek - right_cheek
        features["cheek_delta_l"] = float(cheek_delta[0])
        features["cheek_delta_a"] = float(cheek_delta[1])
        features["cheek_delta_b"] = float(cheek_delta[2])
        features["cheek_delta_e76"] = float(np.linalg.norm(cheek_delta))
        return features, preview


# 讀取圖片(path要先轉成字串再交給openCV)
def readable_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FeatureExtractionError(f"Cannot read image: {path}")
    return image


def image_paths(path: Path) -> Iterable[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FeatureExtractionError(f"Input does not exist: {path}")
    yield from (item for item in sorted(path.iterdir()) if item.suffix.lower() in extensions)


def _preview_cli() -> int:
    """直接執行此檔時，只輸出 ROI 預覽與定位紀錄，不產生特徵表。

    範例：python face_features.py ./sample_images --output ./roi_check
    輸入資料夾只讀取該層圖片，避免誤讀既有預覽與其他處理版本。
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="單張圖片三區 ROI 定位檢查，不提取特徵")
    parser.add_argument("input", type=Path, help="圖片或只含待檢查圖片的資料夾")
    parser.add_argument("--output", type=Path, required=True, help="新的預覽輸出資料夾")
    args = parser.parse_args()
    # 禁止混用舊輸出，避免殘留圖片與本次結果混淆。
    if args.output.exists() and not args.output.is_dir():
        parser.error("輸出路徑必須是資料夾")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("輸出資料夾已有內容，請另選空資料夾")
    paths = list(image_paths(args.input))
    if not paths:
        parser.error("沒有可讀取的圖片")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with FaceFeatureExtractor() as extractor:
        for index, path in enumerate(paths, start=1):
            try:
                preview, boxes = extractor.preview_regions(readable_image(path))
                # 保留原副檔名以區分同名的 PNG/JPG，預覽另存為 PNG。
                target = args.output / f"{path.name}_roi.png"
                if not cv2.imwrite(str(target), preview):
                    raise OSError(f"無法寫入預覽：{target}")
                records.append({"input_path": str(path), "status": "located",
                                "preview_path": str(target), "regions": [
                                    {"name": name, "box_xyxy": list(box)} for name, box, _ in boxes]})
                print(f"[{index}/{len(paths)}] 定位完成，待人工確認：{path.name}", flush=True)
            except (FeatureExtractionError, OSError) as error:
                record = {"input_path": str(path), "status": "failed", "error": str(error)}
                if isinstance(error, FeatureExtractionError) and error.preview is not None:
                    target = args.output / f"{path.name}_failed_roi.png"
                    if cv2.imwrite(str(target), error.preview):
                        record["preview_path"] = str(target)
                records.append(record)
                print(f"[{index}/{len(paths)}] 失敗：{path.name}｜{error}", flush=True)
    (args.output / "roi_review.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    count = sum(row["status"] == "located" for row in records)
    print(f"定位完成 {count}/{len(paths)} 張；位置與頭髮遮擋仍需人工確認。", flush=True)
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(_preview_cli())
