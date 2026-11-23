"""以原始影格中的固定灰色 ROI，比對同一支影片的雙頰 Lab 訊號。

不做色彩校正，不修改來源 CSV。灰塊只用於診斷拍攝條件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np


# 保留 left/right 命名以便比較舊結果。灰區低飽和時色相不穩，故不比 H。
CHANNELS = ("l", "a", "b", "s")
REGIONS = {"left": "image_left_cheek", "right": "image_right_cheek",
           "forehead": "image_forehead"}
SIGNALS = {(side, ch): f"{region}_{'s' if ch == 's' else 'lab_' + ch}_median"
           for side, region in REGIONS.items() for ch in CHANNELS}


def number(value):
    """有限數值才可配對；空欄不補零。"""
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read_features(path):
    """一個灰塊只對應一支影片；沿用 CSV 的影格編號與相對時間。"""
    required = {"video_id", "frame_index", "image_path", "relative_time_s", "timestamp_status",
                "delta_time_s", "previous_frame_index"} | set(SIGNALS.values()) | {
        name + suffix for name in SIGNALS.values() for suffix in ("_rate", "_rate_status")
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"請輸入影片版的 frame_features.csv，缺少欄位：{sorted(missing)}")
        rows = list(reader)
    if not rows or len({r["video_id"] for r in rows}) != 1 or not rows[0]["video_id"]:
        raise ValueError("CSV 必須只有一支影片，且包含影格資料")
    for key in ("session_id", "subject_id"):
        if len({r.get(key, "") for r in rows}) > 1:
            raise ValueError(f"CSV 混有不同的 {key}")
    return rows


def frame_path(root, name):
    """以 CSV 的 Windows 或 POSIX 檔名比對指定資料夾中的原始 PNG。"""
    name = PureWindowsPath(name or "").name
    if not name or Path(name).suffix.lower() != ".png":
        return None
    p = (root / name).resolve()
    return p if p.is_relative_to(root) else None


def read_png(path):
    """以位元組讀圖，支援 Windows 中文路徑。"""
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not buf.size:
        raise ValueError("空的 PNG")
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("PNG 解碼失敗")
    return image


def pick_roi(image):
    """用滑鼠在原始解析度的影格選擇灰塊，可用 --roi 略過視窗。"""
    title = "Select inside the gray patch, then press Enter"
    print("請框選灰塊內部，按 Enter 確認；避免邊緣、陰影、字樣和反光。", flush=True)
    try:
        roi = tuple(int(v) for v in cv2.selectROI(title, image, showCrosshair=True, fromCenter=False))
    except cv2.error as exc:
        raise ValueError("無法開啟選取視窗，請改用 --roi X Y W H") from exc
    finally:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass
    return roi


def measure(image, roi):
    """以與臉頰相同的浮點 OpenCV Lab 和曝光遮罩計算像素中位數。"""
    x, y, w, h = roi
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return {"gray_status": "roi_outside_frame"}
    rgb = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # HSV 的 V 是 RGB 最大通道，沿用臉頰的過暗／過曝規則。
    v = np.max(rgb, axis=2)
    valid = (v > 0.03) & (v < 0.98)
    count = int(valid.sum())
    result = {"gray_valid_pixel_ratio": count / valid.size, "gray_valid_pixels": count}
    if count < max(20, int(valid.size * 0.25)):
        return dict(result, gray_status="too_few_valid_pixels")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    for ch, val in zip(("l", "a", "b"), np.median(lab[valid], axis=0)):
        result[f"gray_{ch}"] = float(val)
    # 飽和度使用和臉部相同的浮點 HSV 與有效像素遮罩。
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    result["gray_s"] = float(np.median(saturation[valid]))
    return dict(result, gray_status="ok")


def get_gray_rate(previous, current, source, old_source):
    """以真實 PTS 差分。缺格、失敗、大間隔都不跨接。"""
    if previous is None:
        return {}, "first_frame"
    if any(source[k+"_rate_status"] == "large_time_gap" for k in SIGNALS.values()):
        return {}, "large_time_gap"
    if (previous["gray_status"] != "ok" or current["gray_status"] != "ok"):
        return {}, "missing_gray_pair"
    a, b = number(previous["frame_index"]), number(current["frame_index"])
    dt = current["relative_time_s"] - previous["relative_time_s"]
    saved_dt = number(source["delta_time_s"])
    if (a is None or b is None or b != a + 1
            or number(source["previous_frame_index"]) != a
            or old_source["timestamp_status"] != "ok" or dt <= 0 or saved_dt is None
            or not math.isclose(dt, saved_dt, rel_tol=1e-6, abs_tol=1e-9)):
        return {}, "invalid_time_pair"
    return {f"gray_{ch}_rate": (current[f"gray_{ch}"] - previous[f"gray_{ch}"]) / dt
            for ch in CHANNELS}, "ok"


def compare(rows, gray_key, cheek_key):
    """同一影格或同一影格對才納入 Pearson r；常數序列不給相關係數。"""
    pairs = [(r[gray_key], r[cheek_key]) for r in rows
             if r.get(gray_key) is not None and r.get(cheek_key) is not None]
    result = {"paired_count": len(pairs), "pearson_r": None}
    if pairs:
        a, b = np.asarray(pairs, dtype=float).T
        result.update(gray_std=float(a.std()), cheek_std=float(b.std()))
        if len(pairs) >= 3 and a.std() > 1e-9 and b.std() > 1e-9:
            result["pearson_r"] = float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))
        else:
            result["note"] = "too_few_pairs_or_constant_series"
    return result


def save_preview(path, image, roi):
    """用複本畫灰塊框，避免修改原始影格。"""
    x, y, w, h = roi
    preview = image.copy()
    cv2.rectangle(preview, (x, y), (x+w-1, y+h-1), (0, 255, 255), 2)
    ok, data = cv2.imencode(".png", preview)
    if not ok:
        raise OSError("無法編碼灰塊預覽")
    path.write_bytes(data.tobytes())


def run(feature_csv, root, output, roi):
    """固定座標量測全片；若解析度改變，該幀不再沿用此座標。"""
    source_rows = read_features(feature_csv)
    paths = [frame_path(root, r["image_path"]) for r in source_rows]
    counts = Counter(str(p).casefold() for p in paths if p is not None)
    reference = None
    for path, row in zip(paths, source_rows):
        if path is None or row["timestamp_status"] != "ok" or counts[str(path).casefold()] != 1:
            continue
        try:
            reference = read_png(path)
            break
        except (OSError, ValueError, cv2.error):
            continue
    if reference is None:
        raise ValueError("找不到可讀的原始影格，請確認 --frames-dir 指到 PNG 所在資料夾")
    if roi is None:
        roi = pick_roi(reference)
    x, y, w, h = roi
    if (x < 0 or y < 0 or w <= 0 or h <= 0 or w*h < 20
            or x+w > reference.shape[1] or y+h > reference.shape[0]):
        raise ValueError("灰塊框必須完整落在影格內，且至少有 20 個像素；若取消選取請重跑")

    output.mkdir(parents=True, exist_ok=True)
    previews = output / "roi_previews"
    previews.mkdir()
    results = []
    for i, (source, path) in enumerate(zip(source_rows, paths)):
        t = number(source["relative_time_s"])
        row = {"frame_index": source["frame_index"], "relative_time_s": t,
               "image_path": source["image_path"], "gray_status": "invalid_timestamp",
               "gray_error": "", "delta_time_s": number(source["delta_time_s"])}
        if source["timestamp_status"] == "ok" and t is not None and t >= 0:
            if path is None:
                row["gray_status"] = "invalid_image_path"
            elif counts[str(path).casefold()] > 1:
                row["gray_status"] = "duplicate_image_path"
            elif not path.is_file():
                row["gray_status"] = "missing_image"
            else:
                try:
                    image = read_png(path)
                    if image.shape[:2] != reference.shape[:2]:
                        row["gray_status"] = "image_size_changed"
                    else:
                        row.update(measure(image, roi))
                except (OSError, ValueError, cv2.error) as exc:
                    row.update(gray_status="image_read_failed", gray_error=str(exc))
        # 變化率只採用原特徵流程標記為有效的同一影格對。
        for (side, ch), name in SIGNALS.items():
            row[f"{side}_{ch}"] = number(source[name]) if source["timestamp_status"] == "ok" else None
            row[f"{side}_{ch}_rate"] = (number(source[name+"_rate"])
                                      if source[name+"_rate_status"] == "ok" else None)
        rates, status = get_gray_rate(results[-1] if i else None, row, source,
                                     source_rows[i-1] if i else None)
        row.update(rates, gray_rate_status=status)
        results.append(row)

    preview_index = {}
    # 依時間涵蓋選前中後；若抽到缺檔，保留錯誤，不以別張默默替換。
    valid = [i for i, r in enumerate(source_rows) if r["timestamp_status"] == "ok"
             and results[i]["relative_time_s"] is not None]
    if valid:
        for label, i in zip(("start", "middle", "end"),
                            (valid[0], valid[len(valid)//2], valid[-1])):
            name = f"{label}_frame_{results[i]['frame_index']}.png"
            entry = {"frame_index": results[i]["frame_index"], "status": results[i]["gray_status"]}
            try:
                if results[i]["gray_status"] not in {"ok", "too_few_valid_pixels"}:
                    raise ValueError(results[i]["gray_status"])
                save_preview(previews / name, read_png(paths[i]), roi)
                entry["file"] = f"roi_previews/{name}"
            except (OSError, ValueError, cv2.error) as exc:
                entry["error"] = str(exc)
            preview_index[label] = entry

    fields = ["frame_index", "relative_time_s", "image_path", "gray_status", "gray_error",
              "gray_valid_pixels", "gray_valid_pixel_ratio", "delta_time_s", "gray_rate_status"]
    fields += [f"gray_{ch}{suffix}" for ch in CHANNELS for suffix in ("", "_rate")]
    fields += [f"{side}_{ch}{suffix}" for side, ch in SIGNALS for suffix in ("", "_rate")]
    with (output / "gray_reference_frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "video_id": source_rows[0]["video_id"], "roi_xywh": list(roi),
        "frame_records": len(results), "gray_status_counts": dict(Counter(r["gray_status"] for r in results)),
        "gray_rate_status_counts": dict(Counter(r["gray_rate_status"] for r in results)),
        "preview_index": preview_index,
        "interpretation": "請先確認灰塊位置穩定。相關係數僅描述同影片同步性，不能判定因果。",
        "comparisons": {f"{side}_{ch}_{label}": compare(results, f"gray_{ch}{suffix}", f"{side}_{ch}{suffix}")
                        for side, ch in SIGNALS for label, suffix in (("level", ""), ("rate", "_rate"))},
    }
    (output / "gray_reference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"灰塊座標 x y w h：{x} {y} {w} {h}")
    print(f"灰塊狀態：{summary['gray_status_counts']}")
    for key, value in summary["comparisons"].items():
        r = value["pearson_r"]
        print(f"{key}: r={r:.3f}, n={value['paired_count']}" if r is not None
              else f"{key}: r=無法估計, n={value['paired_count']}")
    return 0 if summary["gray_status_counts"].get("ok") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_features_csv", type=Path, help="單支影片的 frame_features.csv")
    parser.add_argument("--frames-dir", type=Path, required=True, help="原始 PNG 所在資料夾")
    parser.add_argument("--output", type=Path, required=True, help="新或空的輸出資料夾")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="原始影格上的左上角、寬、高；不填就用滑鼠框選")
    args = parser.parse_args()
    if not args.frames_dir.is_dir():
        parser.error(f"找不到原始影格資料夾：{args.frames_dir}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("請指定新的或空的輸出資料夾")
    try:
        return run(args.frame_features_csv, args.frames_dir.resolve(), args.output, args.roi)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
