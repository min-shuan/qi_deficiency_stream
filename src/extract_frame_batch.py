import argparse
import csv
import uuid
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np

FIELDS = [
    'session_id',       # 本次處理的唯一識別碼
    'subject_id',       # 受測者識別碼
    'source_video',     
    'frame_index',
    'pts', 
    'time_base',        # PTS 時間單位，將 PTS 轉成秒
    'pts_seconds', 
    'relative_time_s',  # 相對第一個有效 PTS 的時間，表示影格在擷取區間內
    'delta_time_s',     # 與商一張影格的時間差，分析影格時間間隔
    'timestamp_source', 
    'timestamp_valid',  # 時間戳記是否有效，PTS 與 time_base 都存在且 PTS 嚴格遞增
    'image_path', 
    'width',            # 影格寬度
    'height',           # 影格高度
    'duplicate_status', # 檢查重複影格，僅判斷相鄰解碼影格像素是否完全相同
    'drop_status', 
    'blur_score',       # 模糊檢測數值
    'possible_blur',    # 疑似模糊，需人工確認
    'frame_status',     # 影格處理狀態，包含時間戳記與 PNG 儲存結果
    'png_saved',        # PNG 是否成功儲存
    'save_error'        # PNG 儲存失敗的錯誤訊息
]

ISSUE_FIELDS = ['session_id', 
                'subject_id', 
                'frame_index', 
                'relative_time_s',
                'issue_type', 
                'severity', 
                'description']


def save_png(path, image, compression):
    try:
        ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not ok:
            return False, 'png_encode_failed'
        path.write_bytes(encoded.tobytes())
        return True, ''
    except Exception as exc:
        return False, str(exc)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_video_duration(source, minimum_seconds=15.0):
    """先解碼確認視訊涵蓋時間；不使用音訊或容器總長度判定。"""
    first = previous = None
    last_duration = None
    intervals = []
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                return None, '無可用的視訊串流，請受測者重新拍攝'
            for frame in container.decode(container.streams.video[0]):
                if frame.pts is None or frame.time_base is None:
                    return None, '視訊時間戳記缺失，無法確認滿 15 秒，請受測者重新拍攝'
                current = frame.pts * frame.time_base
                if first is None:
                    first = current
                if previous is not None:
                    if current <= previous:
                        return None, '視訊時間戳記未遞增，無法確認時長，請受測者重新拍攝'
                    intervals.append(current - previous)
                # 已解碼到第 15 秒的影格，便能確認擷取區間完整。
                if current - first >= minimum_seconds:
                    return float(current - first), ''
                previous = current
                duration = getattr(frame, 'duration', None)
                last_duration = duration * frame.time_base if duration and duration > 0 else None
    except Exception as exc:
        return None, f'影片驗證失敗（{type(exc).__name__}: {exc}），請受測者重新拍攝'
    if first is None:
        return None, '影片沒有可解碼影格，請受測者重新拍攝'
    # 最後一張影格的 PTS 是開始時間，需加上顯示期間。
    # WebM 若未提供 frame.duration，使用最後一段實際 PTS 間隔估計，不能用 FPS * 15。
    tail = last_duration if last_duration is not None else (intervals[-1] if intervals else None)
    if tail is None:
        return None, '無法確認最後影格的顯示期間，請受測者重新拍攝'
    duration_s = float(previous - first + tail)
    if duration_s + 1e-9 < minimum_seconds:
        return duration_s, f'影片不足 15 秒（視訊時長約 {duration_s:.6f} 秒），不採用，請受測者重新拍攝'
    return duration_s, ''


def extract_video(source, output_root, compression, workers, black_threshold, gap_factor, blur_threshold):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f'找不到影片：{source}')
    if source.suffix.lower() != '.webm':
        raise ValueError('目前僅接受 .webm 影片')
    subject_id, video_name = source.parent.name, source.stem
    # 識別碼用於關聯本次處理的紀錄。
    session_id = str(uuid.uuid4())
    folder = Path(output_root) / video_name
    frames_dir = folder / 'frames'
    if folder.exists():
        raise FileExistsError(f'輸出資料夾已存在，為避免覆寫請先確認：{folder}')

    start = time.perf_counter()
    rows, issues, pending = [], [], {}
    fps = None
    first_time = previous_time = None
    previous_frame_has_valid_pts = False  # 僅允許相鄰且有效的影格計算 delta
    previous_image = None
    previous_shape = None
    intervals = []
    expected_duration = None
    decode_error = ''
    completed = False
    output_created = False
    stopped_at_15s = False
    extraction_window_s = 15.0

    def add_issue(index, relative, kind, severity, description):
        issues.append({
            'session_id': session_id, 'subject_id': subject_id,
            'frame_index': index, 'relative_time_s': relative,
            'issue_type': kind, 'severity': severity, 'description': description
        })

    validated_duration_s, rejection_reason = check_video_duration(source, extraction_window_s)
    if rejection_reason:
        # 不合格影片只保留拒收紀錄，不輸出影格，避免下一階段誤用。
        folder.mkdir(parents=True, exist_ok=False)
        add_issue('', '', 'video_rejected', 'error', rejection_reason)
        write_csv(folder / 'capture_issues.csv', ISSUE_FIELDS, issues)
        summary = {
            'session_id': session_id, 'subject_id': subject_id,
            'source_video': str(source), 'video_name': source.name,
            'processed_at_utc': datetime.now(timezone.utc).isoformat(),
            'accepted': False, 'requires_recapture': True,
            'rejection_reason': rejection_reason,
            'validated_video_duration_s': validated_duration_s,
            'minimum_duration_s': extraction_window_s,
            'decoded_frames': 0, 'saved_png': 0,
            'duration_note': '依視訊 PTS 與最後影格顯示期間；缺少 duration 時以最後 PTS 間隔估計',
        }
        with (folder / 'session.json').open('w', encoding='utf-8') as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        print(f'\n不採用：{source}\n{rejection_reason}\n拒收紀錄：{folder}', flush=True)
        return summary

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise RuntimeError('影片沒有可用的視訊串流')
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            # 容器 duration 可能不可靠，僅作完整性輔助參考。
            if container.duration is not None:
                expected_duration = float(container.duration) / 1_000_000
            frames_dir.mkdir(parents=True, exist_ok=False)
            output_created = True
            print(f'來源影片：{source}', flush=True)
            print(f'輸出位置：{folder}', flush=True)
            print(f'影片 FPS：{fps:.3f}' if fps is not None else '影片 FPS：無法取得', flush=True)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    for index, frame in enumerate(container.decode(stream), start=1):
                        # 缺少 PTS 時仍保存影格，不以 FPS 補造時間戳記。
                        has_pts = frame.pts is not None and frame.time_base is not None
                        pts_time = frame.pts * frame.time_base if has_pts else None
                        # 使用影片內逐幀 PTS，從第一個有 PTS 的影格起算前 15 秒。
                        # 不用回報 FPS * 15 推算影格數，以免 VFR 影片擷取錯誤。
                        if (first_time is not None and pts_time is not None
                                and pts_time - first_time >= extraction_window_s):
                            stopped_at_15s = True
                            break
                        relative = None
                        delta = None
                        pts_valid = has_pts

                        if has_pts:
                            # 相對時間以第一個有 PTS 的解碼影格為零點。
                            if first_time is None:
                                first_time = pts_time
                            relative = pts_time - first_time

                            # 只在「上一張影格」也有有效 PTS 時，計算時間差。
                            if previous_frame_has_valid_pts:
                                delta = pts_time - previous_time
                                if delta <= 0:
                                    pts_valid = False
                                    delta = None
                                    # 判斷 PTS 是否嚴格遞增
                                    add_issue(index, round(float(relative), 9),
                                              'non_increasing_pts', 'error',
                                              'PTS 未嚴格遞增，該幀及下一幀不能計算相鄰時間差')
                        else:
                            # 判斷 PTS 是否缺失
                            add_issue(index, '', 'missing_pts', 'error',
                                      '影格缺少 PTS 或 time_base，該幀及下一幀不能計算相鄰時間差')

                        # 時間戳異常時不提供相對時間與時間差，避免後續分析誤用。
                        # 原始 pts / time_base / pts_seconds 仍保留供問題追溯。
                        if not pts_valid:
                            relative = None
                            delta = None

                        # 無效時間戳會切斷連續區間，避免跨幀微分。
                        previous_frame_has_valid_pts = pts_valid
                        if pts_valid:
                            if delta is not None:
                                intervals.append(float(delta))
                            previous_time = pts_time

                        image = frame.to_ndarray(format='bgr24')
                        height, width = image.shape[:2]
                        duplicate = 'not_applicable' if previous_image is None else (
                            'identical_pixels' if image.shape == previous_image.shape and np.array_equal(image, previous_image)
                            else 'different_pixels'
                        )
                        # 判斷重複影格
                        if duplicate == 'identical_pixels':
                            add_issue(index, '' if relative is None else round(float(relative), 9),
                                      'identical_consecutive_frame', 'warning', '相鄰解碼影格像素完全相同；無法判定是否為來源重複')
                        # 判斷解析度變更，可能是影片中途切換解析度或解碼異常。
                        if previous_shape is not None and image.shape != previous_shape:
                            add_issue(index, '' if relative is None else round(float(relative), 9),
                                      'resolution_changed', 'warning', f'尺寸由 {previous_shape[:2]} 改為 {image.shape[:2]}')
                        # 低平均亮度是「疑似黑畫面」，不是確定拍攝失敗。
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        if float(np.mean(gray)) <= black_threshold:
                            add_issue(index, '' if relative is None else round(float(relative), 9),
                                      'possible_black_frame', 'warning', '影格平均灰階亮度過低，疑似黑畫面')
                        # Laplacian 變異數僅作為疑似模糊指標，不刪除影格。
                        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        possible_blur = blur_score < blur_threshold
                        if possible_blur:
                            add_issue(index, '' if relative is None else round(float(relative), 9),
                                      'possible_blur', 'warning',
                                      f'Laplacian 變異數 {blur_score:.3f} 低於門檻 {blur_threshold:.3f}；疑似模糊，需人工確認')
                        previous_image, previous_shape = image, image.shape

                        image_path = frames_dir / f'{video_name}_{index:06d}.png'
                        rows.append({
                            'session_id': session_id, 'subject_id': subject_id,
                            'source_video': str(source), 'frame_index': index,
                            'pts': frame.pts if frame.pts is not None else '',
                            'time_base': str(frame.time_base) if frame.time_base is not None else '',
                            'pts_seconds': round(float(pts_time), 9) if pts_time is not None else '',
                            'relative_time_s': round(float(relative), 9) if relative is not None else '',
                            'delta_time_s': round(float(delta), 9) if pts_valid and delta is not None else '',
                            'timestamp_source': 'webm_frame_pts' if pts_time is not None else 'unavailable',
                            'timestamp_valid': pts_valid,
                            'image_path': str(image_path.relative_to(folder)),
                            'width': width, 'height': height,
                            'duplicate_status': duplicate,
                            'drop_status': 'unknown',  # 沒有來源序號，不能確認錄影端是否漏幀。
                            'blur_score': round(blur_score, 6), 'possible_blur': possible_blur,
                            'frame_status': 'ok' if pts_valid else 'timestamp_invalid',
                            'png_saved': False, 'save_error': '',
                        })
                        pending[pool.submit(save_png, image_path, image, compression)] = len(rows) - 1
                        if len(pending) >= workers * 3:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in done:
                                row_index = pending.pop(future)
                                rows[row_index]['png_saved'], rows[row_index]['save_error'] = future.result()
                    completed = True
                except Exception as exc:
                    decode_error = f'{type(exc).__name__}: {exc}'
                    add_issue(len(rows) + 1, '', 'decode_interrupted', 'error', decode_error)
                finally:
                    # 解碼失敗仍等待已提交的 PNG 完成，確保 CSV 可反映實際結果。
                    for future, row_index in pending.items():
                        try:
                            rows[row_index]['png_saved'], rows[row_index]['save_error'] = future.result()
                        except Exception as exc:
                            rows[row_index]['png_saved'], rows[row_index]['save_error'] = False, str(exc)
    except Exception as exc:
        if not output_created:
            raise
        if not decode_error:
            decode_error = f'{type(exc).__name__}: {exc}'
            add_issue(len(rows) + 1, '', 'processing_interrupted', 'error', decode_error)

    if not output_created:
        raise RuntimeError('尚未建立輸出資料夾')

    if not rows:
        add_issue('', '', 'no_decoded_frames', 'error', '沒有成功解碼任何影格')

    # 使用有效 PTS 間隔中位數判斷「異常間隔」；不能據此確認漏幀。
    if intervals:
        sorted_intervals = sorted(intervals)
        median = sorted_intervals[len(sorted_intervals) // 2]
        if median > 0:
            for row in rows:
                dt = row['delta_time_s']
                # 判斷 PTS 間隔是否大於中位數的 gap_factor 倍，僅作為疑似漏幀提示。
                if isinstance(dt, (float, int)) and dt > median * gap_factor:
                    add_issue(row['frame_index'], row['relative_time_s'], 'large_pts_gap', 'warning',
                              f'時間間隔 {dt:.6f}s 大於中位間隔 {median:.6f}s 的 {gap_factor} 倍；無法確認漏幀')
                    row['drop_status'] = 'possible_gap_not_confirmed'

    # 若容器長度有資訊，僅對「明顯提前結束」提出疑點，不把正常 EOF 誤判為中斷。
    observed_span = float(previous_time - first_time) if first_time is not None and previous_time is not None else None
    if completed and not stopped_at_15s and expected_duration and observed_span is not None and observed_span + 1.0 < expected_duration:
        add_issue('', '', 'possible_incomplete_video', 'warning',
                  f'PTS 首尾跨度 {observed_span:.3f}s，小於容器標示長度 {expected_duration:.3f}s；需人工確認')

    for row in rows:
        if not row['png_saved']:
            # 不覆蓋既有的 timestamp_invalid；同一影格可同時有兩種異常。
            if row['frame_status'] == 'ok':
                row['frame_status'] = 'png_save_failed'
            else:
                row['frame_status'] += '|png_save_failed'
            add_issue(row['frame_index'], row['relative_time_s'], 'png_save_failed', 'error', row['save_error'])

    failures = [r for r in rows if not r['png_saved']]
    write_csv(folder / 'frames.csv', FIELDS, rows)
    write_csv(folder / 'save_errors.csv', FIELDS, failures)
    write_csv(folder / 'capture_issues.csv', ISSUE_FIELDS, issues)

    summary = {
        'session_id': session_id, 'subject_id': subject_id,
        'accepted': not bool(decode_error or failures) and bool(rows),
        'requires_recapture': bool(decode_error or failures) or not bool(rows),
        'validated_video_duration_s': validated_duration_s,
        'minimum_duration_s': extraction_window_s,
        'duration_note': '依視訊 PTS 與最後影格顯示期間；缺少 duration 時以最後 PTS 間隔估計',
        'source_video': str(source), 'video_name': source.name,
        'output_root': str(output_root), 'processed_at_utc': datetime.now(timezone.utc).isoformat(),
        'reported_average_fps': fps, 'container_duration_s': expected_duration,
        'blur_threshold': blur_threshold,
        'blur_metric': 'variance_of_laplacian_grayscale',
        'blur_note': '低於門檻僅表示疑似模糊；門檻需依實際鏡頭與影像內容校準',
        'extraction_window_s': extraction_window_s, 'stopped_at_15s': stopped_at_15s,
        'decoded_frames': len(rows), 'saved_png': len(rows) - len(failures),
        'failed_png': len(failures), 'issue_count': len(issues),
        'decode_completed': completed, 'decode_error': decode_error or None,
        'timestamp_source': 'webm_frame_pts',
        'timestamp_note': '相對第一個有 PTS 的解碼影格；PTS 不等於已驗證的攝影機曝光時間',
        'relative_time_origin': 'first_decoded_frame_with_valid_webm_pts',
        'relative_time_origin_pts_seconds': float(first_time) if first_time is not None else None,
        'measurement_start_time_source': 'unknown_not_recorded',
        'measurement_start_time_note': '未取得 Kebbi 錄影端的量測開始時間；relative_time_s 不是已驗證的相對實際量測開始時間',
        'delta_time_note': '僅在當前與緊鄰上一張解碼影格的 PTS 都有效且嚴格遞增時提供；否則留空；無效 PTS 的 relative_time_s 亦留空',
        'drop_detection_note': '來源端沒有逐幀序號，漏幀狀態無法確定；large_pts_gap 僅為疑似',
        'duplicate_detection_note': 'identical_pixels 僅代表相鄰解碼畫面像素相同',
    }
    # 先寫入摘要，再更新計時（最後一次 JSON 寫入耗時不含於數值內）。
    summary['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    with (folder / 'session.json').open('w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start
    print('\n========== 處理結果 ==========')
    print(f'提取影格：{len(rows)} 張')
    print(f'成功：{len(rows) - len(failures)} 張')
    print(f'失敗：{len(failures)} 張')
    print(f'總處理時間：約 {elapsed:.3f} 秒')
    print('\n輸出檔案：')
    print(f'  frames.csv         - 所有影格的時間、路徑及狀態（{len(rows)} 筆）')
    print(f'  save_errors.csv    - PNG 儲存失敗紀錄（{len(failures)} 筆）')
    print(f'  capture_issues.csv - 時間、影像與解碼異常紀錄（{len(issues)} 筆）')
    print('  session.json       - 量測來源、識別碼及處理摘要')
    print(f'\n輸出資料夾：{folder}')
    if decode_error:
        print(f'警告：{decode_error}')
    return summary


def main():
    parser = argparse.ArgumentParser(description='逐影格 PNG、PTS 與錯誤紀錄')
    parser.add_argument('--source', required=True, help='影片根目錄，例如 videos（讀取直屬影片及各 ID 子資料夾內的影片）')
    parser.add_argument('--output', default='output_extract_frame', help='輸出根目錄')
    parser.add_argument('--compression', type=int, choices=range(10), metavar='0-9', default=1)
    parser.add_argument('--workers', type=int, default=4, help='PNG 平行寫入數')
    parser.add_argument('--black-threshold', type=float, default=5.0, help='疑似黑畫面平均灰階門檻（0-255）')
    parser.add_argument('--gap-factor', type=float, default=2.5, help='疑似 PTS 大間隔門檻倍率')
    parser.add_argument('--blur-threshold', type=float, default=80.0, help='疑似模糊 Laplacian 變異數門檻（預設 80，需依鏡頭校準）')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers 必須至少為 1')
    if not 0 <= args.black_threshold <= 255 or args.gap_factor <= 1 or not np.isfinite(args.blur_threshold) or args.blur_threshold < 0:
        parser.error('黑畫面門檻須在 0~255，gap-factor 必須大於 1，blur-threshold 必須是非負有限數值')
    source_dir = Path(args.source)
    if not source_dir.is_dir():
        parser.error(f'找不到指定資料夾：{source_dir}')

    # 支援 videos/ID/影片，也保留直接指定單一 ID 資料夾的用法。
    candidates = list(source_dir.iterdir())
    for subject_dir in sorted(p for p in candidates if p.is_dir()):
        candidates.extend(subject_dir.iterdir())
    videos = sorted(p for p in candidates if p.is_file() and p.suffix.lower() == '.webm')
    if not videos:
        parser.error(f'指定資料夾及其下一層找不到 WebM 影片：{source_dir}')

    start = time.perf_counter()
    accepted = rejected = failed = skipped = 0
    for index, video in enumerate(videos, start=1):
        print(f'\n[{index}/{len(videos)}] {video}', flush=True)
        output_folder = Path(args.output) / video.stem
        if output_folder.exists():
            skipped += 1
            print(f'略過：輸出位置已存在：{output_folder}', flush=True)
            continue
        try:
            summary = extract_video(video, args.output, args.compression, args.workers,
                                    args.black_threshold, args.gap_factor, args.blur_threshold)
        except Exception as exc:
            failed += 1
            print(f'處理失敗：{video}（{type(exc).__name__}: {exc}）', flush=True)
            continue

        if summary['accepted']:
            accepted += 1
        elif 'rejection_reason' in summary:
            rejected += 1
        else:
            failed += 1
            print(f'處理未完成：{video}，請查閱 {output_folder / "session.json"}', flush=True)

    elapsed = time.perf_counter() - start
    print('\n========== 批次處理結果 ==========')
    print(f'找到 {len(videos)} 部｜採用 {accepted}｜拒收 {rejected}｜失敗 {failed}｜略過 {skipped}')
    print(f'批次總處理時間：約 {elapsed:.3f} 秒')


if __name__ == '__main__':
    main()
