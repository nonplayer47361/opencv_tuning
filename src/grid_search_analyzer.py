# -*- coding: utf-8 -*-

# 1. 필요한 라이브러리들을 모두 임포트합니다.
import os
import cv2
import numpy as np
import pandas as pd
import json
import itertools
import time
from datetime import datetime
from pathlib import Path
from skimage.feature import hessian_matrix_det
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

# 2. 기본 설정
TOP_N = 5
CHECKPOINT_INTERVAL = 100
CHUNK_SIZE = 5000

# ==================================================================
# ⭐⭐⭐ 전체 실험 계획 제어판 (EXPERIMENTAL_PLAN) ⭐⭐⭐
# ==================================================================
EXPERIMENTAL_PLAN = [
    # 1. Baseline
    {'experiment_name': 'Baseline_Tophat', 'use_msrcr': False, 'enhancement_filter': 'tophat', 'base_pipeline': ['clahe', 'blur', 'ENHANCER_SLOT', 'threshold']},
    # 2. Tophat -> DoG로 대체
    {'experiment_name': 'Replace_with_DoG', 'use_msrcr': False, 'enhancement_filter': 'dog', 'base_pipeline': ['clahe', 'blur', 'ENHANCER_SLOT', 'threshold']},
    # 3. Tophat -> Hessian으로 대체
    {'experiment_name': 'Replace_with_Hessian', 'use_msrcr': False, 'enhancement_filter': 'hessian', 'base_pipeline': ['clahe', 'blur', 'ENHANCER_SLOT', 'threshold']},
    # 4. Tophat -> Blackhat으로 대체
    {'experiment_name': 'Replace_with_Blackhat', 'use_msrcr': False, 'enhancement_filter': 'blackhat', 'base_pipeline': ['clahe', 'blur', 'ENHANCER_SLOT', 'threshold']},
    # 5. 특징 강화 필터 없음
    {'experiment_name': 'Baseline_No_Enhancer', 'use_msrcr': False, 'enhancement_filter': None, 'base_pipeline': ['clahe', 'blur', 'ENHANCER_SLOT', 'threshold']},
    # 6. 파이프라인 순서 변경
    {'experiment_name': 'Order_Test_Blur_First', 'use_msrcr': False, 'enhancement_filter': 'tophat', 'base_pipeline': ['blur', 'clahe', 'ENHANCER_SLOT', 'threshold']},
    # 7. MSRCR 전처리기 추가
    {'experiment_name': 'Add_MSRCR_with_Tophat_NoCLAHE', 'use_msrcr': True, 'enhancement_filter': 'tophat', 'base_pipeline': ['blur', 'ENHANCER_SLOT', 'threshold']},
    # 8. MSRCR + Hessian 조합
    {'experiment_name': 'Add_MSRCR_with_Hessian', 'use_msrcr': True, 'enhancement_filter': 'hessian', 'base_pipeline': ['blur', 'ENHANCER_SLOT', 'threshold']},
]

# -------------------------------------------------------------------
# Part 1: 헬퍼 함수 및 필터 구현부
# (이전과 동일)
# -------------------------------------------------------------------
def compute_accuracy(detected, ground, tolerance=20.0):
    matched = 0
    if not detected or not ground: return 0, 0, 0, 0
    detected_matched = [False] * len(detected)
    for g_pt in ground:
        for i, d_pt in enumerate(detected):
            if not detected_matched[i] and np.linalg.norm(np.array(g_pt) - np.array(d_pt)) <= tolerance:
                matched += 1; detected_matched[i] = True; break
    precision = 100.0 * matched / len(detected) if detected else 0
    recall = 100.0 * matched / len(ground) if ground else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, matched

# (이하 모든 필터 및 헬퍼 함수는 이전과 동일하게 여기에 위치합니다)
def _single_scale_retinex(img, sigma):
    return np.log1p(img) - np.log1p(cv2.GaussianBlur(img, (0, 0), sigma))
def apply_msrcr(img, params):
    img_float = img_as_float(img); img_retinex = np.zeros_like(img_float)
    for i in range(img_float.shape[2]):
        channel = img_float[:, :, i]; retinex_channel = np.zeros_like(channel)
        for sigma in params['msrcr_sigmas']: retinex_channel += _single_scale_retinex(channel, sigma)
        img_retinex[:, :, i] = retinex_channel / len(params['msrcr_sigmas'])
    gray_sum = np.sum(img_float, axis=2) / 3
    for i in range(img_float.shape[2]):
        intensity_log = np.log1p(1 + 125 * img_float[:, :, i]); gray_sum_log = np.log1p(1 + 125 * gray_sum)
        img_retinex[:, :, i] = intensity_log - gray_sum_log
    img_retinex = (img_retinex - np.min(img_retinex)) / (np.max(img_retinex) - np.min(img_retinex) + 1e-9)
    return (img_retinex * 255).astype(np.uint8)
def apply_clahe(img, params):
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
    return clahe.apply(img)
def apply_bilateral_blur(img, params):
    return cv2.bilateralFilter(img, d=9, sigmaColor=params['sigmaColor'], sigmaSpace=params['sigmaSpace'])
def apply_adaptive_threshold(img, params):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params['blockSize'], params['cVal'])
def apply_tophat(img, params):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morphK'], params['morphK']))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def apply_blackhat(img, params):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morphK'], params['morphK']))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
def apply_dog(img, params):
    blur1 = cv2.GaussianBlur(img, (0, 0), params['dog_sigma1']); blur2 = cv2.GaussianBlur(img, (0, 0), params['dog_sigma2'])
    return cv2.subtract(blur1, blur2)
def apply_hessian(img, params):
    img_float = img_as_float(img); det_hessian = hessian_matrix_det(img_float, sigma=params['hessian_sigma'])
    det_hessian = np.abs(det_hessian)
    return (det_hessian / (np.max(det_hessian) + 1e-9) * 255).astype(np.uint8)
filter_implementations = {
    'clahe': apply_clahe, 'blur': apply_bilateral_blur, 'threshold': apply_adaptive_threshold,
    'tophat': apply_tophat, 'blackhat': apply_blackhat, 'dog': apply_dog, 'hessian': apply_hessian
}
def run_pipeline(bgr_img, order, params, use_msrcr=False):
    processed_img = apply_msrcr(bgr_img, params) if use_msrcr else bgr_img.copy()
    is_gray = len(processed_img.shape) == 2
    for filter_name in order:
        if not is_gray and filter_name in filter_implementations:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY); is_gray = True
        filter_func = filter_implementations.get(filter_name)
        if filter_func: processed_img = filter_func(processed_img, params)
    return processed_img
def find_centers_from_image(processed_img, params):
    if len(processed_img.shape) == 3: binary_img, _ = cv2.threshold(cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif np.max(processed_img) > 1 and len(processed_img.shape) == 2: _, binary_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: binary_img = processed_img
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_contours = len(contours); centers = []
    max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
    if max_area > 0:
        for c in contours:
            area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
            if peri == 0 or area < params['areaRatio'] * max_area: continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < params['circularity_thresh']: continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if 0.5 < radius < 6: centers.append((x, y))
    stats = {'total_contours': total_contours, 'final_points': len(centers)}
    return centers, stats
def save_checkpoint(filepath, data):
    temp_filepath = str(filepath) + ".tmp"
    with open(temp_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, default=lambda o: o.tolist() if isinstance(o, (np.ndarray, tuple)) else str(o))
    os.replace(temp_filepath, filepath)
def load_checkpoint(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f"✅ 체크포인트 파일 '{filepath}'을(를) 발견했습니다. 작업을 이어서 시작합니다.")
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return None

# -------------------------------------------------------------------
# Part 2: 최종 결과 분석 및 시각화 함수
# -------------------------------------------------------------------

def analyze_and_save_pipeline_steps(original_img, result_info, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(result_info, f, indent=4, default=lambda o: o.tolist() if isinstance(o, (np.ndarray, tuple)) else str(o))
    order, params, use_msrcr = result_info['filter_order'], result_info['params'], result_info.get('use_msrcr', False)
    current_img = original_img.copy()
    cv2.imwrite(str(save_dir / "step0_original.png"), current_img)
    step_counter = 1
    if use_msrcr:
        current_img = apply_msrcr(current_img, params); cv2.imwrite(str(save_dir / f"step{step_counter}_msrcr.png"), current_img); step_counter += 1
    is_gray = len(current_img.shape) == 2
    for filter_name in order:
        if not is_gray and filter_name in filter_implementations:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY); is_gray = True
        filter_func = filter_implementations.get(filter_name)
        if filter_func:
            current_img = filter_func(current_img, params); cv2.imwrite(str(save_dir / f"step{step_counter}_{filter_name}.png"), current_img); step_counter += 1
    return current_img

def save_points_to_csv(points, filepath):
    pd.DataFrame(points, columns=['x', 'y']).to_csv(filepath, index=False, encoding='utf-8-sig')

def draw_detected_points(img, centers, save_path):
    vis = img.copy()
    for pt in centers: cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    cv2.imwrite(str(save_path), vis)

def draw_comparison_image(img, detected_pts, ground_truth_pts, save_path):
    vis = img.copy()
    for pt in ground_truth_pts: cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
    for pt in detected_pts: cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    cv2.imwrite(str(save_path), vis)

def visualize_top_n_summary(df, save_path):
    df_sorted = df.sort_values('Rank'); ranks, x, width = df_sorted['Rank'].astype(str), np.arange(len(df_sorted)), 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, df_sorted['f1'], width, label='F1 Score')
    ax.bar(x, df_sorted['precision'], width, label='Precision')
    ax.bar(x + width, df_sorted['recall'], width, label='Recall')
    ax.set_ylabel('Scores (%)'); ax.set_title(f'Top {len(df)} Combinations Performance Summary')
    ax.set_xticks(x); ax.set_xticklabels([f"Rank {r}" for r in ranks]); ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7); fig.tight_layout()
    plt.savefig(save_path); plt.close(); print(f"✅ 종합 성능 그래프 저장 완료: {save_path}")

def run_post_analysis(original_img, ground_truth_pts, top_results, results_dir):
    print("\n--- 🚀 최종 결과 상세 분석 시작 ---")
    if not top_results: print("분석할 결과가 없습니다."); return
    df_data = []
    for i, res in enumerate(top_results):
        row = {'Rank': i + 1, **res}; row.update(res['params']); del row['params']
        row['filter_order'] = ' > '.join(res['filter_order'])
        df_data.append(row)
    df = pd.DataFrame(df_data)
    df.to_csv(results_dir / f'top_{TOP_N}_summary.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 상위 {TOP_N}개 결과 요약 CSV 저장 완료.")
    for i, result_info in enumerate(top_results):
        rank = i + 1; save_dir = results_dir / f"top{TOP_N}_rank{rank}"; save_dir.mkdir(parents=True, exist_ok=True)
        final_processed_img = analyze_and_save_pipeline_steps(original_img, result_info, save_dir)
        detected_centers, _ = find_centers_from_image(final_processed_img, result_info['params'])
        if detected_centers:
            save_points_to_csv(detected_centers, save_dir / "detected_coordinates.csv")
            draw_detected_points(original_img, detected_centers, save_dir / "visualization_detected.png")
            draw_comparison_image(original_img, detected_centers, ground_truth_pts, save_dir / "visualization_comparison.png")
    visualize_top_n_summary(df, results_dir / f'top_{TOP_N}_summary_chart.png')

def generate_markdown_report(all_results_df, top_n_results, execution_time_secs, results_dir):
    print(f"\n--- ✍️ 분석 리포트 생성 시작 ---")
    report_path = results_dir / "analysis_report.md"
    mins, secs = divmod(execution_time_secs, 60); hours, mins = divmod(mins, 60)
    report = f"# {results_dir.name} 분석 리포트\n\n"
    report += f"**생성일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**해당 실험 소요 시간:** {int(hours)}시간 {int(mins)}분 {int(secs)}초\n"
    report += f"**테스트한 총 조합 수:** {len(all_results_df)}\n\n"
    if not top_n_results: report += "결과가 없습니다."; return
    best_result = top_n_results[0]
    report += f"## 🏆 실행 요약\n\n- **최고 F1 점수:** **{best_result['f1']:.4f}**\n- **최고 성능 조합:** [top{TOP_N}_rank1](./top{TOP_N}_rank1) 폴더 참조\n\n"
    report += f"## 📊 상위 {len(top_n_results)}개 결과 요약\n\n"
    df_top = pd.DataFrame(top_n_results); params_df = pd.json_normalize(df_top['params']); df_top.drop(columns=['params'], inplace=True)
    df_top_display = pd.concat([df_top.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
    display_cols = ['f1', 'precision', 'recall', 'exec_time', 'total_contours', 'final_points', 'filter_order'] + list(params_df.columns)
    report += df_top_display[display_cols].to_markdown(index=False, floatfmt=".4f") + "\n\n"
    report += f"## 🔬 파라미터 영향도 분석\n\n"
    influential_params = ['clahe_clip', 'sigmaColor', 'blockSize', 'cVal', 'morphK']
    for param in influential_params:
        if param in all_results_df.columns:
            report += f"### '`{param}`' 값에 따른 평균 F1 점수\n\n"
            influence_df = all_results_df.groupby(param)['f1'].mean().reset_index().sort_values(by='f1', ascending=False)
            report += influence_df.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    report += f"### '`filter_order`'에 따른 평균 F1 점수\n\n"
    all_results_df['filter_order_str'] = all_results_df['filter_order'].astype(str)
    order_influence_df = all_results_df.groupby('filter_order_str')['f1'].mean().reset_index().sort_values(by='f1', ascending=False)
    report += order_influence_df.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    report += f"## 🔗 파라미터 간 상관관계 히트맵\n\n"
    plt.figure(figsize=(12, 10))
    corr_cols = ['f1', 'exec_time', 'total_contours', 'final_points'] + influential_params
    corr_cols_exist = [col for col in corr_cols if col in all_results_df.columns]
    correlation_matrix = all_results_df[corr_cols_exist].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f"); plt.title("Correlation Heatmap")
    heatmap_path = results_dir / "parameter_correlation_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches='tight'); plt.close()
    report += f"![Correlation Heatmap](./{heatmap_path.name})\n\n"
    report += f"## 📈 튜닝 진행에 따른 정확도 향상 기록\n\n"
    plt.figure(figsize=(12, 7))
    all_results_df['cumulative_max_f1'] = all_results_df['f1'].cummax()
    plt.plot(all_results_df.index, all_results_df['cumulative_max_f1']); plt.xlabel("Combination Index"); plt.ylabel("Best F1 Score So Far"); plt.title("F1 Score Improvement During Tuning")
    plt.grid(True); trend_path = results_dir / "accuracy_trend.png"; plt.savefig(trend_path); plt.close()
    report += f"![Accuracy Trend](./{trend_path.name})\n\n"
    with open(report_path, 'w', encoding='utf-8') as f: f.write(report)
    print(f"✅ 분석 리포트 저장 완료: {report_path}")

# -------------------------------------------------------------------
# Part 3: 실행 엔진 (Orchestrator)
# -------------------------------------------------------------------

def run_single_experiment(experiment_config):
    """하나의 실험 설정에 따라 전체 탐색 및 분석을 수행하는 함수"""
    # --- 1. 실험 설정 및 경로 정의 ---
    run_name = experiment_config['experiment_name']
    use_msrcr = experiment_config['use_msrcr']
    enhancement_filter = experiment_config['enhancement_filter']
    base_pipeline = experiment_config['base_pipeline']

    try: PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError: PROJECT_ROOT = Path('.').resolve()
    
    RESULTS_DIR = PROJECT_ROOT / "results" / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = RESULTS_DIR / "checkpoint.json"
    temp_log_dir = RESULTS_DIR / "temp_logs"
    temp_log_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 체크포인트 로드 ---
    checkpoint = load_checkpoint(checkpoint_path)
    top_n_results, resume_from = (checkpoint.get('top_n_results', []), checkpoint.get('last_completed_count', 0)) if checkpoint else ([], 0)

    # --- 3. 데이터 로드 및 파라미터 조합 생성 ---
    IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "dot.png"
    GT_PATH = PROJECT_ROOT / "data" / "raw" / "ground_truth_001.csv"
    img = cv2.imread(str(IMAGE_PATH)); 
    if img is None: print(f"[{run_name}] ERROR: 이미지 로드 실패"); return
    try: ground_truth = [(float(x), float(y)) for x, y in [line.strip().split(',') for line in open(GT_PATH)]]
    except FileNotFoundError: print(f"[{run_name}] ERROR: GT 파일 로드 실패"); return

    params_grid = {
        'msrcr_sigmas': [[15, 80, 250]], 'clahe_clip': [1.0, 2.0, 3.0], 'sigmaColor': [25, 50, 75],
        'sigmaSpace': [25, 50, 75], 'blockSize': [9, 11, 13, 15], 'cVal': [2, 5, 10], 'morphK': [3, 5, 7],
        'dog_sigma1': [1.0], 'dog_sigma2': [1.6], 'hessian_sigma': [1.0, 2.0],
        'areaRatio': [0.005, 0.01, 0.02], 'circularity_thresh': [0.4, 0.5, 0.6]
    }
    
    final_pipeline_order = [enhancement_filter if step == 'ENHANCER_SLOT' else step for step in base_pipeline if step != 'ENHANCER_SLOT' or enhancement_filter]
    filter_orders_to_test = [tuple(final_pipeline_order)]

    param_keys = list(params_grid.keys())
    param_combinations = list(itertools.product(*params_grid.values()))
    all_tasks_iterator = itertools.product([use_msrcr], filter_orders_to_test, param_combinations)
    total_combinations = len(param_combinations)
    
    print(f"총 {total_combinations}개의 조합을 테스트합니다."); print(f"{resume_from}번째 작업부터 재개합니다." if resume_from > 0 else "")
    
    if resume_from >= total_combinations:
        print("이미 모든 작업이 완료되었습니다.")
        # 작업이 이미 완료되었더라도, 결과 분석 및 리포트 생성은 수행하도록 함
    else:
        # --- 4. 메인 루프 실행 ---
        log_buffer, chunk_count = [], len(list(temp_log_dir.glob('*.parquet')))
        for i, task in enumerate(all_tasks_iterator):
            if i < resume_from: continue
            current_count = i + 1; combination_start_time = time.time()
            use_msrcr_flag, order, param_values = task
            params = dict(zip(param_keys, param_values))
            
            processed_img = run_pipeline(img, order, params, use_msrcr_flag)
            centers, stats = find_centers_from_image(processed_img, params)
            precision, recall, f1, matched = compute_accuracy(centers, ground_truth)
            combination_exec_time = time.time() - combination_start_time
            
            current_result = {'f1': f1, 'precision': precision, 'recall': recall, 'exec_time': combination_exec_time, **stats, 'use_msrcr': use_msrcr_flag, 'filter_order': order, 'params': params}
            log_buffer.append(current_result)
            
            if len(log_buffer) >= CHUNK_SIZE:
                df_to_write = pd.DataFrame([{**res, **res['params'], 'filter_order': ' > '.join(res['filter_order'])} for res in log_buffer]).drop(columns='params')
                df_to_write.to_parquet(temp_log_dir / f"temp_log_chunk_{chunk_count}.parquet")
                print(f"📦 Chunk {chunk_count} ({len(log_buffer)}개)를 임시 저장했습니다.")
                log_buffer, chunk_count = [], chunk_count + 1

            top_n_results.append(current_result); top_n_results.sort(key=lambda x: x.get('f1', 0), reverse=True)
            # ⭐⭐⭐ 여기를 수정합니다 ⭐⭐⭐
            top_n_results = top_n_results[:TOP_N] # 소문자 top_n -> 대문자 TOP_N

            if current_count % CHECKPOINT_INTERVAL == 0 or (top_n_results and current_result == top_n_results[0] and f1 > 0):
                print(f"🔄 [{current_count}/{total_combinations}] F1={f1:.2f}. 체크포인트를 저장합니다...")
                save_checkpoint(checkpoint_path, {'top_n_results': top_n_results, 'last_completed_count': current_count})

        if log_buffer:
            df_to_write = pd.DataFrame([{**res, **res['params'], 'filter_order': ' > '.join(res['filter_order'])} for res in log_buffer]).drop(columns='params')
            df_to_write.to_parquet(temp_log_dir / f"temp_log_chunk_{chunk_count}.parquet")
            print(f"📦 마지막 Chunk {chunk_count} ({len(log_buffer)}개)를 임시 저장했습니다.")

    # --- 5. 최종 로그 병합 및 분석 실행 ---
    print("\n--- 🔀 임시 로그 파일들을 최종 Parquet 파일로 병합합니다 ---")
    all_log_files = sorted(temp_log_dir.glob('*.parquet'))
    final_log_path = RESULTS_DIR / "full_results_log.parquet"
    if all_log_files:
        try:
            schema = pq.read_schema(all_log_files[0])
            with pq.ParquetWriter(final_log_path, schema=schema) as writer:
                for file_path in all_log_files:
                    writer.write_table(pq.read_table(file_path, schema=schema))
            print(f"✅ 최종 로그 파일 저장 완료: {final_log_path}")
        except Exception as e: print(f"🔥 로그 파일 병합 중 오류 발생: {e}")
    
    run_post_analysis(img, ground_truth, top_n_results, RESULTS_DIR)
    
    try:
        if final_log_path.exists():
            all_results_df = pd.read_parquet(final_log_path)
            if not all_results_df.empty:
                all_results_df['filter_order'] = all_results_df['filter_order'].apply(lambda x: tuple(x.split(' > ')))
                # 이 부분은 main()에서 관리하므로 run_single_experiment에서는 시간을 인자로 받지 않습니다.
                # generate_markdown_report(all_results_df, top_n_results, execution_time_secs, RESULTS_DIR)
    except FileNotFoundError:
        print("⚠️ 최종 로그 파일이 없어 리포트를 생성할 수 없습니다.")


def main():
    """전체 실험 계획을 순차적으로 실행하는 오케스트레이터(Orchestrator)"""
    print("="*50); print("         전체 자동 튜닝 및 분석을 시작합니다"); print("="*50)
    print(f"총 {len(EXPERIMENTAL_PLAN)}개의 실험이 아래 순서로 진행될 예정입니다.\n")
    for i, config in enumerate(EXPERIMENTAL_PLAN): print(f"  {i+1}. {config['experiment_name']}")
    print("\n스크립트를 중단했다가 다시 실행하면, 마지막 지점부터 자동으로 이어합니다.\n")

    overall_start_time = time.time()
    
    for i, config in enumerate(EXPERIMENTAL_PLAN):
        print("\n" + "="*50); print(f"  [{i+1}/{len(EXPERIMENTAL_PLAN)}] 번째 실험 시작: {config['experiment_name']}"); print("="*50)
        experiment_start_time = time.time()
        run_single_experiment(config)
        experiment_end_time = time.time()
        experiment_duration = experiment_end_time - experiment_start_time
        print(f"해당 실험 소요 시간: {experiment_duration:.2f}초")

        # --- 해당 실험에 대한 리포트 생성 ---
        try:
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
        except NameError:
            PROJECT_ROOT = Path('.').resolve()
        RESULTS_DIR = PROJECT_ROOT / "results" / config['experiment_name']
        final_log_path = RESULTS_DIR / "full_results_log.parquet"
        checkpoint_path = RESULTS_DIR / "checkpoint.json"
        
        if final_log_path.exists():
            all_results_df = pd.read_parquet(final_log_path)
            checkpoint = load_checkpoint(checkpoint_path)
            top_n_results = checkpoint.get('top_n_results', []) if checkpoint else []
            if not all_results_df.empty:
                all_results_df['filter_order'] = all_results_df['filter_order'].apply(lambda x: tuple(x.split(' > ')))
                generate_markdown_report(all_results_df, top_n_results, experiment_duration, RESULTS_DIR)
    
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print("\n\n" + "="*50); print("🎉 모든 실험이 성공적으로 완료되었습니다!"); print(f"모든 실험의 총 소요 시간: {total_duration:.2f}초"); print("="*50)

if __name__ == '__main__':
    main()