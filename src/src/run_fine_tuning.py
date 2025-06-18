# -*- coding: utf-8 -*-

# 1. 필요한 라이브러리들을 모두 임포트합니다.
import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import shutil
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# --- 기본 설정 ---
TOP_N = 5
CHECKPOINT_INTERVAL = 100
CHUNK_SIZE = 5000

# ==================================================================
# ⭐⭐⭐ 정밀 튜닝 실험 계획 제어판 (EXPERIMENTAL_PLAN) ⭐⭐⭐
#
# 이 리스트에 그리드 서치로 테스트할 실험들을 정의합니다.
# ==================================================================
EXPERIMENTAL_PLAN = [
    # ---- 1. 노이즈 필터 비교 그룹 ----
    {'group': 'Noise_Filter', 'experiment_name': 'Grid_NoiseFilter_Bilateral', 'pipeline_order': ['clahe', 'bilateral', 'tophat', 'threshold']},
    {'group': 'Noise_Filter', 'experiment_name': 'Grid_NoiseFilter_Median', 'pipeline_order': ['clahe', 'median_blur', 'tophat', 'threshold']},

    # ---- 2. 특징 강화 필터 비교 그룹 ----
    {'group': 'Enhancer_Filter', 'experiment_name': 'Grid_Enhancer_None', 'pipeline_order': ['clahe', 'bilateral', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Grid_Enhancer_Tophat', 'pipeline_order': ['clahe', 'bilateral', 'tophat', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Grid_Enhancer_DoG', 'pipeline_order': ['clahe', 'bilateral', 'dog', 'threshold']},
]

# --- 파라미터 그리드 정의 ---
PARAMS_GRID = {
    'clahe_clip': [1.0, 2.0, 3.0],
    'sigmaColor': [50, 75, 100],
    'sigmaSpace': [50, 75, 100],
    'median_ksize': [3, 5, 7],
    'blockSize': [9, 11, 13, 15, 17],
    'cVal': [1, 3, 5, 7],
    'morphK': [3, 5, 7],
    'dog_sigma1': [0.5, 1.0, 1.5],
    'dog_sigma2': [1.0, 1.6, 2.5],
    'areaRatio': [0.001, 0.005, 0.01],
    'circularity_thresh': [0.4, 0.5, 0.6],
    'radius_range': [(0.5, 6.0), (0.4, 7.0)]
}


# --- 헬퍼 함수 및 필터 구현부 ---
def compute_accuracy(detected, ground, tolerance=20.0):
    matched = 0;
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

def apply_clahe(img, params): return cv2.createCLAHE(clipLimit=params.get('clahe_clip', 2.0), tileGridSize=(8, 8)).apply(img)
def apply_bilateral(img, params): return cv2.bilateralFilter(img, d=9, sigmaColor=params.get('sigmaColor', 75), sigmaSpace=params.get('sigmaSpace', 75))
def apply_median_blur(img, params): k = params.get('median_ksize', 5); return cv2.medianBlur(img, k)
def apply_adaptive_threshold(img, params): return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params.get('blockSize', 11), params.get('cVal', 2))
def apply_tophat(img, params): k = params.get('morphK', 5); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)); return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
def apply_dog(img, params): s1 = params.get('dog_sigma1', 1.0); s2 = params.get('dog_sigma2', 1.6); blur1 = cv2.GaussianBlur(img, (0, 0), s1); blur2 = cv2.GaussianBlur(img, (0, 0), s2); return cv2.subtract(blur1, blur2)

filter_implementations = {'clahe': apply_clahe, 'bilateral': apply_bilateral, 'median_blur': apply_median_blur, 'threshold': apply_adaptive_threshold, 'tophat': apply_tophat, 'dog': apply_dog}
FILTER_TO_PARAMS = {
    'clahe': ['clahe_clip'], 'bilateral': ['sigmaColor', 'sigmaSpace'], 'median_blur': ['median_ksize'],
    'threshold': ['blockSize', 'cVal'], 'tophat': ['morphK'], 'dog': ['dog_sigma1', 'dog_sigma2'],
}

def run_pipeline(bgr_img, order, params):
    processed_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    for filter_name in order:
        filter_func = filter_implementations.get(filter_name)
        if filter_func: processed_img = filter_func(processed_img, params)
    return processed_img

def find_centers(processed_img, params):
    if np.max(processed_img) > 1 and len(processed_img.shape) == 2: _, binary_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: binary_img = processed_img
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
    if max_area > 0:
        for c in contours:
            area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
            if peri == 0 or area < params.get('areaRatio', 0.01) * max_area: continue
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            if circularity < params.get('circularity_thresh', 0.5): continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            min_rad, max_rad = params.get('radius_range', (0.5, 6.0))
            if min_rad < radius < max_rad: centers.append((x, y))
    stats = {'total_contours': len(contours), 'final_points': len(centers)}
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

# --- 결과 분석 및 리포팅 함수들 ---
def run_post_analysis(original_img, ground_truth, top_f1_results, top_recall_results, results_dir):
    print("\n--- 🚀 최종 결과 상세 분석 시작 ---")
    if not top_f1_results: print("분석할 결과가 없습니다."); return
    
    # F1 Score 챔피언 분석
    print("\n--- 🏆 F1 Score 챔피언 상세 분석 ---")
    f1_summary_data = [{'Rank': i + 1, **res} for i, res in enumerate(top_f1_results)]
    pd.DataFrame(pd.json_normalize(f1_summary_data)).to_csv(results_dir / f"top_{TOP_N}_f1_summary.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 상위 {TOP_N}개 (F1 기준) 결과 요약 CSV 저장 완료.")
    for i, result_info in enumerate(top_f1_results): save_detailed_report(original_img, ground_truth, result_info, results_dir, f"top_f1_rank{i+1}")

    # Recall 챔피언 분석
    print("\n--- 🎯 Recall 챔피언 상세 분석 ---")
    recall_summary_data = [{'Rank': i + 1, **res} for i, res in enumerate(top_recall_results)]
    pd.DataFrame(pd.json_normalize(recall_summary_data)).to_csv(results_dir / f"top_{TOP_N}_recall_summary.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 상위 {TOP_N}개 (Recall 기준) 결과 요약 CSV 저장 완료.")
    for i, result_info in enumerate(top_recall_results): save_detailed_report(original_img, ground_truth, result_info, results_dir, f"top_recall_rank{i+1}")
    
def run_pipeline_for_analysis(img, params, pipeline_order):
    step_images = {'step0_original': img.copy()}; current_img = img.copy()
    processed_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    for i, filter_name in enumerate(pipeline_order):
        filter_func = filter_implementations.get(filter_name)
        if filter_func:
            processed_img = filter_func(processed_img, params)
            step_images[f'step{i+1}_{filter_name}'] = processed_img.copy()
    return processed_img, step_images

def save_detailed_report(original_img, ground_truth, result_info, results_dir, rank_prefix):
    save_dir = results_dir / rank_prefix; save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Processing {rank_prefix} -> saving to {save_dir}/")
    
    with open(save_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(result_info, f, indent=4, default=lambda o: o.tolist() if isinstance(o, (np.ndarray, tuple)) else str(o))
    
    _, step_images = run_pipeline_for_analysis(original_img, result_info['params'], result_info['pipeline_order'])
    for name, step_img in step_images.items():
        cv2.imwrite(str(save_dir / f"{name}.png"), step_img)
    
    centers, _ = find_centers(step_images[list(step_images.keys())[-1]], result_info['params'])

    if centers:
        vis = original_img.copy()
        for pt in centers: cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        for pt in ground_truth: cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 1)
        cv2.imwrite(str(save_dir / "visualization_comparison.png"), vis)
        pd.DataFrame(centers, columns=['x','y']).to_csv(save_dir / "detected_coordinates.csv", index=False)

def generate_final_summary_report(results_root_dir):
    print("\n\n" + "="*60); print("    📊 모든 실험 결과 종합 분석 및 리포팅 시작 📊"); print("="*60)
    all_trials_data, best_results = [], []

    for config in EXPERIMENTAL_PLAN:
        exp_name = config['experiment_name']; log_path = results_root_dir / exp_name / "full_results_log.parquet"
        checkpoint_path = results_root_dir / exp_name / "checkpoint.json"
        if log_path.exists():
            df = pd.read_parquet(log_path); df['experiment'] = exp_name; df['group'] = config.get('group', 'Misc'); all_trials_data.append(df)
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f: checkpoint_data = json.load(f)
                best_trial = checkpoint_data.get('top_f1_results', [{}])[0]
                if best_trial: best_results.append({'experiment': exp_name, 'group': config.get('group', 'Misc'), **best_trial})
    
    if not all_trials_data: print("분석할 데이터가 없습니다."); return
        
    full_df = pd.concat(all_trials_data, ignore_index=True); 
    if not best_results: print("최고 기록이 없어 요약을 생성할 수 없습니다."); return
    best_df = pd.json_normalize(best_results).sort_values('f1', ascending=False)
    
    report_path = results_root_dir / "_Overall_Grid_Analysis_Report.md"
    report = f"# 그리드 서치 종합 분석 리포트\n\n## 🏆 최종 챔피언\n\n최고의 성능을 보인 실험은 **{best_df.iloc[0]['experiment']}** 이며, F1 점수는 **{best_df.iloc[0]['f1']:.2f}%** 입니다.\n\n"
    report += "## 🥇 그룹별 상위 3개 결과 비교\n\n각 실험 그룹 내에서 가장 높은 F1 점수를 기록한 상위 3개 파이프라인입니다.\n\n"
    group_winners_top3 = best_df.groupby('group').head(3)
    report += group_winners_top3[['group', 'experiment', 'f1', 'precision', 'recall']].to_markdown(index=False, floatfmt=".2f") + "\n\n"
    
    report += "## 📈 실험별 성능 분포 비교\n\n각 파이프라인의 전반적인 성능 안정성을 보여줍니다.\n\n"
    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    sns.boxplot(ax=axes[0], data=full_df, x='f1', y='experiment', orient='h'); axes[0].set_title('F1 Score Distribution')
    sns.boxplot(ax=axes[1], data=full_df, x='precision', y='experiment', orient='h'); axes[1].set_title('Precision Distribution')
    sns.boxplot(ax=axes[2], data=full_df, x='recall', y='experiment', orient='h'); axes[2].set_title('Recall Distribution')
    fig.tight_layout()
    dist_path = results_root_dir / "_metrics_distribution.png"; plt.savefig(dist_path, bbox_inches='tight'); plt.close()
    report += f"![Metrics Distribution](./{dist_path.name})\n\n"
    
    with open(report_path, 'w', encoding='utf-8') as f: f.write(report)
    print(f"✅ 종합 분석 리포트 저장 완료: {report_path}")

# --- 메인 실행 엔진 ---
def run_single_grid_experiment(config):
    try: PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError: PROJECT_ROOT = Path('.').resolve()
    
    run_name = config['experiment_name']
    RESULTS_DIR = PROJECT_ROOT / "fine_results" / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = RESULTS_DIR / "checkpoint.json"
    temp_log_dir = RESULTS_DIR / "temp_logs"; temp_log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    top_f1_results = checkpoint.get('top_f1_results', []) if checkpoint else []
    top_recall_results = checkpoint.get('top_recall_results', []) if checkpoint else []
    resume_from = checkpoint.get('last_completed_count', 0) if checkpoint else 0

    IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "dot.png"
    GT_PATH = PROJECT_ROOT / "data" / "raw" / "ground_truth_001.csv"
    img = cv2.imread(str(IMAGE_PATH));
    if img is None: print(f"[{run_name}] ERROR: 이미지 로드 실패"); return
    try: ground_truth = [(float(x), float(y)) for x, y in [line.strip().split(',') for line in open(GT_PATH)]]
    except FileNotFoundError: print(f"[{run_name}] ERROR: GT 파일 로드 실패"); return

    pipeline_order = config['pipeline_order']
    
    active_param_names = set(['areaRatio', 'circularity_thresh', 'radius_range'])
    for filter_name in pipeline_order: active_param_names.update(FILTER_TO_PARAMS.get(filter_name, []))
    params_grid = {key: PARAMS_GRID[key] for key in active_param_names if key in PARAMS_GRID}
    
    param_keys = list(params_grid.keys())
    for key in param_keys:
        if key.endswith('ksize') or key == 'blockSize': params_grid[key] = np.unique([(v//2)*2+1 for v in params_grid[key] if v % 2 != 0]).tolist()

    param_combinations = list(itertools.product(*params_grid.values()))
    total_combinations = len(param_combinations)
    
    print(f"총 {total_combinations}개의 조합을 테스트합니다."); print(f"{resume_from}번째 작업부터 재개합니다." if resume_from > 0 else "")
    
    if resume_from >= total_combinations: print("이미 모든 작업이 완료되었습니다.")
    else:
        log_buffer, chunk_count = [], len(list(temp_log_dir.glob('*.parquet')))
        with tqdm(total=total_combinations, desc=run_name, initial=resume_from) as pbar:
            for i, param_values in enumerate(itertools.islice(param_combinations, resume_from, None)):
                current_count = resume_from + i + 1; pbar.update(1)
                params = dict(zip(param_keys, param_values))
                
                processed = run_pipeline(img, pipeline_order, params)
                centers, stats = find_centers(processed, params)
                precision, recall, f1, matched = compute_accuracy(centers, ground_truth)
                
                current_result = {'f1': f1, 'precision': precision, 'recall': recall, 'matched': matched, **stats, 'params': params, 'pipeline_order': pipeline_order}
                log_buffer.append(current_result)
                
                top_f1_results.append(current_result); top_f1_results.sort(key=lambda x: x['f1'], reverse=True); top_f1_results = top_f1_results[:TOP_N]
                top_recall_results.append(current_result); top_recall_results.sort(key=lambda x: x['recall'], reverse=True); top_recall_results = top_recall_results[:TOP_N]
                pbar.set_postfix(BestF1=f"{top_f1_results[0]['f1']:.2f}", BestRecall=f"{top_recall_results[0]['recall']:.2f}")

                if len(log_buffer) >= CHUNK_SIZE:
                    pd.DataFrame([{**res, **res['params']} for res in log_buffer]).drop(columns='params').to_parquet(temp_log_dir / f"temp_log_chunk_{chunk_count}.parquet")
                    log_buffer, chunk_count = [], chunk_count + 1

                if current_count % CHECKPOINT_INTERVAL == 0 or (top_f1_results and current_result == top_f1_results[0] and f1 > 0):
                    save_checkpoint(checkpoint_path, {'top_f1_results': top_f1_results, 'top_recall_results': top_recall_results, 'last_completed_count': current_count})

        if log_buffer: pd.DataFrame([{**res, **res['params']} for res in log_buffer]).drop(columns='params').to_parquet(temp_log_dir / f"temp_log_chunk_{chunk_count}.parquet")

    print("\n--- 🔀 임시 로그 파일들을 최종 Parquet 파일로 병합합니다 ---")
    all_log_files = sorted(temp_log_dir.glob('*.parquet'))
    final_log_path = RESULTS_DIR / "full_results_log.parquet"
    if all_log_files:
        try:
            pd.concat([pd.read_parquet(f) for f in all_log_files], ignore_index=True).to_parquet(final_log_path)
            print(f"✅ 최종 로그 파일 저장 완료: {final_log_path}")
        except Exception as e: print(f"🔥 로그 파일 병합 중 오류 발생: {e}")
    
    run_post_analysis(img, ground_truth, top_f1_results, top_recall_results, RESULTS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Grid Search anaylzer for dot detection.")
    parser.add_argument('--fresh-start', action='store_true', help="대상 실험의 기존 진행 상황을 모두 삭제하고 새로 시작합니다.")
    args = parser.parse_args()
    
    print("="*50); print("         Grid Search 정밀 튜닝을 시작합니다"); print("="*50)
    
    if args.fresh_start:
        try: PROJECT_ROOT = Path(__file__).resolve().parent.parent
        except NameError: PROJECT_ROOT = Path('.').resolve()
        fine_results_dir = PROJECT_ROOT / "fine_results"
        if fine_results_dir.exists():
            print(f"✨ '새로 시작' 모드가 활성화되었습니다. '{fine_results_dir}' 폴더를 초기화합니다.")
            shutil.rmtree(fine_results_dir)
    
    for i, config in enumerate(EXPERIMENTAL_PLAN):
        print("\n" + "="*50); print(f"  [{i+1}/{len(EXPERIMENTAL_PLAN)}] 번째 실험 시작: {config['experiment_name']}"); print("="*50)
        run_single_grid_experiment(config)

    # 모든 실험 완료 후 최종 리포트 생성
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        PROJECT_ROOT = Path('.').resolve()
    generate_final_summary_report(PROJECT_ROOT / "fine_results")
    
    print("\n\n" + "="*50); print("🎉 모든 정밀 튜닝 실험이 성공적으로 완료되었습니다!"); print("="*50)

if __name__ == '__main__':
    main()
