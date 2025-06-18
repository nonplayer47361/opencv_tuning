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
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# --- 베이즈 최적화 라이브러리 임포트 ---
from skopt import gp_minimize, load, dump
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations
from skimage.feature import hessian_matrix_det
from skimage.util import img_as_float
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================================================================
# ⭐⭐⭐ 전체 실험 계획 제어판 (EXPERIMENTAL_PLAN) - 최종 확장판 ⭐⭐⭐
# ==================================================================
EXPERIMENTAL_PLAN = [
    # ---- 1. 노이즈 필터 비교 그룹 ----
    {'group': 'Noise_Filter', 'experiment_name': 'NoiseFilter_Bilateral', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Noise_Filter', 'experiment_name': 'NoiseFilter_Gaussian', 'enhancement_filter': 'tophat', 'noise_filter': 'gaussian_blur', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Noise_Filter', 'experiment_name': 'NoiseFilter_Median', 'enhancement_filter': 'tophat', 'noise_filter': 'median_blur', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Noise_Filter', 'experiment_name': 'NoNoiseFilter', 'enhancement_filter': 'tophat', 'noise_filter': None, 'base_pipeline': ['clahe', 'ENHANCER_SLOT', 'threshold']},

    # ---- 2. 특징 강화(Enhancer) 필터 비교 그룹 ----
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_None', 'enhancement_filter': None, 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Tophat', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Blackhat', 'enhancement_filter': 'blackhat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Sobel', 'enhancement_filter': 'sobel', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Scharr', 'enhancement_filter': 'scharr', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Laplacian', 'enhancement_filter': 'laplacian', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_Canny', 'enhancement_filter': 'canny', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT']},
    {'group': 'Enhancer_Filter', 'experiment_name': 'Enhancer_DoG', 'enhancement_filter': 'dog', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold']},


    # ---- 3. 이진화(Threshold) 기법 비교 그룹 ----
    {'group': 'Threshold_Method', 'experiment_name': 'Threshold_Otsu', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold_otsu']},
    {'group': 'Threshold_Method', 'experiment_name': 'Threshold_Mean', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold_mean']},

    # ---- 4. 파이프라인 순서 변경 그룹 ----
    {'group': 'Pipeline_Order', 'experiment_name': 'Order_Blur_First', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['NOISE_SLOT', 'clahe', 'ENHANCER_SLOT', 'threshold']},
    {'group': 'Pipeline_Order', 'experiment_name': 'Order_Enhancer_First', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['ENHANCER_SLOT', 'clahe', 'NOISE_SLOT', 'threshold']},

    # ---- 5. 후처리(Post-processing) 단계 추가 그룹 ----
    {'group': 'Post_Processing', 'experiment_name': 'Post_MorphClose', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold', 'morph_close']},
    {'group': 'Post_Processing', 'experiment_name': 'Post_RemoveSmall', 'enhancement_filter': 'tophat', 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT', 'threshold', 'remove_small']},
    
    # ---- 6. 2단계 특징 강화 그룹 ----
    {'group': 'Multi_Enhancer', 'experiment_name': 'Enhancer_Tophat_DoG', 'enhancement_filter': ['tophat', 'dog'], 'noise_filter': 'bilateral', 'base_pipeline': ['clahe', 'NOISE_SLOT', 'ENHANCER_SLOT1', 'ENHANCER_SLOT2', 'threshold']},
]

# ==================================================================
# ⭐⭐⭐ 파라미터 마스터 정의 ⭐⭐⭐
# ==================================================================
ALL_SKOPT_PARAMS = {
    'clahe_clip': Real(1.0, 5.0, name='clahe_clip'), 'sigmaColor': Integer(20, 150, name='sigmaColor'), 'sigmaSpace': Integer(20, 150, name='sigmaSpace'),
    'median_ksize': Integer(3, 9, name='median_ksize'), 'gaussian_ksize': Integer(3, 9, name='gaussian_ksize'),
    'blockSize': Integer(3, 25, name='blockSize'), 'cVal': Integer(1, 10, name='cVal'),
    'areaRatio': Real(0.001, 0.05, name='areaRatio'), 'circularity_thresh': Real(0.3, 0.8, name='circularity_thresh'),
    'morphK': Integer(3, 9, name='morphK'), 'dog_sigma1': Real(0.5, 2.0, name='dog_sigma1'),
    'dog_sigma2': Real(1.0, 3.0, name='dog_sigma2'), 'hessian_sigma': Real(0.5, 5.0, name='hessian_sigma'),
    'log_ksize': Integer(3, 9, name='log_ksize'), 'sobel_ksize': Integer(3, 5, name='sobel_ksize'),
    'laplacian_ksize': Integer(3, 11, name='laplacian_ksize'),
    'canny_thresh1': Integer(10, 100, name='canny_thresh1'), 'canny_thresh2': Integer(50, 200, name='canny_thresh2'),
    'post_morphK': Integer(3, 9, name='post_morphK'), 'min_area_size': Integer(1, 10, name='min_area_size')
}

FILTER_TO_PARAMS = {
    'clahe': ['clahe_clip'], 'clahe_strong': [], 'clahe_gridlarge': ['clahe_clip'], 
    'bilateral': ['sigmaColor', 'sigmaSpace'], 'median_blur': ['median_ksize'], 'gaussian_blur': ['gaussian_ksize'],
    'threshold': ['blockSize', 'cVal'], 'threshold_otsu': [], 'threshold_mean': ['blockSize', 'cVal'],
    'tophat': ['morphK'], 'blackhat': ['morphK'], 'dog': ['dog_sigma1', 'dog_sigma2'], 'hessian': ['hessian_sigma'], 'log': ['log_ksize'],
    'canny': ['canny_thresh1', 'canny_thresh2'], 'sobel': ['sobel_ksize'], 'scharr': [],
    'laplacian': ['laplacian_ksize'], 'morph_close': ['post_morphK'], 'remove_small': ['min_area_size']
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
def apply_median_blur(img, params): return cv2.medianBlur(img, params.get('median_ksize', 5))
def apply_gaussian_blur(img, params): k = params.get('gaussian_ksize', 5); return cv2.GaussianBlur(img, (k, k), 0)
def apply_adaptive_threshold(img, params): return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params.get('blockSize', 11), params.get('cVal', 2))
def apply_threshold_otsu(img, params): _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); return thresh
def apply_threshold_mean(img, params): return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, params.get('blockSize', 11), params.get('cVal', 2))
def apply_tophat(img, params): k = params.get('morphK', 5); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)); return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
def apply_blackhat(img, params): k = params.get('morphK', 5); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)); return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
def apply_dog(img, params): s1 = params.get('dog_sigma1', 1.0); s2 = params.get('dog_sigma2', 1.6); blur1 = cv2.GaussianBlur(img, (0, 0), s1); blur2 = cv2.GaussianBlur(img, (0, 0), s2); return cv2.subtract(blur1, blur2)
def apply_hessian(img, params): img_float = img_as_float(img); det_hessian = hessian_matrix_det(img_float, sigma=params.get('hessian_sigma', 1.0)); det_hessian = np.abs(det_hessian); return (det_hessian / (np.max(det_hessian) + 1e-9) * 255).astype(np.uint8)
def apply_log(img, params): k = params.get('log_ksize', 5); blur = cv2.GaussianBlur(img, (k, k), 0); laplacian = cv2.Laplacian(blur, cv2.CV_64F); laplacian = np.abs(laplacian); return (laplacian / (np.max(laplacian) + 1e-9) * 255).astype(np.uint8)
def apply_canny(img, params): t1 = params.get('canny_thresh1', 50); t2 = params.get('canny_thresh2', 150); return cv2.Canny(img, t1, t2)
def apply_sobel(img, params): k = params.get('sobel_ksize', 3); sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k); sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k); abs_sobel = np.sqrt(sobelx**2 + sobely**2); return cv2.convertScaleAbs(abs_sobel)
def apply_scharr(img, params): scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0); scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1); abs_scharr = np.sqrt(scharrx**2 + scharry**2); return cv2.convertScaleAbs(abs_scharr)
def apply_laplacian(img, params): k = params.get('laplacian_ksize', 3); lap = cv2.Laplacian(img, cv2.CV_64F, ksize=k); return cv2.convertScaleAbs(lap)
def apply_morph_close(img, params): k = params.get('post_morphK', 3); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)); return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
def apply_remove_small(img, params):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8); output = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= params.get('min_area_size', 3): output[labels == i] = 255
    return output
def apply_clahe_strong(img, params): return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(img)
def apply_clahe_gridlarge(img, params): return cv2.createCLAHE(clipLimit=params.get('clahe_clip', 2.0), tileGridSize=(16, 16)).apply(img)

filter_implementations = {
    'clahe': apply_clahe, 'clahe_strong': apply_clahe_strong, 'clahe_gridlarge': apply_clahe_gridlarge, 'bilateral': apply_bilateral,
    'median_blur': apply_median_blur, 'gaussian_blur': apply_gaussian_blur, 'threshold': apply_adaptive_threshold,
    'threshold_otsu': apply_threshold_otsu, 'threshold_mean': apply_threshold_mean, 'tophat': apply_tophat,
    'blackhat': apply_blackhat, 'dog': apply_dog, 'hessian': apply_hessian, 'log': apply_log,
    'canny': apply_canny, 'sobel': apply_sobel, 'scharr': apply_scharr, 'laplacian': apply_laplacian,
    'morph_close': apply_morph_close, 'remove_small': apply_remove_small
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
            if 0.5 < radius < 6.0: centers.append((x, y))
    return centers

def draw_comparison_image(img, detected_pts, ground_truth_pts, save_path):
    vis = img.copy()
    for pt in ground_truth_pts: cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
    for pt in detected_pts: cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    cv2.imwrite(str(save_path), vis)


# --- 베이즈 최적화 실행 엔진 ---
class BayesianOptimizer:
    def __init__(self, config, n_calls, project_root):
        self.config = config; self.n_calls = n_calls
        self.run_name = config['experiment_name']; self.PROJECT_ROOT = project_root
        self.RESULTS_DIR = self.PROJECT_ROOT / "bayesian_results" / self.run_name
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.RESULTS_DIR / "skopt_checkpoint.gz"
        self.trial_log_path = self.RESULTS_DIR / "trial_log.parquet"
        self.image = cv2.imread(str(self.PROJECT_ROOT / "data" / "raw" / "dot.png"))
        self.ground_truth = pd.read_csv(self.PROJECT_ROOT / "data" / "raw" / "ground_truth_001.csv").to_records(index=False).tolist()
        self.pipeline_order = self._build_pipeline(); self.search_space = self._build_search_space()
        self.trial_results = []

    def _build_pipeline(self):
        pipeline_slots = {'noise_filter': self.config.get('noise_filter'), 'enhancement_filter': self.config.get('enhancement_filter')}
        base_pipeline = self.config['base_pipeline']; pipeline = []
        for step in base_pipeline:
            if step == 'NOISE_SLOT':
                if pipeline_slots['noise_filter']: pipeline.append(pipeline_slots['noise_filter'])
            elif step.startswith('ENHANCER_SLOT'):
                enhancers = pipeline_slots.get('enhancement_filter')
                if enhancers:
                    enhancers = enhancers if isinstance(enhancers, list) else [enhancers]
                    slot_num = int(step.replace('ENHANCER_SLOT', '') or '1')
                    if slot_num <= len(enhancers): pipeline.append(enhancers[slot_num-1])
            else: pipeline.append(step)
        return pipeline

    def _build_search_space(self):
        active_param_names = set(['areaRatio', 'circularity_thresh'])
        for filter_name in self.pipeline_order:
            active_param_names.update(FILTER_TO_PARAMS.get(filter_name, []))
        return [ALL_SKOPT_PARAMS[name] for name in sorted(list(active_param_names))]

    def objective(self, params):
        for key in ['blockSize', 'morphK', 'median_ksize', 'gaussian_ksize', 'sobel_ksize', 'laplacian_ksize', 'post_morphK']:
            if key in params:
                val = int(round(params.get(key, 3)))
                if val % 2 == 0: val += 1
                params[key] = max(3, val)
        processed = run_pipeline(self.image, self.pipeline_order, params)
        centers = find_centers(processed, params)
        precision, recall, f1, _ = compute_accuracy(centers, self.ground_truth)
        self.trial_results.append({'f1': f1, 'precision': precision, 'recall': recall, **params})
        return -f1

    def run(self):
        x0, y0 = (load(self.checkpoint_path).x_iters, load(self.checkpoint_path).func_vals) if self.checkpoint_path.exists() else (None, None)
        if x0: 
            if self.trial_log_path.exists(): self.trial_results = pd.read_parquet(self.trial_log_path).to_dict('records')

        with tqdm(total=self.n_calls, desc=self.run_name, initial=len(x0) if x0 else 0) as pbar:
            @use_named_args(self.search_space)
            def tqdm_objective(**params):
                result = self.objective(params)
                pbar.update(1); pbar.set_postfix(F1=f"{-result:.2f}"); return result
                
            checkpoint_saver = CheckpointSaver(self.checkpoint_path, self.trial_log_path, self)
            result = gp_minimize(func=tqdm_objective, dimensions=self.search_space, n_calls=self.n_calls, n_initial_points=10, x0=x0, y0=y0, random_state=123, callback=checkpoint_saver)

        best_f1 = -result.fun
        best_params = {dim.name: val for dim, val in zip(result.space, result.x)}
        
        for key in ['blockSize', 'morphK', 'median_ksize', 'gaussian_ksize', 'sobel_ksize', 'laplacian_ksize', 'post_morphK']:
            if key in best_params:
                val = int(round(best_params[key]))
                if val % 2 == 0: val += 1
                best_params[key] = max(3, val)
        
        with open(self.RESULTS_DIR / "best_params.json", "w") as f: json.dump(best_params, f, indent=4, default=str)
        final_processed = run_pipeline(self.image, self.pipeline_order, best_params)
        final_centers = find_centers(final_processed, best_params)
        draw_comparison_image(self.image, final_centers, self.ground_truth, self.RESULTS_DIR / "visualization_comparison.png")
        
        _ = plot_objective(result, n_samples=50); plt.suptitle(f"{self.run_name} - Objective Plot"); plt.savefig(self.RESULTS_DIR / "skopt_objective_plot.png"); plt.close()
        _ = plot_evaluations(result, bins=20); plt.suptitle(f"{self.run_name} - Evaluations Plot"); plt.savefig(self.RESULTS_DIR / "skopt_evaluations_plot.png"); plt.close()
        
        return self.run_name, best_f1

class CheckpointSaver:
    def __init__(self, checkpoint_path, trial_log_path, optimizer_instance):
        self.checkpoint_path = checkpoint_path; self.trial_log_path = trial_log_path; self.optimizer_instance = optimizer_instance
    def __call__(self, res):
        dump(res, self.checkpoint_path, store_objective=False)
        pd.DataFrame(self.optimizer_instance.trial_results).to_parquet(self.trial_log_path)

def run_experiment_task(config, n_calls, project_root):
    optimizer = BayesianOptimizer(config, n_calls, project_root)
    return optimizer.run()

def generate_final_summary_report(results_root_dir):
    print("\n\n" + "="*60); print("    📊 모든 실험 결과 종합 분석 및 리포팅 시작 📊"); print("="*60)
    all_trials_data, best_results = [], []
    for config in EXPERIMENTAL_PLAN:
        exp_name = config['experiment_name']; log_path = results_root_dir / exp_name / "trial_log.parquet"; best_param_path = results_root_dir / exp_name / "best_params.json"
        if log_path.exists():
            df = pd.read_parquet(log_path); df['experiment'] = exp_name; df['group'] = config.get('group', 'Misc'); all_trials_data.append(df)
            if best_param_path.exists():
                with open(best_param_path, 'r') as f: best_params_data = json.load(f)
                best_trial = df.loc[df['f1'].idxmax()]; best_trial_dict = best_trial.to_dict()
                best_results.append({'experiment': exp_name, 'group': config.get('group', 'Misc'), **best_trial_dict, 'best_params': best_params_data})
    
    if not all_trials_data: print("분석할 데이터가 없습니다."); return
    full_df = pd.concat(all_trials_data, ignore_index=True); 
    if not best_results: print("최고 기록이 없어 요약을 생성할 수 없습니다."); return
    best_df = pd.json_normalize(best_results).sort_values('f1', ascending=False)
    
    report_path = results_root_dir / "_Overall_Analysis_Report.md"
    report = f"# 베이즈 최적화 종합 분석 리포트\n\n## 🏆 최종 챔피언\n\n"
    if not best_df.empty: report += f"최고의 성능을 보인 실험은 **{best_df.iloc[0]['experiment']}** 이며, F1 점수는 **{best_df.iloc[0]['f1']:.2f}%** 입니다.\n\n"
    
    report += "## 🥇 그룹별 상위 3개 결과 비교\n\n각 실험 그룹 내에서 가장 높은 F1 점수를 기록한 상위 3개 파이프라인입니다.\n\n"
    group_winners_top3 = best_df.groupby('group').head(3)
    report += group_winners_top3[['group', 'experiment', 'f1', 'precision', 'recall']].to_markdown(index=False, floatfmt=".2f") + "\n\n"
    report += "## 📈 실험별 성능 분포 비교\n\n각 파이프라인의 전반적인 성능 안정성을 보여줍니다.\n\n"
    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    sns.boxplot(ax=axes[0], data=full_df, x='f1', y='experiment', orient='h'); axes[0].set_title('F1 Score Distribution')
    sns.boxplot(ax=axes[1], data=full_df, x='precision', y='experiment', orient='h'); axes[1].set_title('Precision Distribution')
    sns.boxplot(ax=axes[2], data=full_df, x='recall', y='experiment', orient='h'); axes[2].set_title('Recall Distribution')
    fig.tight_layout(); dist_path = results_root_dir / "_metrics_distribution.png"; plt.savefig(dist_path, bbox_inches='tight'); plt.close()
    report += f"![Metrics Distribution](./{dist_path.name})\n\n"
    report += "## 🔬 최적 파라미터 트렌드 분석\n\n"
    report += "### 고성능 조합들의 파라미터 상관관계\n\nF1 점수가 높았던 조합들 내에서 파라미터 간의 숨겨진 관계를 보여줍니다.\n\n"
    param_cols = [c for c in best_df.columns if c.startswith('best_params.')]
    if len(param_cols) > 1:
        corr_df = best_df[['f1'] + param_cols].copy(); corr_df.columns = [c.replace('best_params.', '') for c in corr_df.columns]
        plt.figure(figsize=(12, 10)); sns.heatmap(corr_df.corr(), annot=True, cmap='viridis', fmt=".2f"); plt.title("Correlation Heatmap of High-Performing Runs")
        corr_path = results_root_dir / "_best_params_correlation.png"; plt.savefig(corr_path, bbox_inches='tight'); plt.close()
        report += f"![High-Performer Correlation](./{corr_path.name})\n\n"
    if not best_df.empty:
        champion_exp_name = best_df.iloc[0]['experiment']
        champion_df = full_df[full_df['experiment'] == champion_exp_name]
        report += f"### 챔피언 실험('{champion_exp_name}') 파라미터-스코어 분포\n\n최고 성능을 보인 실험에서, 주요 파라미터 값 변화에 따른 F1 점수 분포입니다.\n\n"
        champion_params = [p.name for p in ALL_SKOPT_PARAMS.values() if p.name in champion_df.columns and champion_df[p.name].nunique() > 1][:4]
        if champion_params:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for i, param in enumerate(champion_params):
                ax = axes[i//2, i%2]; sns.scatterplot(ax=ax, data=champion_df, x=param, y='f1', alpha=0.5); ax.set_title(f"F1 Score vs {param}"); ax.grid(True)
            fig.tight_layout(); scatter_path = results_root_dir / "_champion_param_scatter.png"; plt.savefig(scatter_path, bbox_inches='tight'); plt.close()
            report += f"![Champion Parameter-Score Scatter](./{scatter_path.name})\n\n"
    
    with open(report_path, 'w', encoding='utf-8') as f: f.write(report)
    print(f"✅ 종합 분석 리포트 저장 완료: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization anaylzer for dot detection.")
    parser.add_argument('--group', type=str, help="실행할 특정 실험 그룹을 지정합니다.")
    parser.add_argument('--n_calls', type=int, default=100, help="각 실험별 시도 횟수를 지정합니다.")
    parser.add_argument('--workers', type=int, default=4, help="병렬 처리 시 사용할 최대 CPU 코어 수를 지정합니다.")
    args = parser.parse_args()

    plan_to_run = EXPERIMENTAL_PLAN
    if args.group:
        plan_to_run = [c for c in EXPERIMENTAL_PLAN if c.get('group') == args.group]
        if not plan_to_run:
            print(f"'{args.group}' 그룹을 찾을 수 없습니다. 사용 가능한 그룹: {set(c.get('group') for c in EXPERIMENTAL_PLAN)}"); return
            
    print("="*50); print("    🧠 베이즈 최적화 자동 튜닝을 시작합니다 (프로페셔널 버전) 🧠"); print("="*50)
    print(f"총 {len(plan_to_run)}개의 실험이 아래 순서로 진행됩니다. (각 {args.n_calls}회 시도)\n")
    for i, config in enumerate(plan_to_run): print(f"  {i+1}. {config['experiment_name']} (그룹: {config.get('group', 'N/A')})")
    print("\n스크립트를 중단해도 각 실험별로 진행상황이 저장됩니다.\n")

    try: PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError: PROJECT_ROOT = Path('.').resolve()

    num_workers = min(os.cpu_count() or 1, args.workers) 
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_experiment_task, config, args.n_calls, PROJECT_ROOT): config for config in plan_to_run}
        
        for future in tqdm(as_completed(futures), total=len(plan_to_run), desc="전체 실험 진행률"):
            exp_name = futures[future]['experiment_name']
            try:
                run_name, best_f1 = future.result()
                print(f"✅ 실험 '{run_name}' 완료, 최고 F1: {best_f1:.4f}")
            except Exception as e:
                print(f"🔥 실험 '{exp_name}' 실행 중 오류 발생: {e}")

    generate_final_summary_report(PROJECT_ROOT / "bayesian_results")
    
    print("\n\n" + "="*50); print("🎉 모든 실험 및 분석이 성공적으로 완료되었습니다!"); print("="*50)

if __name__ == '__main__':
    main()
