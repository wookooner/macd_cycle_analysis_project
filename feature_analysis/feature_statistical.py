import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import jarque_bera, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
import warnings
from collections import defaultdict
from datetime import datetime
import sys
import os
from pathlib import Path

# 경고 메시지 무시
warnings.filterwarnings('ignore')

class TeeOutput:
    """콘솔과 파일에 동시에 출력하는 클래스"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

def extract_timeframe_from_path(file_path):
    """파일 경로에서 timeframe 정보를 추출합니다."""
    file_name = Path(file_path).stem
    # cycles_4h.parquet -> 4h 추출
    if 'cycles_' in file_name:
        timeframe = file_name.split('cycles_')[1].split('.')[0]
        return timeframe
    return 'unknown'

def load_and_prepare_data(file_path):
    """Parquet 파일을 로드하고 중첩된 cycle_features를 평탄화합니다."""
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None, None
    
    # timeframe 정보 추출
    timeframe = extract_timeframe_from_path(file_path)
    
    # cycle_features를 평탄화
    features_df = pd.json_normalize(df['cycle_features'])
    features_df.columns = features_df.columns.str.replace('.', '_', regex=False)
    
    # 불필요한 컬럼 제거
    df = df.drop(columns=['cycle_features', 'candle_data'])
    
    # 데이터 결합
    prepared_df = pd.concat([df, features_df], axis=1)
    
    # 컬럼명 정리
    column_renames = {
        'aggregate_volume': 'all_volume', 
        'change_price_pct': 'price_change_pct',
        'change_hist': 'macd_histogram_change', 
        'change_macd': 'macd_change'
    }
    prepared_df = prepared_df.rename(columns=column_renames)
    
    return prepared_df, timeframe

def extended_descriptive_stats(data):
    """확장된 기술통계를 계산합니다."""
    stats_dict = {}
    
    # 기본 통계
    stats_dict['Count'] = len(data)
    stats_dict['Missing'] = data.isna().sum()
    stats_dict['Mean'] = data.mean()
    stats_dict['Std Dev'] = data.std()
    stats_dict['Variance'] = data.var()
    stats_dict['Min'] = data.min()
    stats_dict['Max'] = data.max()
    stats_dict['Range'] = data.max() - data.min()
    stats_dict['Median'] = data.median()
    
    # Mode (최빈값)
    mode_val = data.mode()
    stats_dict['Mode'] = mode_val.iloc[0] if len(mode_val) > 0 else data.iloc[0]
    
    # 분위수
    stats_dict['Q1 (25%)'] = data.quantile(0.25)
    stats_dict['Q3 (75%)'] = data.quantile(0.75)
    stats_dict['IQR'] = stats_dict['Q3 (75%)'] - stats_dict['Q1 (25%)']
    
    # MAD (Median Absolute Deviation)
    stats_dict['MAD'] = np.median(np.abs(data - stats_dict['Median']))
    
    # 변동계수
    stats_dict['CV (계수)'] = stats_dict['Std Dev'] / abs(stats_dict['Mean']) if stats_dict['Mean'] != 0 else np.inf
    
    # 백분위수
    stats_dict['1th Percentile'] = data.quantile(0.01)
    stats_dict['5th Percentile'] = data.quantile(0.05)
    stats_dict['10th Percentile'] = data.quantile(0.10)
    stats_dict['90th Percentile'] = data.quantile(0.90)
    stats_dict['95th Percentile'] = data.quantile(0.95)
    stats_dict['99th Percentile'] = data.quantile(0.99)
    
    return stats_dict

def distribution_analysis(data):
    """분포 특성 분석을 수행합니다."""
    # 왜도와 첨도
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # excess kurtosis
    
    # 분포 유형 추정
    if abs(skewness) > 2:
        skew_interpretation = "매우 비대칭적 (우편향)" if skewness > 0 else "매우 비대칭적 (좌편향)"
    elif abs(skewness) > 1:
        skew_interpretation = "비대칭적 (우편향)" if skewness > 0 else "비대칭적 (좌편향)"
    else:
        skew_interpretation = "대칭적"
    
    if kurtosis > 3:
        kurt_interpretation = "첨예한 분포 (heavy-tailed)"
    elif kurtosis < -1:
        kurt_interpretation = "평평한 분포 (light-tailed)"
    else:
        kurt_interpretation = "정규분포와 유사"
    
    # 분포 유형 추정
    if kurtosis > 3:
        distribution_type = "두꺼운 꼬리 분포 (Fat-tailed)"
    elif abs(skewness) > 2:
        distribution_type = "비대칭 분포"
    else:
        distribution_type = "정규분포와 유사"
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'skew_interpretation': skew_interpretation,
        'kurt_interpretation': kurt_interpretation,
        'distribution_type': distribution_type
    }

def normality_tests(data):
    """다양한 정규성 검정을 수행합니다."""
    tests = {}
    
    # Shapiro-Wilk (샘플 크기가 작을 때)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        tests['Shapiro-Wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'conclusion': '정규분포' if shapiro_p > 0.05 else '비정규분포'
        }
    
    # Anderson-Darling
    ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
    tests['Anderson-Darling'] = {
        'statistic': ad_stat,
        'critical_value': ad_critical[2],  # 5% significance level
        'conclusion': '정규분포' if ad_stat < ad_critical[2] else '비정규분포'
    }
    
    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    tests['Kolmogorov-Smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_p,
        'conclusion': '정규분포' if ks_p > 0.05 else '비정규분포'
    }
    
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(data)
    tests['Jarque-Bera'] = {
        'statistic': jb_stat,
        'p_value': jb_p,
        'conclusion': '정규분포' if jb_p > 0.05 else '비정규분포'
    }
    
    # D'Agostino
    dag_stat, dag_p = normaltest(data)
    tests["D'Agostino"] = {
        'statistic': dag_stat,
        'p_value': dag_p,
        'conclusion': '정규분포' if dag_p > 0.05 else '비정규분포'
    }
    
    # 종합 판단
    normal_count = sum(1 for test in tests.values() if test['conclusion'] == '정규분포')
    total_tests = len(tests)
    
    return tests, normal_count, total_tests

def outlier_analysis(data):
    """다양한 방법으로 이상치를 탐지합니다."""
    outliers = {}
    
    # IQR Method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers['IQR Method'] = len(iqr_outliers)
    
    # Z-Score (>3)
    z_scores = np.abs(stats.zscore(data))
    z_outliers = data[z_scores > 3]
    outliers['Z-Score (>3)'] = len(z_outliers)
    
    # Modified Z-Score
    median = data.median()
    mad = np.median(np.abs(data - median))
    if mad != 0:
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
        outliers['Modified Z-Score'] = len(modified_z_outliers)
    else:
        outliers['Modified Z-Score'] = 0
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_outliers = iso_forest.fit_predict(data.values.reshape(-1, 1))
    iso_outlier_count = sum(1 for x in iso_outliers if x == -1)
    outliers['Isolation Forest'] = iso_outlier_count
    
    # Percentile Method (1%-99%)
    p1 = data.quantile(0.01)
    p99 = data.quantile(0.99)
    percentile_outliers = data[(data < p1) | (data > p99)]
    outliers['Percentile (1%-99%)'] = len(percentile_outliers)
    
    # 극값 분석
    min_val = data.min()
    max_val = data.max()
    min_count = sum(1 for x in data if x == min_val)
    max_count = sum(1 for x in data if x == max_val)
    
    # 가장 많은 이상치를 탐지한 방법
    max_method = max(outliers, key=outliers.get)
    
    return outliers, min_val, max_val, min_count, max_count, max_method

def distribution_fitting(data):
    """다양한 분포에 대한 적합도 검정을 수행합니다."""
    distributions = {}
    
    # 정규분포
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    distributions['정규분포'] = {'ks_stat': ks_stat, 'p_value': ks_p}
    
    # 로그정규분포
    log_data = np.log(data[data > 0])
    if len(log_data) > 0:
        ks_stat, ks_p = stats.kstest(log_data, 'norm', args=(log_data.mean(), log_data.std()))
        distributions['로그정규분포'] = {'ks_stat': ks_stat, 'p_value': ks_p}
    
    # t분포
    try:
        df_param = len(data) - 1
        ks_stat, ks_p = stats.kstest(data, 't', args=(df_param,))
        distributions['t분포'] = {'ks_stat': ks_stat, 'p_value': ks_p}
    except:
        distributions['t분포'] = {'ks_stat': np.inf, 'p_value': 0}
    
    # 지수분포
    try:
        ks_stat, ks_p = stats.kstest(data, 'expon', args=(0, data.mean()))
        distributions['지수분포'] = {'ks_stat': ks_stat, 'p_value': ks_p}
    except:
        distributions['지수분포'] = {'ks_stat': np.inf, 'p_value': 0}
    
    # 감마분포
    try:
        shape, loc, scale = stats.gamma.fit(data)
        ks_stat, ks_p = stats.kstest(data, 'gamma', args=(shape, loc, scale))
        distributions['감마분포'] = {'ks_stat': ks_stat, 'p_value': ks_p}
    except:
        distributions['감마분포'] = {'ks_stat': np.inf, 'p_value': 0}
    
    # 최적 분포 찾기 (p-value 기준)
    best_dist = max(distributions, key=lambda x: distributions[x]['p_value'])
    
    return distributions, best_dist

def robust_statistics(data):
    """로버스트 통계량을 계산합니다."""
    # 일반 통계량
    mean_val = data.mean()
    std_val = data.std()
    
    # 로버스트 통계량
    median_val = data.median()
    mad_val = np.median(np.abs(data - median_val))
    
    # 절단평균
    trimmed_5 = stats.trim_mean(data, 0.05)
    trimmed_10 = stats.trim_mean(data, 0.10)
    
    # 차이 계산
    mean_diff = abs(mean_val - median_val) / abs(mean_val) * 100 if mean_val != 0 else 0
    std_diff = abs(std_val - mad_val) / abs(std_val) * 100 if std_val != 0 else 0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'mad': mad_val,
        'trimmed_5': trimmed_5,
        'trimmed_10': trimmed_10,
        'mean_diff': mean_diff,
        'std_diff': std_diff
    }

def analyze_single_feature(df, feature_name, cycle_type, original_total):
    """단일 특징에 대한 상세한 분석을 수행합니다."""
    if feature_name not in df.columns:
        print(f"\n[오류] '{feature_name}' 특징을 찾을 수 없습니다.")
        return

    data = df[feature_name].dropna()
    
    print("\n" + "="*120)
    print(f"🔬 {cycle_type.upper()} 사이클 단변량 분석: '{feature_name}'")
    print("="*120)

    if len(data) < 4:
        print("데이터가 부족하여 분석을 수행할 수 없습니다.")
        return

    # 1. 확장 기술통계
    print("\n📊 1. 확장 기술통계 (Extended Descriptive Statistics)")
    print("-" * 80)
    stats_dict = extended_descriptive_stats(data)
    
    stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.4e}" if isinstance(x, (int, float)) and abs(x) > 1000 else f"{x:.4f}")
    print(stats_df.to_string(index=False))
    
    # 변동계수 해석
    cv = stats_dict['CV (계수)']
    if cv > 1:
        cv_interpretation = "높음 변동성"
    elif cv > 0.5:
        cv_interpretation = "중간 변동성"
    else:
        cv_interpretation = "낮음 변동성"
    print(f"\n💡 변동계수 해석: {cv_interpretation} ({cv:.4f})")

    # 2. 분포 특성 분석
    print("\n📈 2. 분포 특성 분석 (Distribution Analysis)")
    print("-" * 80)
    dist_analysis = distribution_analysis(data)
    print(f"왜도 (Skewness): {dist_analysis['skewness']:.4f}")
    print(f"   해석: {dist_analysis['skew_interpretation']}")
    print(f"첨도 (Excess Kurtosis): {dist_analysis['kurtosis']:.4f}")
    print(f"   해석: {dist_analysis['kurt_interpretation']}")
    print(f"🎯 분포 유형 추정: {dist_analysis['distribution_type']}")

    # 3. 정규성 검정
    print("\n🧪 3. 정규성 검정 (Normality Tests)")
    print("-" * 80)
    tests, normal_count, total_tests = normality_tests(data)
    for test_name, test_result in tests.items():
        if 'p_value' in test_result:
            print(f"{test_name:<20}: 통계량={test_result['statistic']:.4f}, p-value={test_result['p_value']:.4f}, 결론={test_result['conclusion']}")
        else:
            print(f"{test_name:<20}: 통계량={test_result['statistic']:.4f}, 임계값={test_result['critical_value']:.4f}, 결론={test_result['conclusion']}")
    print(f"\n🎯 종합 판단: {'정규분포' if normal_count > total_tests/2 else '비정규분포'} (정규성 지지: {normal_count}/{total_tests})")

    # 4. 이상치 분석
    print("\n🎯 4. 고급 이상치 분석 (Advanced Outlier Analysis)")
    print("-" * 80)
    outliers, min_val, max_val, min_count, max_count, max_method = outlier_analysis(data)
    print("이상치 탐지 방법별 결과:")
    for method, count in outliers.items():
        percentage = count / len(data) * 100
        print(f"{method:<20}: {count:3d}개 ({percentage:5.2f}%)")
    
    print(f"\n극값 분석:")
    print(f"최솟값: {min_val:.4f} (전체 중 {min_count}개)")
    print(f"최댓값: {max_val:.4f} (전체 중 {max_count}개)")
    print(f"💡 가장 많은 이상치를 탐지한 방법: {max_method} ({outliers[max_method]}개)")

    # 5. 분포 적합도 검정
    print("\n📊 5. 분포 적합도 검정 (Distribution Fitting)")
    print("-" * 80)
    distributions, best_dist = distribution_fitting(data)
    print("분포별 적합도 검정 결과 (p-value 기준):")
    sorted_dists = sorted(distributions.items(), key=lambda x: x[1]['p_value'], reverse=True)
    for i, (dist_name, dist_result) in enumerate(sorted_dists, 1):
        p_val = dist_result['p_value']
        ks_stat = dist_result['ks_stat']
        status = "✓" if p_val > 0.05 else "✗"
        print(f"{i}. {dist_name:<15}: KS={ks_stat:.4f}, p-value={p_val:.4f} {status}")
    print(f"\n🏆 최적 분포: {best_dist} (p-value: {distributions[best_dist]['p_value']:.4f})")

    # 6. 로버스트 통계
    print("\n🛡️ 6. 로버스트 통계 (Robust Statistics)")
    print("-" * 80)
    robust_stats = robust_statistics(data)
    print("로버스트 vs 일반 통계량 비교:")
    print("통계량                            일반         로버스트      차이(%)")
    print("-" * 60)
    print(f"중심위치                 {robust_stats['mean']:>12.4f}    {robust_stats['median']:>12.4f}     {robust_stats['mean_diff']:>6.2f}%")
    print(f"산포                   {robust_stats['std']:>12.4f}    {robust_stats['mad']:>12.4f}     {robust_stats['std_diff']:>6.2f}%")
    
    print(f"\n절단평균 비교:")
    print(f"5% 절단평균:  {robust_stats['trimmed_5']:.4f}")
    print(f"10% 절단평균: {robust_stats['trimmed_10']:.4f}")
    print(f"일반 평균:    {robust_stats['mean']:.4f}")
    
    impact_level = "높음" if robust_stats['mean_diff'] > 50 else "중간" if robust_stats['mean_diff'] > 10 else "낮음"
    print(f"\n💡 이상치 영향도: {impact_level} (중심위치 차이: {robust_stats['mean_diff']:.2f}%)")

def compare_cycle_types(up_df, down_df, feature_name):
    """Up과 Down 사이클 간의 비교 분석을 수행합니다."""
    if feature_name not in up_df.columns or feature_name not in down_df.columns:
        return
    
    up_data = up_df[feature_name].dropna()
    down_data = down_df[feature_name].dropna()
    
    if len(up_data) < 2 or len(down_data) < 2:
        return
    
    print("\n" + "="*120)
    print(f"🔄 UP vs DOWN 사이클 비교 분석: '{feature_name}'")
    print("="*120)
    
    # 기본 통계 비교
    print("\n📊 기본 통계 비교")
    print("-" * 80)
    print(f"{'통계량':<15} {'UP 사이클':<15} {'DOWN 사이클':<15} {'차이':<15}")
    print("-" * 65)
    
    stats_comparison = {
        '평균': (up_data.mean(), down_data.mean()),
        '중앙값': (up_data.median(), down_data.median()),
        '표준편차': (up_data.std(), down_data.std()),
        '최솟값': (up_data.min(), down_data.min()),
        '최댓값': (up_data.max(), down_data.max()),
        '왜도': (stats.skew(up_data), stats.skew(down_data)),
        '첨도': (stats.kurtosis(up_data), stats.kurtosis(down_data))
    }
    
    for stat_name, (up_val, down_val) in stats_comparison.items():
        diff = up_val - down_val
        print(f"{stat_name:<15} {up_val:>15.4f} {down_val:>15.4f} {diff:>15.4f}")
    
    # 통계적 검정
    print("\n🧪 통계적 검정")
    print("-" * 80)
    
    # Mann-Whitney U 검정 (비모수)
    try:
        mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(up_data, down_data, alternative='two-sided')
        print(f"Mann-Whitney U 검정:")
        print(f"  통계량: {mannwhitney_stat:.4f}")
        print(f"  p-value: {mannwhitney_p:.4f}")
        print(f"  결론: {'유의한 차이 없음' if mannwhitney_p > 0.05 else '유의한 차이 있음'} (α=0.05)")
    except:
        print("Mann-Whitney U 검정을 수행할 수 없습니다.")
    
    # Kolmogorov-Smirnov 검정 (분포 차이)
    try:
        ks_stat, ks_p = stats.ks_2samp(up_data, down_data)
        print(f"\nKolmogorov-Smirnov 검정:")
        print(f"  통계량: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.4f}")
        print(f"  결론: {'분포가 동일함' if ks_p > 0.05 else '분포가 다름'} (α=0.05)")
    except:
        print("Kolmogorov-Smirnov 검정을 수행할 수 없습니다.")
    
    # Levene 검정 (분산 동질성)
    try:
        levene_stat, levene_p = stats.levene(up_data, down_data)
        print(f"\nLevene 검정 (분산 동질성):")
        print(f"  통계량: {levene_stat:.4f}")
        print(f"  p-value: {levene_p:.4f}")
        print(f"  결론: {'분산이 동일함' if levene_p > 0.05 else '분산이 다름'} (α=0.05)")
    except:
        print("Levene 검정을 수행할 수 없습니다.")

def multivariate_analysis(df, cycle_type):
    """다변량 분석을 수행합니다."""
    print(f"\n" + "="*120)
    print(f"🔗 {cycle_type.upper()} 사이클 다변량 분석 (Multivariate Analysis)")
    print("="*120)
    
    # 숫자형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        print("다변량 분석을 위한 충분한 수치형 특징이 없습니다.")
        return
    
    # 상관관계 분석
    print("\n📊 1. 상관관계 분석 (Correlation Analysis)")
    print("-" * 80)
    
    # Pearson 상관계수
    pearson_corr = numeric_df.corr()
    # Spearman 상관계수
    spearman_corr = numeric_df.corr(method='spearman')
    
    # 강한 상관관계 찾기 (|r| > 0.7)
    strong_correlations = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            pearson_r = pearson_corr.iloc[i, j]
            spearman_r = spearman_corr.iloc[i, j]
            if abs(pearson_r) > 0.7:
                strong_correlations.append({
                    'pair': f"{pearson_corr.columns[i]} ↔ {pearson_corr.columns[j]}",
                    'pearson': pearson_r,
                    'spearman': spearman_r
                })
    
    # 상관계수 기준으로 정렬
    strong_correlations.sort(key=lambda x: abs(x['pearson']), reverse=True)
    
    print("강한 상관관계 (|r| > 0.7):")
    for corr in strong_correlations:
        print(f"{corr['pair']}: Pearson={corr['pearson']:.3f}, Spearman={corr['spearman']:.3f}")
    
    # 상관계수 요약 통계
    all_correlations = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            all_correlations.append(pearson_corr.iloc[i, j])
    
    if len(all_correlations) > 0:
        print(f"\n상관계수 요약 통계:")
        print(f"평균 상관계수: {np.mean(all_correlations):.4f}")
        print(f"최대 상관계수: {np.max(all_correlations):.4f}")
        print(f"최소 상관계수: {np.min(all_correlations):.4f}")
    
    # 주성분 분석
    if len(numeric_df.columns) >= 2:
        print("\n🎯 2. 주성분 분석 (Principal Component Analysis)")
        print("-" * 80)
        
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df.fillna(0))
        
        # PCA 수행
        pca = PCA()
        pca_result = pca.fit(scaled_data)
        
        # 설명된 분산 비율
        explained_variance_ratio = pca_result.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print("주성분별 설명된 분산 비율:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio[:10], cumulative_variance[:10])):
            print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%) [누적: {cum_var:.4f}]")
        
        # 80% 분산 설명에 필요한 주성분 수
        components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        print(f"\n80% 분산 설명에 필요한 주성분 수: {components_80}")
        
        # 첫 번째 주성분에 대한 특징 기여도
        pc1_loadings = pca_result.components_[0]
        feature_contributions = list(zip(numeric_df.columns, pc1_loadings))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n첫 번째 주성분에 대한 특징 기여도 (상위 10개):")
        print(f"{'Feature':<25} {'PC1_Loading':<12}")
        print("-" * 40)
        for feature, loading in feature_contributions[:10]:
            print(f"{feature:<25} {loading:>12.6f}")

def main():
    """메인 실행 함수"""
    # 분석할 파일 경로 설정
    file_path = 'data/cycle_data/structured/cycles_1h.parquet'
    
    # 데이터 로드
    df, timeframe = load_and_prepare_data(file_path)
    if df is None:
        return

    original_total = len(df)
    
    # cycle_type별로 데이터 분할
    up_cycles = df[df['cycle_type'] == 'up'].copy()
    down_cycles = df[df['cycle_type'] == 'down'].copy()
    
    print(f"총 데이터: {original_total}개")
    print(f"UP 사이클: {len(up_cycles)}개")
    print(f"DOWN 사이클: {len(down_cycles)}개")
    
    if len(up_cycles) == 0 or len(down_cycles) == 0:
        print("UP 또는 DOWN 사이클 데이터가 부족합니다.")
        return
    
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    
    # 결과 저장을 위한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'feature_analysis/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'cycle_comparative_analysis_{timeframe}_{timestamp}.txt')
    
    # TeeOutput으로 콘솔과 파일에 동시 출력
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        print("="*120)
        print("🔬 사이클별 고급 통계 분석 시스템")
        print("="*120)
        print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"데이터 파일: {file_path}")
        print(f"시간대 (Timeframe): {timeframe}")
        print(f"총 데이터 샘플: {original_total}개")
        print(f"UP 사이클: {len(up_cycles)}개 ({len(up_cycles)/original_total*100:.1f}%)")
        print(f"DOWN 사이클: {len(down_cycles)}개 ({len(down_cycles)/original_total*100:.1f}%)")
        print(f"숫자형 특징: {len(numeric_features)}개")
        print(f"특징 목록: {', '.join(numeric_features)}")
        print(f"결과 저장 파일: {output_file}")
        print("="*120)
        
        # UP 사이클 단변량 분석
        print("\n" + "🔵"*50)
        print("UP 사이클 단변량 분석")
        print("🔵"*50)
        
        for i, feature in enumerate(numeric_features, 1):
            print(f"\n[UP {i}/{len(numeric_features)}] {feature} 분석 중...")
            analyze_single_feature(up_cycles, feature, 'up', len(up_cycles))
        
        # DOWN 사이클 단변량 분석
        print("\n" + "🔴"*50)
        print("DOWN 사이클 단변량 분석")
        print("🔴"*50)
        
        for i, feature in enumerate(numeric_features, 1):
            print(f"\n[DOWN {i}/{len(numeric_features)}] {feature} 분석 중...")
            analyze_single_feature(down_cycles, feature, 'down', len(down_cycles))
        
        # UP vs DOWN 비교 분석
        print("\n" + "🔄"*50)
        print("UP vs DOWN 사이클 비교 분석")
        print("🔄"*50)
        
        for i, feature in enumerate(numeric_features, 1):
            print(f"\n[비교 {i}/{len(numeric_features)}] {feature} 비교 분석 중...")
            compare_cycle_types(up_cycles, down_cycles, feature)
        
        # UP 사이클 다변량 분석
        print(f"\n🔵 UP 사이클 다변량 분석 수행 중...")
        multivariate_analysis(up_cycles, 'up')
        
        # DOWN 사이클 다변량 분석
        print(f"\n🔴 DOWN 사이클 다변량 분석 수행 중...")
        multivariate_analysis(down_cycles, 'down')
        
        print("\n" + "="*120)
        print("🎉 사이클별 종합 고급 통계 분석 완료")
        print(f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"시간대: {timeframe}")
        print(f"결과가 저장된 파일: {output_file}")
        print("="*120)
        
    finally:
        # 출력 스트림 복원
        tee.close()
        print(f"\n✅ 분석 결과가 파일에 저장되었습니다: {output_file}")

if __name__ == "__main__":
    main()