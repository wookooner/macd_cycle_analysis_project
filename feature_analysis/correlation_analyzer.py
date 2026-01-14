import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
import warnings
from datetime import datetime
import sys
import os

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

class CorrelationAnalyzer:
    """특징 간 상관관계 분석기"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.numeric_features = None
        self.load_data()
    
    def load_data(self):
        """데이터 로드 및 전처리"""
        try:
            df = pd.read_parquet(self.file_path)
        except FileNotFoundError:
            print(f"오류: '{self.file_path}' 파일을 찾을 수 없습니다.")
            return None
        
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
        
        self.df = prepared_df
        self.numeric_features = prepared_df.select_dtypes(include=np.number).columns.tolist()
        
        print(f"✅ 데이터 로드 완료: {len(self.df)}개 샘플, {len(self.numeric_features)}개 숫자형 특징")
    
    def display_available_features(self):
        """사용 가능한 특징들을 카테고리별로 표시"""
        if self.numeric_features is None:
            print("데이터가 로드되지 않았습니다.")
            return
        
        # 특징을 카테고리별로 분류
        categories = {}
        for feature in self.numeric_features:
            category = feature.split('_')[0] if '_' in feature else '기본정보'
            if category not in categories:
                categories[category] = []
            categories[category].append(feature)
        
        print("\n" + "="*80)
        print("📊 사용 가능한 사이클 특징 목록")
        print("="*80)
        
        for category, features in categories.items():
            print(f"\n--- {category.upper()} 카테고리 ---")
            for i in range(0, len(features), 4):
                print("  ".join([f"{name:<25}" for name in features[i:i+4]]))
        
        print(f"\n총 {len(self.numeric_features)}개의 숫자형 특징이 있습니다.")
    
    def select_target_feature(self):
        """사용자가 분석할 대상 특징을 선택"""
        while True:
            print("\n" + "="*60)
            print("🎯 분석할 대상 특징 선택")
            print("="*60)
            
            target_feature = input("분석할 특징 이름을 입력하세요 (종료: 'quit'): ").strip()
            
            if target_feature.lower() == 'quit':
                return None
            
            if target_feature in self.numeric_features:
                return target_feature
            else:
                print(f"❌ '{target_feature}'는 유효하지 않은 특징입니다.")
                print("사용 가능한 특징 목록을 보려면 'list'를 입력하세요.")
                
                if target_feature.lower() == 'list':
                    self.display_available_features()
    
    def analyze_correlations(self, target_feature, correlation_threshold=0.5):
        """선택된 특징과 다른 모든 특징들 간의 상관관계 분석"""
        if target_feature not in self.numeric_features:
            print(f"❌ '{target_feature}'는 유효하지 않은 특징입니다.")
            return None
        
        # 대상 특징을 제외한 다른 특징들
        other_features = [f for f in self.numeric_features if f != target_feature]
        
        print("\n" + "="*100)
        print(f"🔗 상관관계 분석: '{target_feature}' vs 다른 모든 특징들")
        print("="*100)
        print(f"분석 대상 특징: {target_feature}")
        print(f"비교 대상 특징: {len(other_features)}개")
        print(f"상관계수 임계값: {correlation_threshold}")
        print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 결측값 제거
        analysis_data = self.df[[target_feature] + other_features].dropna()
        
        if len(analysis_data) < 10:
            print("❌ 분석에 충분한 데이터가 없습니다.")
            return None
        
        print(f"분석 데이터 샘플 수: {len(analysis_data)}개")
        
        # 상관관계 계산
        correlations = []
        
        for feature in other_features:
            # Pearson 상관계수
            pearson_r, pearson_p = stats.pearsonr(analysis_data[target_feature], analysis_data[feature])
            
            # Spearman 상관계수
            spearman_r, spearman_p = stats.spearmanr(analysis_data[target_feature], analysis_data[feature])
            
            correlations.append({
                'feature': feature,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'abs_pearson': abs(pearson_r),
                'abs_spearman': abs(spearman_r)
            })
        
        # 절댓값 기준으로 정렬
        correlations.sort(key=lambda x: x['abs_pearson'], reverse=True)
        
        # 임계값 이상의 상관관계만 필터링
        high_correlations = [corr for corr in correlations if corr['abs_pearson'] >= correlation_threshold]
        
        print(f"\n📊 상관관계 분석 결과 (임계값: {correlation_threshold})")
        print("-" * 100)
        print(f"전체 특징 수: {len(other_features)}개")
        print(f"높은 상관관계 특징 수: {len(high_correlations)}개")
        
        if not high_correlations:
            print(f"❌ 상관계수 {correlation_threshold} 이상인 특징이 없습니다.")
            print("임계값을 낮춰서 다시 시도해보세요.")
            return None
        
        print(f"\n🎯 높은 상관관계 특징들 (상관계수 ≥ {correlation_threshold}):")
        print("-" * 100)
        print(f"{'특징명':<30} {'Pearson':<12} {'Spearman':<12} {'P-value':<12} {'해석':<20}")
        print("-" * 100)
        
        for corr in high_correlations:
            # 상관관계 강도 해석
            abs_r = corr['abs_pearson']
            if abs_r >= 0.8:
                strength = "매우 강함"
            elif abs_r >= 0.6:
                strength = "강함"
            elif abs_r >= 0.4:
                strength = "보통"
            else:
                strength = "약함"
            
            # 방향성
            direction = "양의 상관" if corr['pearson_r'] > 0 else "음의 상관"
            interpretation = f"{direction} ({strength})"
            
            print(f"{corr['feature']:<30} {corr['pearson_r']:>11.4f} {corr['spearman_r']:>11.4f} {corr['pearson_p']:>11.4f} {interpretation:<20}")
        
        # 상관관계 요약 통계
        print(f"\n📈 상관관계 요약 통계:")
        print("-" * 60)
        pearson_values = [corr['pearson_r'] for corr in high_correlations]
        spearman_values = [corr['spearman_r'] for corr in high_correlations]
        
        print(f"Pearson 상관계수:")
        print(f"  평균: {np.mean(pearson_values):.4f}")
        print(f"  최대: {np.max(pearson_values):.4f}")
        print(f"  최소: {np.min(pearson_values):.4f}")
        print(f"  표준편차: {np.std(pearson_values):.4f}")
        
        print(f"Spearman 상관계수:")
        print(f"  평균: {np.mean(spearman_values):.4f}")
        print(f"  최대: {np.max(spearman_values):.4f}")
        print(f"  최소: {np.min(spearman_values):.4f}")
        print(f"  표준편차: {np.std(spearman_values):.4f}")
        
        return high_correlations
    
    def multivariate_analysis(self, target_feature, high_correlation_features):
        """높은 상관관계를 보인 특징들에 대한 다변량 분석"""
        if not high_correlation_features:
            print("❌ 다변량 분석할 특징이 없습니다.")
            return
        
        # 분석할 특징들 (대상 특징 + 높은 상관관계 특징들)
        analysis_features = [target_feature] + [corr['feature'] for corr in high_correlation_features]
        
        print("\n" + "="*100)
        print("🔗 다변량 분석 (Multivariate Analysis)")
        print("="*100)
        print(f"분석 특징 수: {len(analysis_features)}개")
        print(f"분석 특징: {', '.join(analysis_features)}")
        
        # 결측값 제거
        analysis_data = self.df[analysis_features].dropna()
        
        if len(analysis_data) < 10:
            print("❌ 다변량 분석에 충분한 데이터가 없습니다.")
            return
        
        print(f"분석 데이터 샘플 수: {len(analysis_data)}개")
        
        # 1. 상관관계 매트릭스 분석
        print(f"\n📊 1. 상관관계 매트릭스 분석")
        print("-" * 80)
        
        corr_matrix = analysis_data.corr()
        
        # 상관관계 매트릭스에서 가장 높은 상관관계들 찾기
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.5:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print("높은 상관관계 쌍들 (|r| ≥ 0.5):")
        for pair in high_corr_pairs[:10]:  # 상위 10개만 표시
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.4f}")
        
        # 2. 주성분 분석 (PCA)
        print(f"\n🎯 2. 주성분 분석 (Principal Component Analysis)")
        print("-" * 80)
        
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_data)
        
        # PCA 수행
        pca = PCA()
        pca_result = pca.fit(scaled_data)
        
        # 설명된 분산 비율
        explained_variance_ratio = pca_result.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print("주성분별 설명된 분산 비율:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"  PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%) [누적: {cum_var:.4f}]")
        
        # 80% 분산 설명에 필요한 주성분 수
        components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        print(f"\n80% 분산 설명에 필요한 주성분 수: {components_80}")
        
        # 첫 번째 주성분에 대한 특징 기여도
        pc1_loadings = pca_result.components_[0]
        feature_contributions = list(zip(analysis_features, pc1_loadings))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n첫 번째 주성분에 대한 특징 기여도:")
        print(f"{'Feature':<30} {'PC1_Loading':<12}")
        print("-" * 45)
        for feature, loading in feature_contributions:
            print(f"{feature:<30} {loading:>12.6f}")
        
        # 3. 클러스터링 분석
        print(f"\n🎪 3. 클러스터링 분석 (Clustering Analysis)")
        print("-" * 80)
        
        # 최적 클러스터 수 찾기 (2-10)
        silhouette_scores = []
        k_range = range(2, min(11, len(analysis_data)//10 + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        if silhouette_scores:
            best_k = k_range[np.argmax(silhouette_scores)]
            best_score = max(silhouette_scores)
            
            print(f"최적 클러스터 수: {best_k} (실루엣 점수: {best_score:.4f})")
            
            # 최적 클러스터로 클러스터링 수행
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            print(f"\n클러스터별 샘플 수:")
            for i in range(best_k):
                count = sum(1 for label in cluster_labels if label == i)
                percentage = count / len(cluster_labels) * 100
                print(f"  클러스터 {i}: {count}개 ({percentage:.1f}%)")
        
        # 4. 상호정보량 분석
        print(f"\n🔄 4. 상호정보량 분석 (Mutual Information Analysis)")
        print("-" * 80)
        
        # 대상 특징과의 상호정보량 계산
        X = analysis_data.drop(columns=[target_feature])
        y = analysis_data[target_feature]
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_results = list(zip(X.columns, mi_scores))
        mi_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"'{target_feature}'와의 상호정보량:")
        print(f"{'Feature':<30} {'MI_Score':<12}")
        print("-" * 45)
        for feature, score in mi_results:
            print(f"{feature:<30} {score:>12.6f}")
    
    def run_analysis(self):
        """전체 분석 프로세스 실행"""
        if self.df is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return
        
        # 결과 저장을 위한 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'feature_analysis/results'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'correlation_analysis_{timestamp}.txt')
        
        # TeeOutput으로 콘솔과 파일에 동시 출력
        tee = TeeOutput(output_file)
        sys.stdout = tee
        
        try:
            print("="*100)
            print("🔗 사이클 특징 상관관계 분석 시스템")
            print("="*100)
            print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"데이터 파일: {self.file_path}")
            print(f"결과 저장 파일: {output_file}")
            print("="*100)
            
            # 사용 가능한 특징 표시
            self.display_available_features()
            
            # 대상 특징 선택
            target_feature = self.select_target_feature()
            if target_feature is None:
                print("분석을 종료합니다.")
                return
            
            # 상관관계 임계값 설정
            print(f"\n상관관계 임계값을 설정하세요 (기본값: 0.5)")
            threshold_input = input("임계값 입력 (Enter: 기본값 사용): ").strip()
            
            try:
                correlation_threshold = float(threshold_input) if threshold_input else 0.5
            except ValueError:
                correlation_threshold = 0.5
                print("잘못된 입력입니다. 기본값 0.5를 사용합니다.")
            
            # 상관관계 분석
            high_correlations = self.analyze_correlations(target_feature, correlation_threshold)
            
            if high_correlations:
                # 다변량 분석
                self.multivariate_analysis(target_feature, high_correlations)
            
            print("\n" + "="*100)
            print("🎉 상관관계 분석 완료")
            print(f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"결과가 저장된 파일: {output_file}")
            print("="*100)
            
        finally:
            # 출력 스트림 복원
            tee.close()
            print(f"\n✅ 분석 결과가 파일에 저장되었습니다: {output_file}")

def main():
    """메인 실행 함수"""
    # 데이터 파일 경로 설정
    file_path = 'data/cycle_data/structured/cycles_4h.parquet'
    
    # 분석기 초기화 및 실행
    analyzer = CorrelationAnalyzer(file_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
