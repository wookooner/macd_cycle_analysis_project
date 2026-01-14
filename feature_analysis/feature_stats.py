import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
# 폰트가 없을 경우 경고가 나타날 수 있습니다.
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

def generate_comprehensive_eda_report():
    """
    모든 cycle_feature에 대한 상세 통계와 분포 시각화를 포함하는
    종합 분석 PDF 리포트를 생성합니다.
    """
    # --- 1. 경로 설정 및 데이터 로드 ---
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "cycle_data" / "structured" / "cycles_1w.parquet"
    output_path = project_root / "feature_analysis"
    output_path.mkdir(exist_ok=True) # 폴더가 없으면 생성
    
    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다.")
        return

    df = pd.read_parquet(data_path)
    features_df = pd.json_normalize(df['cycle_features'])
    
    print("✅ 데이터 로드 및 준비 완료.")
    print(f"🔬 총 {len(features_df.columns)}개의 특징에 대한 분석 리포트를 생성합니다.")

    # --- 2. 전체 특징에 대한 기본 통계 요약 ---
    print("\n" + "="*80)
    print("📊 모든 특징에 대한 기본 통계 요약 (Descriptive Statistics)")
    print("="*80)
    print(features_df.describe().transpose().round(2))
    
    # --- 3. 데이터 정규화 ---
    scaler = StandardScaler()
    features_scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

    # --- 4. PDF 리포트 생성 시작 ---
    report_path = output_path / "종합_특징_분석_리포트.pdf"
    with PdfPages(report_path) as pdf:
        # 각 특징별 상세 분석 및 시각화
        for feature in features_df.columns:
            print(f"   - '{feature}' 특징 분석 중...")
            
            # 4-in-1 시각화 대시보드 생성
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"'{feature}' 특징 종합 분석", fontsize=20, y=0.98)

            # 플롯 1: 원본 데이터 분포
            sns.histplot(features_df[feature], kde=True, ax=axes[0, 0], bins=50)
            axes[0, 0].set_title('원본 데이터 분포 (히스토그램 + KDE)')
            axes[0, 0].axvline(features_df[feature].mean(), color='red', linestyle='--', label=f"평균: {features_df[feature].mean():.2f}")
            axes[0, 0].axvline(features_df[feature].median(), color='green', linestyle='-', label=f"중앙값: {features_df[feature].median():.2f}")
            axes[0, 0].legend()

            # 플롯 2: 원본 데이터 박스 플롯
            sns.boxplot(x=features_df[feature], ax=axes[0, 1])
            axes[0, 1].set_title('원본 데이터 범위 및 이상치 (박스 플롯)')

            # 플롯 3: 정규화된 데이터 분포
            sns.histplot(features_scaled_df[feature], kde=True, ax=axes[1, 0], bins=50, color='orange')
            axes[1, 0].set_title('정규화(Z-score) 데이터 분포')
            
            # 플롯 4: 정규화된 데이터 박스 플롯
            sns.boxplot(x=features_scaled_df[feature], ax=axes[1, 1], color='orange')
            axes[1, 1].set_title('정규화(Z-score) 데이터 범위')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 현재 페이지를 PDF에 저장
            pdf.savefig(fig)
            plt.close(fig) # 메모리 해제를 위해 창 닫기
    
    print("\n" + "="*80)
    print("🎉 분석 완료! PDF 리포트가 아래 경로에 저장되었습니다.")
    print(f"   -> {report_path.resolve()}")
    print("="*80)


if __name__ == '__main__':
    # Seaborn 경고 메시지 무시
    warnings.filterwarnings("ignore", "is_categorical_dtype")
    warnings.filterwarnings("ignore", "use_inf_as_na")
    
    generate_comprehensive_eda_report()