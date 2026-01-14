"""
test1.py 자동 패치 스크립트 (Strength 범위 변환 포함)
======================================================
Strength 특징이 0-100 범위 (예: 63.3%)로 저장된 경우 자동으로 0-1 범위로 변환

사용 방법:
1. 이 스크립트를 test1.py와 같은 디렉토리에 저장
2. python apply_strength_patch.py 실행
3. 백업이 자동으로 생성되고 패치가 적용됨
"""

import re
from pathlib import Path
from datetime import datetime
import shutil

def create_backup(filepath):
    """원본 파일 백업"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
    shutil.copy(filepath, backup_path)
    print(f"✅ 백업 생성: {backup_path}")
    return backup_path


def patch_extract_features_with_strength_conversion(content):
    """extract_all_cycle_features 메서드에 Strength 범위 변환 추가"""
    print("\n🔧 extract_all_cycle_features 메서드 패치 중...")
    
    # Strength 변환 코드를 추가할 위치 찾기
    # "if isinstance(value, (int, float" 부분을 찾아서 strength 변환 로직 추가
    
    # 패턴 1: 중첩된 구조에서의 변환
    pattern1 = r'(                            if isinstance\(value, \(int, float.*?\):\n\s+if not np\.isnan\(value\):\n)'
    
    replacement1 = r'''\1                                    # ✅ 추가: Strength 특징은 0-100 범위일 수 있으므로 0-1로 변환
                                    if category == 'strength':
                                        if value > 1.0:
                                            value = value / 100.0
                                    
'''
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content, count=1)
        print("   ✅ 중첩 구조 Strength 변환 추가")
    else:
        print("   ⚠️ 중첩 구조 패턴을 찾을 수 없습니다")
    
    # 패턴 2: 평면 구조에서의 변환
    pattern2 = r'(                for key, value in cycle_features\.items\(\):\n\s+if isinstance\(value, \(int, float.*?\):\n\s+if not np\.isnan\(value\):\n)'
    
    replacement2 = r'''\1                            # ✅ 추가: Strength 특징 변환
                            if key.startswith('strength_'):
                                if value > 1.0:
                                    value = value / 100.0
                            
'''
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content, count=1)
        print("   ✅ 평면 구조 Strength 변환 추가")
    else:
        print("   ⚠️ 평면 구조 패턴을 찾을 수 없습니다")
    
    # 패턴 3: 결과 검증 로직 추가 (result = pd.DataFrame 이후)
    pattern3 = r'(        result = pd\.DataFrame\(feature_data\)\n)'
    
    replacement3 = r'''\1        
        # ✅ 추가: Strength 특징 최종 검증
        strength_cols = [c for c in result.columns if c.startswith('strength_')]
        if strength_cols:
            print(f"\\n🔍 Strength 특징 추출 결과:")
            for col in strength_cols:
                valid_data = result[col].dropna()
                if len(valid_data) > 0:
                    min_val = valid_data.min()
                    max_val = valid_data.max()
                    mean_val = valid_data.mean()
                    
                    if max_val > 1.0:
                        print(f"   ⚠️ {col}: 여전히 1 초과 (범위: {min_val:.2f}-{max_val:.2f}) - 추가 변환")
                        result[col] = result[col] / 100.0
                    else:
                        print(f"   ✅ {col}: 정상 범위 (범위: {min_val:.4f}-{max_val:.4f}, 평균: {mean_val:.4f})")
        
'''
    
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content, count=1)
        print("   ✅ Strength 검증 로직 추가")
    else:
        print("   ⚠️ 검증 로직 추가 실패")
    
    return content


def patch_plot_ratio_with_auto_conversion(content):
    """plot_ratio_feature 메서드에 자동 범위 감지 및 변환 추가"""
    print("\n🔧 plot_ratio_feature 메서드 패치 중...")
    
    # 기존 메서드 전체를 교체
    pattern = r'    def plot_ratio_feature\(self, ax, data_dict, feature_name, title\):.*?(?=    def plot_continuous_feature|    def _create_category_plot)'
    
    new_method = '''    def plot_ratio_feature(self, ax, data_dict, feature_name, title):
        """Ratio 타입 특징 시각화 (범위 자동 감지 및 변환)"""
        if not data_dict:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
            ax.set_title(title)
            return
        
        # ✅ 추가: 유효한 데이터만 필터링 및 범위 자동 변환
        valid_data_dict = {}
        for label, data in data_dict.items():
            data_series = pd.Series(data)
            clean_data = data_series.dropna()
            
            if len(clean_data) == 0:
                print(f"⚠️ {label}의 {feature_name}: 유효한 데이터 없음 (모두 NaN)")
                continue
            
            # ✅ 추가: 데이터 범위 자동 감지
            min_val = clean_data.min()
            max_val = clean_data.max()
            
            print(f"🔍 {label}의 {feature_name}:")
            print(f"   원본 범위: [{min_val:.4f}, {max_val:.4f}]")
            
            # 0-100 범위로 보이면 0-1로 변환
            if max_val > 1.0:
                print(f"   🔄 0-100 범위로 감지됨 → 0-1로 변환")
                clean_data = clean_data / 100.0
                min_val = clean_data.min()
                max_val = clean_data.max()
                print(f"   변환 후 범위: [{min_val:.4f}, {max_val:.4f}]")
            
            # 범위 검증 (0-1 범위로 변환 후에도 벗어난 경우)
            out_of_range = ((clean_data < 0) | (clean_data > 1)).sum()
            if out_of_range > 0:
                print(f"   ⚠️ {out_of_range}개 값이 여전히 0-1 범위 밖 (클리핑 적용)")
                clean_data = clean_data.clip(0, 1)
            
            valid_data_dict[label] = clean_data.values
            print(f"   ✅ 최종 데이터: {len(clean_data)}개")
        
        if not valid_data_dict:
            ax.text(0.5, 0.5, f'유효한 데이터 없음\\n(NaN 또는 변환 실패)', 
                    ha='center', va='center')
            ax.set_title(title)
            return
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        distribution = {}
        for label, data in valid_data_dict.items():
            data_series = pd.Series(data)
            try:
                counts = pd.cut(data_series, bins=bins, labels=bin_labels, include_lowest=True).value_counts()
                distribution[label] = counts.reindex(bin_labels, fill_value=0)
                
                # 구간별 분포 출력
                print(f"\\n   📊 {label} 구간 분포:")
                for bin_label, count in distribution[label].items():
                    if count > 0:
                        print(f"      {bin_label}: {count}개")
            except Exception as e:
                print(f"   ❌ {label}의 {feature_name} 구간 분류 실패: {e}")
                continue
        
        if not distribution:
            ax.text(0.5, 0.5, '구간 분류 실패', ha='center', va='center')
            ax.set_title(title)
            return
        
        df_dist = pd.DataFrame(distribution)
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
        
        try:
            df_dist.T.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)
        except Exception as e:
            print(f"   ❌ 차트 생성 실패: {e}")
            ax.text(0.5, 0.5, f'차트 생성 실패\\n{str(e)}', ha='center', va='center')
            ax.set_title(title)
            return
        
        ax.set_ylabel('빈도')
        ax.set_xlabel('')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(title='비율 구간', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 통계 정보
        stats_text = []
        for label, data in valid_data_dict.items():
            data_series = pd.Series(data)
            mean_pct = data_series.mean() * 100
            median_pct = data_series.median() * 100
            stats_text.append(f'{label}: μ={mean_pct:.1f}%, M={median_pct:.1f}%')
        
        stats_str = '\\n'.join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
'''
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_method, content, flags=re.DOTALL)
        print("   ✅ 패치 완료")
    else:
        print("   ⚠️ 패턴을 찾을 수 없습니다 - 수동 교체 필요")
    
    return content


def patch_create_category_plot_debug(content):
    """_create_category_plot 메서드에 디버깅 로그 추가"""
    print("\n🔧 _create_category_plot 디버깅 로그 추가 중...")
    
    # for idx, col in enumerate 부분 찾기
    pattern = r'(        for idx, col in enumerate\(features, 1\):\n\s+ax = plt\.subplot.*?\n\s+data_dict = \{\}\n)'
    
    replacement = r'''\1            
            # ✅ 추가: 디버깅 정보
            print(f"\\n🔍 [{category_name}] {col} 시각화 준비:")
            
            if col not in feature_df.columns:
                print(f"   ❌ {col}이 feature_df에 없습니다!")
                ax.text(0.5, 0.5, f'{col}\\n특징 없음', ha='center', va='center')
                ax.set_title(f'{col}', fontsize=10)
                continue
            
'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print("   ✅ 디버깅 로그 추가")
    else:
        print("   ⚠️ 패턴을 찾을 수 없습니다")
    
    # 데이터 추출 부분에 범위 출력 추가
    pattern2 = r'(            if not up_cycles\.empty and col in up_cycles\.columns:\n\s+up_data = up_cycles\[col\]\.dropna\(\)\n\s+if len\(up_data\) > 0:)'
    
    replacement2 = r'''\1
                    print(f"   상승 데이터: {len(up_data)}개")
                    print(f"      범위: [{up_data.min():.4f}, {up_data.max():.4f}]")'''
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        print("   ✅ 상승 데이터 로그 추가")
    
    pattern3 = r'(            if not down_cycles\.empty and col in down_cycles\.columns:\n\s+down_data = down_cycles\[col\]\.dropna\(\)\n\s+if len\(down_data\) > 0:)'
    
    replacement3 = r'''\1
                    print(f"   하락 데이터: {len(down_data)}개")
                    print(f"      범위: [{down_data.min():.4f}, {down_data.max():.4f}]")'''
    
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content)
        print("   ✅ 하락 데이터 로그 추가")
    
    return content


def main():
    """메인 패치 실행 함수"""
    print("="*80)
    print("🔧 test1.py Strength 범위 자동 변환 패치")
    print("="*80)
    
    # test1.py 파일 찾기
    test_file = Path("test1.py")
    
    if not test_file.exists():
        print(f"❌ {test_file}를 찾을 수 없습니다.")
        print(f"   현재 디렉토리: {Path.cwd()}")
        return False
    
    print(f"\n📄 대상 파일: {test_file}")
    
    # 백업 생성
    backup_path = create_backup(test_file)
    
    # 파일 읽기
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 패치 적용
    content = patch_extract_features_with_strength_conversion(content)
    content = patch_plot_ratio_with_auto_conversion(content)
    content = patch_create_category_plot_debug(content)
    
    # 파일 저장
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*80)
    print("✅ Strength 범위 자동 변환 패치 완료!")
    print("="*80)
    print(f"\n📋 주요 변경 사항:")
    print("   1. extract_all_cycle_features:")
    print("      - Strength 특징이 0-100 범위면 자동으로 0-1로 변환")
    print("      - 추출 후 최종 검증 및 재변환")
    print("   2. plot_ratio_feature:")
    print("      - 시각화 시 데이터 범위 자동 감지")
    print("      - 0-100 범위 감지 시 실시간 변환")
    print("      - 구간별 분포 상세 출력")
    print("   3. _create_category_plot:")
    print("      - 각 특징별 데이터 범위 출력")
    print("      - 디버깅 정보 추가")
    
    print(f"\n🔄 원본 복구 방법:")
    print(f"   copy {backup_path} {test_file}")
    
    print(f"\n▶️  다음 단계:")
    print(f"   python test1.py")
    print(f"\n   예상 출력:")
    print(f"   🔍 Strength 특징 추출 결과:")
    print(f"      ✅ strength_direction_pct: 정상 범위 (범위: 0.xxxx-0.xxxx, 평균: 0.xxxx)")
    print(f"      ✅ strength_hist_positive_ratio: 정상 범위 ...")
    print(f"   🔍 [Strength (강도)] strength_direction_pct 시각화 준비:")
    print(f"      상승 데이터: XXX개")
    print(f"      범위: [0.xxxx, 0.xxxx]")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ 패치 실패")
            exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        exit(1)