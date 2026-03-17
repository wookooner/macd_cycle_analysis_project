"""
사이클 시각화 모듈 (Cycle Visualizer)
=====================================
모든 분석 결과를 시각화하는 통합 모듈.
분석 함수가 반환한 구조화된 딕셔너리를 받아 차트를 생성.

핵심 원칙:
- 시각화 함수는 데이터를 직접 로딩하지 않음 (분석 결과를 전달받음)
- 다크 테마 + 전문적 스타일
- 한글/영문 자동 대응
- save_path 지정 시 파일 저장, 미지정 시 화면 표시

사용법:
    from feature_analysis.viz.cycle_plots import CyclePlots
    
    plots = CyclePlots(save_dir="feature_analysis/output")
    
    # 피처 프로파일 시각화
    plots.plot_feature_profile(profile_result)
    
    # 조건부 확률 테이블 시각화
    plots.plot_conditional_table(single_condition_result)
    
    # 전이 확률 행렬 시각화
    plots.plot_transition_matrix(transition_result)
"""

import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class CyclePlots:
    """사이클 분석 시각화 클래스."""
    
    # 색상 팔레트
    COLORS = {
        'up': '#00E676',
        'down': '#FF5252',
        'UP_PROFIT': '#00C853',
        'UP_LOSS': '#69F0AE',
        'DOWN_PROFIT': '#FF6D00',
        'DOWN_LOSS': '#FF1744',
        'neutral': '#9E9E9E',
        'accent': '#00B0FF',
        'gold': '#FFD700',
        'bg': '#1A1A1A',
        'panel': '#2D2D2D',
        'grid': '#404040',
        'text': '#FFFFFF',
        'positive': '#00E676',
        'negative': '#FF5252',
        'bar_colors': ['#00B0FF', '#FF6D00', '#00E676', '#FF5252', 
                       '#9C27B0', '#FFD700', '#00BCD4', '#FF9800']
    }
    
    CATEGORY_LABELS = {
        'UP_PROFIT': 'UP+이익', 'UP_LOSS': 'UP+손실',
        'DOWN_PROFIT': 'DOWN+이익', 'DOWN_LOSS': 'DOWN+손실'
    }
    
    def __init__(self, save_dir: Optional[str] = None, dpi: int = 120):
        self.save_dir = Path(save_dir) if save_dir else None
        self.dpi = dpi
        self._setup_style()
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_style(self):
        """전문적 다크 테마 설정."""
        plt.style.use('dark_background')
        params = {
            'figure.facecolor': self.COLORS['bg'],
            'axes.facecolor': self.COLORS['panel'],
            'axes.edgecolor': '#606060',
            'grid.color': self.COLORS['grid'],
            'grid.alpha': 0.3,
            'text.color': self.COLORS['text'],
            'axes.labelcolor': self.COLORS['text'],
            'xtick.color': self.COLORS['text'],
            'ytick.color': self.COLORS['text'],
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'axes.unicode_minus': False,
        }
        plt.rcParams.update(params)
        
        # 한글 폰트 시도 → fallback to English
        try:
            import platform
            if platform.system() == 'Windows':
                font_path = 'C:/Windows/Fonts/malgun.ttf'
                if Path(font_path).exists():
                    import matplotlib.font_manager as fm
                    prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = prop.get_name()
                    return
            # Linux에서 Noto Sans CJK 시도
            plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'NanumGothic', 
                                           'DejaVu Sans', 'sans-serif']
        except Exception:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # =========================================================
    # 1. 피처 프로파일 시각화
    # =========================================================
    
    def plot_feature_profile(self, result: Dict, save_name: Optional[str] = None) -> str:
        """
        피처 프로파일 종합 시각화.
        (1) 전체 분포 히스토그램 + KDE
        (2) UP vs DOWN 비교 박스플롯
        (3) 4분류 바이올린 플롯
        (4) 통계 요약 텍스트 패널
        """
        if 'error' in result:
            return f"Error: {result['error']}"
        
        feature = result['feature']
        overall = result.get('overall', {})
        up_down = result.get('up_vs_down', {})
        four_way = result.get('four_way', {})
        meta = result.get('meta', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Feature Profile: {feature}", fontsize=16, fontweight='bold', y=0.98)
        
        # (1) 전체 분포 — 얻을 수 있는 정보: 분포 형태, 이상치
        ax1 = axes[0, 0]
        desc = overall.get('descriptive', {})
        if desc:
            # 데이터 복원: mean ± 3*std 범위의 정규분포 근사치로 시뮬레이션
            # 실제 데이터가 필요하므로 overall에서 가져올 수 없는 경우 스킵
            ax1.text(0.5, 0.5, 
                    f"n = {desc.get('n', '?')}\n"
                    f"Mean = {desc.get('mean', 0):.4f}\n"
                    f"Median = {desc.get('median', 0):.4f}\n"
                    f"Std = {desc.get('std', 0):.4f}\n"
                    f"Skew = {desc.get('skew', 0):.3f}\n"
                    f"Kurtosis = {desc.get('kurtosis', 0):.3f}\n"
                    f"Range: [{desc.get('min', 0):.4f}, {desc.get('max', 0):.4f}]\n"
                    f"IQR: [{desc.get('q25', 0):.4f}, {desc.get('q75', 0):.4f}]",
                    transform=ax1.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))
        ax1.set_title("Distribution Summary", fontweight='bold')
        ax1.axis('off')
        
        # (2) UP vs DOWN 비교 — 효과크기 강조
        ax2 = axes[0, 1]
        if 'descriptive' in up_down and 'error' not in up_down:
            self._plot_group_comparison_bar(ax2, up_down)
        else:
            ax2.text(0.5, 0.5, "UP vs DOWN 비교 불가", 
                    transform=ax2.transAxes, ha='center', fontsize=12)
            ax2.axis('off')
        
        # (3) 4분류 비교
        ax3 = axes[1, 0]
        if 'descriptive' in four_way and 'error' not in four_way:
            self._plot_4way_comparison(ax3, four_way)
        else:
            ax3.text(0.5, 0.5, "4-Way 비교 불가",
                    transform=ax3.transAxes, ha='center', fontsize=12)
            ax3.axis('off')
        
        # (4) 효과크기 + 해석
        ax4 = axes[1, 1]
        self._plot_effect_summary(ax4, up_down, meta)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return self._save_or_show(fig, save_name or f"profile_{feature}")
    
    def _plot_group_comparison_bar(self, ax, up_down: Dict):
        """UP vs DOWN 평균/중앙값 비교 막대 차트."""
        desc = up_down.get('descriptive', {})
        groups = list(desc.keys())
        means = [desc[g].get('mean', 0) for g in groups]
        medians = [desc[g].get('median', 0) for g in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        colors = [self.COLORS.get(g.lower(), self.COLORS['neutral']) for g in groups]
        
        bars1 = ax.bar(x - width/2, means, width, label='Mean', 
                       color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, medians, width, label='Median',
                       color=colors, alpha=0.5, edgecolor='white', linewidth=0.5,
                       hatch='///')
        
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontweight='bold')
        ax.set_title("UP vs DOWN Comparison", fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', fontsize=9, color=self.COLORS['gold'])
    
    def _plot_4way_comparison(self, ax, four_way: Dict):
        """4분류 비교 막대 차트."""
        desc = four_way.get('descriptive', {})
        if not desc:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
            return
        
        groups = list(desc.keys())
        means = [desc[g].get('mean', 0) for g in groups]
        ns = [desc[g].get('n', 0) for g in groups]
        
        colors = [self.COLORS.get(g, self.COLORS['neutral']) for g in groups]
        labels = [self.CATEGORY_LABELS.get(g, g) for g in groups]
        
        bars = ax.bar(range(len(groups)), means, color=colors, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontweight='bold', fontsize=9)
        ax.set_title("4-Category Comparison", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ax.annotate(f'{h:.3f}\n(n={n})', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', fontsize=8, color=self.COLORS['text'])
    
    def _plot_effect_summary(self, ax, up_down: Dict, meta: Optional[Dict]):
        """효과크기 + 해석 텍스트 패널."""
        ax.axis('off')
        
        lines = []
        if meta:
            lines.append(f"Category: {meta.get('category', '?')}")
            lines.append(f"Enabled: {meta.get('enabled', '?')}")
            lines.append(f"Description: {meta.get('description', '')[:60]}")
            lines.append("")
        
        if 'effect_sizes' in up_down:
            es = up_down['effect_sizes']
            lines.append("=== Effect Sizes ===")
            lines.append(f"  {es.get('cliffs_delta', {}).get('description', 'N/A')}")
            lines.append(f"  {es.get('rank_biserial', {}).get('description', 'N/A')}")
            lines.append(f"  {es.get('cohens_d', {}).get('description', 'N/A')}")
            lines.append("")
        
        if 'tests' in up_down:
            tests = up_down['tests']
            lines.append("=== Statistical Tests ===")
            mw = tests.get('mann_whitney', {})
            lines.append(f"  Mann-Whitney U: p={mw.get('p_value', 'N/A'):.4e}"
                        f" {'✓' if mw.get('significant') else '✗'}")
            ks = tests.get('ks', {})
            lines.append(f"  KS Test: p={ks.get('p_value', 'N/A'):.4e}")
        
        if 'interpretation' in up_down:
            lines.append("")
            lines.append(f"→ {up_down['interpretation'][:80]}")
        
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=self.COLORS['panel'], alpha=0.8))
    
    # =========================================================
    # 2. 피처 랭킹 시각화
    # =========================================================
    
    def plot_feature_ranking(self, ranking_df: pd.DataFrame, top_n: int = 20,
                            save_name: Optional[str] = None) -> str:
        """
        효과크기 기준 피처 랭킹 수평 막대 차트.
        
        Args:
            ranking_df: FeatureProfiler.rank_features_by_effect_size()의 결과
        """
        if ranking_df.empty:
            return "Empty ranking"
        
        df = ranking_df.head(top_n).iloc[::-1]  # 하단=높은 순위
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.4)))
        
        colors = [self.COLORS['positive'] if v > 0 else self.COLORS['negative'] 
                 for v in df['cliffs_delta']]
        
        y_pos = range(len(df))
        bars = ax.barh(y_pos, df['cliffs_delta'], color=colors, alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.set_xlabel("Cliff's Delta (Effect Size)", fontweight='bold')
        ax.set_title("Feature Ranking by Effect Size (UP vs DOWN)", 
                     fontsize=14, fontweight='bold')
        ax.axvline(x=0, color=self.COLORS['gold'], linestyle='--', linewidth=1, alpha=0.7)
        
        # 크기 기준선
        for threshold, label in [(0.147, 'small'), (0.33, 'medium'), (0.474, 'large')]:
            ax.axvline(x=threshold, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
            ax.axvline(x=-threshold, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
        
        # p-value 표시
        for i, (_, row) in enumerate(df.iterrows()):
            sig_marker = "★" if row['significant'] else ""
            ax.annotate(f" {row['magnitude']} {sig_marker}", 
                       xy=(row['cliffs_delta'], i),
                       xytext=(5, 0), textcoords="offset points",
                       fontsize=8, va='center', color=self.COLORS['gold'])
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return self._save_or_show(fig, save_name or "feature_ranking")
    
    # =========================================================
    # 3. 조건부 확률 시각화
    # =========================================================
    
    def plot_conditional_table(self, result: Dict, save_name: Optional[str] = None) -> str:
        """
        단일 조건 분석 결과의 구간별 승률/수익률 차트.
        (1) 구간별 승률 막대 + 신뢰구간
        (2) 구간별 평균 수익률
        (3) 구간별 샘플 수
        """
        if 'error' in result:
            return f"Error: {result['error']}"
        
        table = result.get('table')
        if table is None or table.empty:
            return "No table data"
        
        feature = result.get('feature', 'Unknown')
        overall = result.get('overall', {})
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                                gridspec_kw={'height_ratios': [3, 3, 1.5]})
        fig.suptitle(f"Conditional Analysis: {feature}", fontsize=15, fontweight='bold')
        
        x = range(len(table))
        
        # (1) 승률 막대 + 신뢰구간 + 기준선
        ax1 = axes[0]
        bar_colors = [self.COLORS['positive'] if wr > 50 else self.COLORS['negative'] 
                     for wr in table['win_rate']]
        # 신뢰 부족한 구간은 투명하게
        alphas = [0.85 if r else 0.35 for r in table['reliable']]
        
        for i, (wr, col, al, ci_l, ci_u) in enumerate(
            zip(table['win_rate'], bar_colors, alphas, table['ci_lower'], table['ci_upper'])):
            ax1.bar(i, wr, color=col, alpha=al, edgecolor='white', linewidth=0.5)
            # 신뢰구간 에러바
            ax1.errorbar(i, wr, yerr=[[wr - ci_l], [ci_u - wr]], 
                        fmt='none', color=self.COLORS['gold'], capsize=5, linewidth=1.5)
        
        # 전체 승률 기준선
        baseline_wr = overall.get('win_rate', 50)
        ax1.axhline(y=baseline_wr, color=self.COLORS['gold'], linestyle='--', 
                    linewidth=1.5, alpha=0.7, label=f'Overall: {baseline_wr:.1f}%')
        ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(table['range'], rotation=30, ha='right', fontsize=8)
        ax1.set_ylabel("Win Rate (%)", fontweight='bold')
        ax1.set_title("Win Rate by Range (with 95% CI)", fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, row in table.iterrows():
            ax1.annotate(f"{row['win_rate']:.1f}%", 
                        xy=(i, row['win_rate']), xytext=(0, 5),
                        textcoords="offset points", ha='center', fontsize=9,
                        fontweight='bold', color=self.COLORS['text'])
        
        # (2) 평균 수익률
        ax2 = axes[1]
        ret_colors = [self.COLORS['positive'] if r > 0 else self.COLORS['negative'] 
                     for r in table['avg_return']]
        ret_alphas = [0.85 if r else 0.35 for r in table['reliable']]
        
        for i, (ret, col, al) in enumerate(zip(table['avg_return'], ret_colors, ret_alphas)):
            ax2.bar(i, ret, color=col, alpha=al, edgecolor='white', linewidth=0.5)
        
        baseline_ret = overall.get('avg_return', 0)
        ax2.axhline(y=baseline_ret, color=self.COLORS['gold'], linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Overall: {baseline_ret:.4f}')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(table['range'], rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel("Avg Return (%)", fontweight='bold')
        ax2.set_title("Average Return by Range", fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        # (3) 샘플 수
        ax3 = axes[2]
        sample_colors = [self.COLORS['accent'] if r else '#555555' for r in table['reliable']]
        ax3.bar(x, table['n'], color=sample_colors, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax3.axhline(y=30, color=self.COLORS['gold'], linestyle='--', linewidth=1, 
                    alpha=0.7, label='Min samples (30)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(table['range'], rotation=30, ha='right', fontsize=8)
        ax3.set_ylabel("Sample Count", fontweight='bold')
        ax3.set_title("Sample Size per Range", fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_or_show(fig, save_name or f"conditional_{feature}")
    
    # =========================================================
    # 4. 패턴 탐색 결과 시각화
    # =========================================================
    
    def plot_discovered_patterns(self, patterns_df: pd.DataFrame, top_n: int = 15,
                                save_name: Optional[str] = None) -> str:
        """자동 발견된 패턴의 승률/수익률 시각화."""
        if patterns_df.empty:
            return "No patterns found"
        
        df = patterns_df.head(top_n).iloc[::-1]  # 순위 역순
        
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(df) * 0.45)))
        fig.suptitle("Discovered Patterns (Sorted by Win Rate)", 
                    fontsize=14, fontweight='bold')
        
        y_pos = range(len(df))
        
        # (1) 승률
        ax1 = axes[0]
        colors = [self.COLORS['positive'] if wr > 55 else 
                 self.COLORS['negative'] if wr < 45 else self.COLORS['neutral']
                 for wr in df['win_rate']]
        
        bars = ax1.barh(y_pos, df['win_rate'], color=colors, alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        ax1.axvline(x=50, color=self.COLORS['gold'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df['conditions'], fontsize=7)
        ax1.set_xlabel("Win Rate (%)", fontweight='bold')
        ax1.set_title("Win Rate", fontweight='bold')
        
        for i, (_, row) in enumerate(df.iterrows()):
            ax1.annotate(f" {row['win_rate']:.1f}% (n={row['n']})",
                        xy=(row['win_rate'], i), fontsize=8, va='center')
        
        # (2) 평균 수익률
        ax2 = axes[1]
        ret_colors = [self.COLORS['positive'] if r > 0 else self.COLORS['negative']
                     for r in df['avg_return']]
        ax2.barh(y_pos, df['avg_return'], color=ret_colors, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax2.axvline(x=0, color=self.COLORS['gold'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df['conditions'], fontsize=7)
        ax2.set_xlabel("Avg Return (%)", fontweight='bold')
        ax2.set_title("Average Return", fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return self._save_or_show(fig, save_name or "discovered_patterns")
    
    # =========================================================
    # 5. 전이 확률 행렬 시각화
    # =========================================================
    
    def plot_transition_matrix(self, result: Dict, save_name: Optional[str] = None) -> str:
        """전이 확률 행렬 히트맵 + 수익률 히트맵."""
        if 'probability_matrix' not in result:
            return "No transition data"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Cycle Transition Analysis", fontsize=14, fontweight='bold')
        
        # (1) 전이 확률
        self._plot_heatmap(axes[0], result['probability_matrix'], 
                          "Transition Probability (%)", fmt='.1f', cmap='YlOrRd')
        
        # (2) 전이별 평균 수익률
        self._plot_heatmap(axes[1], result['return_matrix'],
                          "Avg Return per Transition (%)", fmt='.4f', cmap='RdYlGn')
        
        # (3) 전이별 샘플 수
        self._plot_heatmap(axes[2], result['count_matrix'],
                          "Sample Count", fmt='.0f', cmap='Blues')
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return self._save_or_show(fig, save_name or "transition_matrix")
    
    def _plot_heatmap(self, ax, matrix: pd.DataFrame, title: str, 
                     fmt: str = '.2f', cmap: str = 'YlOrRd'):
        """히트맵 서브플롯."""
        import matplotlib.colors as mcolors
        
        data = matrix.values.astype(float)
        
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # 값 표시
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = 'white' if val > data.mean() else 'black'
                ax.text(j, i, f"{val:{fmt}}", ha='center', va='center',
                       fontsize=14, fontweight='bold', color=color)
        
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([f"→{c}" for c in matrix.columns], fontweight='bold')
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontweight='bold')
        ax.set_xlabel("Next Cycle", fontweight='bold')
        ax.set_ylabel("Current Cycle", fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # =========================================================
    # 6. 연속 패턴(Streak) 시각화
    # =========================================================
    
    def plot_streak_analysis(self, result: Dict, save_name: Optional[str] = None) -> str:
        """연속 패턴 분석 결과 시각화."""
        streaks_df = result.get('streaks', pd.DataFrame())
        if streaks_df.empty:
            return "No streak data"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Streak Pattern Analysis", fontsize=14, fontweight='bold')
        
        for idx, streak_type in enumerate(['UP', 'DOWN']):
            ax = axes[idx]
            data = streaks_df[streaks_df['streak_type'] == streak_type]
            
            if data.empty:
                ax.text(0.5, 0.5, f"No {streak_type} streak data", 
                       transform=ax.transAxes, ha='center')
                continue
            
            x = range(len(data))
            color = self.COLORS['up'] if streak_type == 'UP' else self.COLORS['down']
            
            bars = ax.bar(x, data['next_win_rate'], color=color, alpha=0.8,
                         edgecolor='white', linewidth=0.5)
            
            ax.axhline(y=50, color=self.COLORS['gold'], linestyle='--', linewidth=1, alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(data['streak_length'], fontweight='bold')
            ax.set_xlabel(f"Consecutive {streak_type} Streak Length", fontweight='bold')
            ax.set_ylabel("Next Cycle Win Rate (%)", fontweight='bold')
            ax.set_title(f"After {streak_type} Streak", fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            for i, (_, row) in enumerate(data.iterrows()):
                ax.annotate(f"{row['next_win_rate']:.1f}%\n(n={row['n_occurrences']})",
                           xy=(i, row['next_win_rate']), xytext=(0, 5),
                           textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_or_show(fig, save_name or "streak_analysis")
    
    # =========================================================
    # 7. 임계값 탐색 시각화
    # =========================================================
    
    def plot_threshold_search(self, result: Dict, save_name: Optional[str] = None) -> str:
        """최적 임계값 탐색 결과 시각화."""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        table = result.get('threshold_table', pd.DataFrame())
        if table.empty:
            return "No threshold data"
        
        feature = result.get('feature', 'Unknown')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f"Optimal Threshold Search: {feature}", fontsize=14, fontweight='bold')
        
        # (1) 승률 곡선 (below / above)
        ax1 = axes[0]
        ax1.plot(table['threshold'], table['win_rate_below'], 
                color=self.COLORS['accent'], linewidth=2, marker='o', markersize=4,
                label=f'{feature} < threshold')
        ax1.plot(table['threshold'], table['win_rate_above'],
                color=self.COLORS['gold'], linewidth=2, marker='s', markersize=4,
                label=f'{feature} >= threshold')
        ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # 최적점 표시
        if result.get('optimal_below'):
            opt = result['optimal_below']
            ax1.axvline(x=opt['threshold'], color=self.COLORS['accent'], 
                       linestyle='--', alpha=0.5)
            ax1.annotate(f"Best below: {opt['win_rate_below']:.1f}%",
                        xy=(opt['threshold'], opt['win_rate_below']),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color=self.COLORS['accent'],
                        arrowprops=dict(arrowstyle='->', color=self.COLORS['accent']))
        
        ax1.set_xlabel(f"{feature} Threshold", fontweight='bold')
        ax1.set_ylabel("Win Rate (%)", fontweight='bold')
        ax1.set_title("Win Rate by Threshold", fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # (2) 샘플 수 분포
        ax2 = axes[1]
        ax2.fill_between(table['threshold'], table['n_below'], 
                        color=self.COLORS['accent'], alpha=0.3, label='n (below)')
        ax2.fill_between(table['threshold'], table['n_above'],
                        color=self.COLORS['gold'], alpha=0.3, label='n (above)')
        ax2.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label='Min samples (30)')
        ax2.set_xlabel(f"{feature} Threshold", fontweight='bold')
        ax2.set_ylabel("Sample Count", fontweight='bold')
        ax2.set_title("Sample Size Distribution", fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return self._save_or_show(fig, save_name or f"threshold_{feature}")
    
    # =========================================================
    # 유틸리티
    # =========================================================
    
    def _save_or_show(self, fig, name: str) -> str:
        """파일 저장 또는 표시."""
        if self.save_dir:
            path = self.save_dir / f"{name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            return str(path)
        else:
            plt.show()
            plt.close(fig)
            return "displayed"