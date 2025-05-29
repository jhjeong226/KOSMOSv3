# src/visualization/neutron_plots.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from ..core.logger import CRNPLogger, ProcessTimer
from .styles import ModernPlotStyle


class NeutronPlotter:
    """중성자 카운트 및 보정 관련 플롯을 생성하는 클래스"""
    
    def __init__(self, station_id: str, logger: Optional[CRNPLogger] = None):
        self.station_id = station_id
        self.logger = logger or CRNPLogger("NeutronPlotter")
        
        # 스타일 설정
        self.style = ModernPlotStyle()
        
        # 플롯 설정
        self.figure_size = (15, 7)
        self.dpi = 300
        
        # 색상 설정
        self.colors = {
            'raw_neutron': '#2E86AB',
            'corrected_neutron': '#A23B72',
            'fi': '#F18F01',
            'fp': '#C73E1D',
            'fw': '#6C757D',
            'fb': '#28A745'
        }
        
    def plot_neutron_timeseries(self, neutron_data: pd.DataFrame, 
                               output_dir: str = None,
                               show_corrections: bool = True) -> Dict[str, str]:
        """중성자 카운트 시계열 플롯"""
        
        with ProcessTimer(self.logger, "Creating neutron timeseries plots"):
            
            plot_files = {}
            
            # 1. 원시 중성자 카운트 플롯
            if 'total_raw_counts' in neutron_data.columns:
                fig_raw = self._plot_raw_neutron_counts(neutron_data)
                if output_dir:
                    raw_file = Path(output_dir) / f"{self.station_id}_raw_neutron_timeseries.png"
                    fig_raw.savefig(raw_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['raw_neutron'] = str(raw_file)
                    plt.close(fig_raw)
                    
            # 2. 보정된 중성자 카운트 플롯
            if 'total_corrected_neutrons' in neutron_data.columns:
                fig_corrected = self._plot_corrected_neutron_counts(neutron_data)
                if output_dir:
                    corrected_file = Path(output_dir) / f"{self.station_id}_corrected_neutron_timeseries.png"
                    fig_corrected.savefig(corrected_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['corrected_neutron'] = str(corrected_file)
                    plt.close(fig_corrected)
                    
            # 3. 원시 vs 보정 비교 플롯
            if ('total_raw_counts' in neutron_data.columns and 
                'total_corrected_neutrons' in neutron_data.columns):
                fig_comparison = self._plot_neutron_comparison(neutron_data)
                if output_dir:
                    comparison_file = Path(output_dir) / f"{self.station_id}_neutron_comparison.png"
                    fig_comparison.savefig(comparison_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['neutron_comparison'] = str(comparison_file)
                    plt.close(fig_comparison)
                    
            # 4. 보정계수 플롯
            if show_corrections:
                correction_factors = ['fi', 'fp', 'fw', 'fb']
                available_factors = [f for f in correction_factors if f in neutron_data.columns]
                
                if available_factors:
                    fig_corrections = self._plot_correction_factors(neutron_data, available_factors)
                    if output_dir:
                        corrections_file = Path(output_dir) / f"{self.station_id}_correction_factors.png"
                        fig_corrections.savefig(corrections_file, dpi=self.dpi, bbox_inches='tight')
                        plot_files['correction_factors'] = str(corrections_file)
                        plt.close(fig_corrections)
                        
            self.logger.info(f"Generated {len(plot_files)} neutron plots")
            return plot_files
            
    def _plot_raw_neutron_counts(self, data: pd.DataFrame) -> plt.Figure:
        """원시 중성자 카운트 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 데이터 플롯
        ax.plot(data.index, data['total_raw_counts'], 
               color=self.colors['raw_neutron'], linewidth=1.0, alpha=0.8)
        
        # 스타일링
        ax.set_title(f'{self.station_id} - Raw Neutron Counts', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Raw Neutron Counts', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        # 그리드
        ax.grid(True, alpha=0.3)
        
        # 범례에 통계 정보 추가
        mean_val = data['total_raw_counts'].mean()
        std_val = data['total_raw_counts'].std()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_corrected_neutron_counts(self, data: pd.DataFrame) -> plt.Figure:
        """보정된 중성자 카운트 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 데이터 플롯
        ax.plot(data.index, data['total_corrected_neutrons'], 
               color=self.colors['corrected_neutron'], linewidth=1.0, alpha=0.8)
        
        # 스타일링
        ax.set_title(f'{self.station_id} - Corrected Neutron Counts', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Corrected Neutron Counts', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        # 그리드
        ax.grid(True, alpha=0.3)
        
        # 통계 정보
        mean_val = data['total_corrected_neutrons'].mean()
        std_val = data['total_corrected_neutrons'].std()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_neutron_comparison(self, data: pd.DataFrame) -> plt.Figure:
        """원시 vs 보정 중성자 카운트 비교 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 데이터 플롯
        ax.plot(data.index, data['total_raw_counts'], 
               color=self.colors['raw_neutron'], linewidth=1.0, alpha=0.8, 
               label='Raw Counts', linestyle='--')
        ax.plot(data.index, data['total_corrected_neutrons'], 
               color=self.colors['corrected_neutron'], linewidth=1.0, alpha=0.8,
               label='Corrected Counts')
        
        # 스타일링
        ax.set_title(f'{self.station_id} - Raw vs Corrected Neutron Counts', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Neutron Counts', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 보정 효과 계산
        raw_mean = data['total_raw_counts'].mean()
        corrected_mean = data['total_corrected_neutrons'].mean()
        correction_pct = ((corrected_mean - raw_mean) / raw_mean) * 100
        
        # 보정 효과 텍스트
        ax.text(0.02, 0.98, f'Correction Effect: {correction_pct:+.2f}%', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_correction_factors(self, data: pd.DataFrame, factors: List[str]) -> plt.Figure:
        """보정계수 시계열 플롯"""
        
        n_factors = len(factors)
        fig, axes = plt.subplots(n_factors, 1, figsize=(self.figure_size[0], 4*n_factors), sharex=True)
        
        if n_factors == 1:
            axes = [axes]
            
        factor_labels = {
            'fi': 'Incoming Flux Correction (fi)',
            'fp': 'Pressure Correction (fp)', 
            'fw': 'Humidity Correction (fw)',
            'fb': 'Biomass Correction (fb)'
        }
        
        for i, factor in enumerate(factors):
            ax = axes[i]
            
            # 데이터 플롯
            ax.plot(data.index, data[factor], 
                   color=self.colors.get(factor, '#666666'), linewidth=1.0, alpha=0.8)
            
            # 1.0 기준선
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # 스타일링
            ax.set_ylabel(factor_labels.get(factor, factor), fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # 통계 정보
            mean_val = data[factor].mean()
            std_val = data[factor].std()
            range_val = data[factor].max() - data[factor].min()
            
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nRange: {range_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
            
        # 공통 제목 및 x축 라벨
        fig.suptitle(f'{self.station_id} - Neutron Correction Factors', fontsize=16, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=12)
        
        # x축 날짜 포맷
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def plot_correction_summary(self, data: pd.DataFrame, output_dir: str = None) -> Optional[str]:
        """보정계수 요약 플롯 (박스플롯)"""
        
        with ProcessTimer(self.logger, "Creating correction summary plot"):
            
            correction_factors = ['fi', 'fp', 'fw', 'fb']
            available_factors = [f for f in correction_factors if f in data.columns]
            
            if not available_factors:
                self.logger.warning("No correction factors available for summary plot")
                return None
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 데이터 준비
            correction_data = []
            labels = []
            factor_labels = {
                'fi': 'Incoming\nFlux (fi)',
                'fp': 'Pressure\n(fp)', 
                'fw': 'Humidity\n(fw)',
                'fb': 'Biomass\n(fb)'
            }
            
            for factor in available_factors:
                correction_data.append(data[factor].dropna())
                labels.append(factor_labels.get(factor, factor))
                
            # 박스플롯
            box_colors = [self.colors.get(f, '#666666') for f in available_factors]
            bp = ax.boxplot(correction_data, labels=labels, patch_artist=True, 
                           boxprops=dict(alpha=0.7), medianprops=dict(color='black', linewidth=2))
            
            # 색상 적용
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                
            # 1.0 기준선
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='No correction (1.0)')
            
            # 스타일링
            ax.set_title(f'{self.station_id} - Correction Factors Summary', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Correction Factor', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            
            # 통계 테이블 추가
            stats_text = []
            for i, factor in enumerate(available_factors):
                factor_data = data[factor].dropna()
                stats_text.append(f'{factor}: μ={factor_data.mean():.3f}, σ={factor_data.std():.3f}')
                
            ax.text(0.02, 0.98, '\n'.join(stats_text), 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10)
            
            plt.tight_layout()
            
            if output_dir:
                summary_file = Path(output_dir) / f"{self.station_id}_correction_summary.png"
                fig.savefig(summary_file, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                return str(summary_file)
            else:
                return None
                
    def plot_neutron_statistics(self, data: pd.DataFrame, output_dir: str = None) -> Optional[str]:
        """중성자 카운트 통계 플롯"""
        
        with ProcessTimer(self.logger, "Creating neutron statistics plot"):
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 히스토그램 (원시)
            if 'total_raw_counts' in data.columns:
                axes[0, 0].hist(data['total_raw_counts'].dropna(), bins=50, 
                               color=self.colors['raw_neutron'], alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Raw Neutron Counts Distribution')
                axes[0, 0].set_xlabel('Counts')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
            # 2. 히스토그램 (보정)
            if 'total_corrected_neutrons' in data.columns:
                axes[0, 1].hist(data['total_corrected_neutrons'].dropna(), bins=50, 
                               color=self.colors['corrected_neutron'], alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Corrected Neutron Counts Distribution')
                axes[0, 1].set_xlabel('Counts')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
                
            # 3. 일간 변화량
            if 'total_corrected_neutrons' in data.columns:
                daily_change = data['total_corrected_neutrons'].diff()
                axes[1, 0].hist(daily_change.dropna(), bins=50, 
                               color='orange', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Daily Change in Corrected Counts')
                axes[1, 0].set_xlabel('Change (counts/day)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
                
            # 4. 시간별 패턴 (시간이 있는 경우)
            if hasattr(data.index, 'hour'):
                hourly_mean = data.groupby(data.index.hour)['total_corrected_neutrons'].mean()
                axes[1, 1].plot(hourly_mean.index, hourly_mean.values, 'o-', 
                               color=self.colors['corrected_neutron'], linewidth=2, markersize=6)
                axes[1, 1].set_title('Hourly Pattern (Average)')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Average Counts')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xticks(range(0, 24, 4))
            else:
                # 월별 패턴으로 대체
                if len(data) > 30:
                    monthly_mean = data.groupby(data.index.month)['total_corrected_neutrons'].mean()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    axes[1, 1].plot(monthly_mean.index, monthly_mean.values, 'o-', 
                                   color=self.colors['corrected_neutron'], linewidth=2, markersize=6)
                    axes[1, 1].set_title('Monthly Pattern (Average)')
                    axes[1, 1].set_xlabel('Month')
                    axes[1, 1].set_ylabel('Average Counts')
                    axes[1, 1].set_xticks(monthly_mean.index)
                    axes[1, 1].set_xticklabels([month_names[i-1] for i in monthly_mean.index])
                    axes[1, 1].grid(True, alpha=0.3)
                    
            plt.suptitle(f'{self.station_id} - Neutron Count Statistics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if output_dir:
                stats_file = Path(output_dir) / f"{self.station_id}_neutron_statistics.png"
                fig.savefig(stats_file, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                return str(stats_file)
            else:
                return None


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    import pandas as pd
    
    # 테스트용 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'total_raw_counts': np.random.normal(1000, 50, 100),
        'total_corrected_neutrons': np.random.normal(950, 40, 100),
        'fi': np.random.normal(1.0, 0.05, 100),
        'fp': np.random.normal(1.0, 0.03, 100),
        'fw': np.random.normal(1.0, 0.02, 100)
    }, index=dates)
    
    # NeutronPlotter 테스트
    logger = setup_logger("NeutronPlotter_Test")
    plotter = NeutronPlotter("TEST", logger)
    
    try:
        plot_files = plotter.plot_neutron_timeseries(test_data, show_corrections=True)
        print("✅ NeutronPlotter 테스트 완료!")
        print(f"생성된 플롯: {len(plot_files)}개")
        
    except Exception as e:
        print(f"❌ NeutronPlotter 테스트 실패: {e}")
        import traceback
        traceback.print_exc()