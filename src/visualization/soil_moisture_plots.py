# src/visualization/soil_moisture_plots.py

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


class SoilMoisturePlotter:
    """토양수분 관련 플롯을 생성하는 클래스"""
    
    def __init__(self, station_id: str, logger: Optional[CRNPLogger] = None):
        self.station_id = station_id
        self.logger = logger or CRNPLogger("SoilMoisturePlotter")
        
        # 스타일 설정
        self.style = ModernPlotStyle()
        
        # 플롯 설정
        self.figure_size = (15, 7)
        self.dpi = 300
        
        # 색상 설정
        self.colors = {
            'vwc': '#2E86AB',
            'vwc_uncertainty': '#A23B72',
            'sensing_depth': '#F18F01',
            'storage': '#28A745',
            'precipitation': '#6C757D',
            'temperature': '#C73E1D'
        }
        
    def plot_soil_moisture_timeseries(self, sm_data: pd.DataFrame, 
                                    weather_data: Optional[pd.DataFrame] = None,
                                    output_dir: str = None) -> Dict[str, str]:
        """토양수분 시계열 플롯 생성"""
        
        with ProcessTimer(self.logger, "Creating soil moisture timeseries plots"):
            
            plot_files = {}
            
            # 1. 기본 VWC 시계열
            if 'VWC' in sm_data.columns:
                fig_vwc = self._plot_vwc_timeseries(sm_data)
                if output_dir:
                    vwc_file = Path(output_dir) / f"{self.station_id}_vwc_timeseries.png"
                    fig_vwc.savefig(vwc_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['vwc_timeseries'] = str(vwc_file)
                    plt.close(fig_vwc)
                    
            # 2. 불확실성 포함 VWC 플롯
            if 'VWC' in sm_data.columns and 'sigma_VWC' in sm_data.columns:
                fig_uncertainty = self._plot_vwc_with_uncertainty(sm_data)
                if output_dir:
                    uncertainty_file = Path(output_dir) / f"{self.station_id}_vwc_uncertainty.png"
                    fig_uncertainty.savefig(uncertainty_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['vwc_uncertainty'] = str(uncertainty_file)
                    plt.close(fig_uncertainty)
                    
            # 3. 유효깊이 시계열
            if 'sensing_depth' in sm_data.columns:
                fig_depth = self._plot_sensing_depth(sm_data)
                if output_dir:
                    depth_file = Path(output_dir) / f"{self.station_id}_sensing_depth.png"
                    fig_depth.savefig(depth_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['sensing_depth'] = str(depth_file)
                    plt.close(fig_depth)
                    
            # 4. 토양수분 저장량
            if 'storage' in sm_data.columns:
                fig_storage = self._plot_storage(sm_data)
                if output_dir:
                    storage_file = Path(output_dir) / f"{self.station_id}_storage.png"
                    fig_storage.savefig(storage_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['storage'] = str(storage_file)
                    plt.close(fig_storage)
                    
            # 5. 기상 조건과 함께 플롯
            if weather_data is not None:
                fig_weather = self._plot_vwc_with_weather(sm_data, weather_data)
                if output_dir:
                    weather_file = Path(output_dir) / f"{self.station_id}_vwc_weather.png"
                    fig_weather.savefig(weather_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['vwc_weather'] = str(weather_file)
                    plt.close(fig_weather)
                    
            # 6. 종합 대시보드
            fig_dashboard = self._plot_soil_moisture_dashboard(sm_data)
            if output_dir:
                dashboard_file = Path(output_dir) / f"{self.station_id}_sm_dashboard.png"
                fig_dashboard.savefig(dashboard_file, dpi=self.dpi, bbox_inches='tight')
                plot_files['sm_dashboard'] = str(dashboard_file)
                plt.close(fig_dashboard)
                
            self.logger.info(f"Generated {len(plot_files)} soil moisture plots")
            return plot_files
            
    def _plot_vwc_timeseries(self, sm_data: pd.DataFrame) -> plt.Figure:
        """기본 VWC 시계열 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # VWC 데이터 플롯
        ax.plot(sm_data.index, sm_data['VWC'], 
               color=self.colors['vwc'], linewidth=1.5, alpha=0.8)
        
        # 스무딩된 데이터가 있으면 함께 플롯
        if 'VWC_smoothed' in sm_data.columns:
            ax.plot(sm_data.index, sm_data['VWC_smoothed'], 
                   color='red', linewidth=2.0, alpha=0.7, linestyle='--', 
                   label='Smoothed')
            ax.legend()
            
        # 스타일링
        ax.set_title(f'{self.station_id} - Volumetric Water Content', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('VWC (m³/m³)', fontsize=12)
        
        # y축 범위 설정
        ax.set_ylim(0, max(0.8, sm_data['VWC'].max() * 1.1))
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        # 그리드
        ax.grid(True, alpha=0.3)
        
        # 통계 정보
        vwc_stats = sm_data['VWC'].describe()
        stats_text = f"Mean: {vwc_stats['mean']:.3f}\nStd: {vwc_stats['std']:.3f}\nRange: {vwc_stats['min']:.3f} - {vwc_stats['max']:.3f}"
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_vwc_with_uncertainty(self, sm_data: pd.DataFrame) -> plt.Figure:
        """불확실성 포함 VWC 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # VWC 중심선
        ax.plot(sm_data.index, sm_data['VWC'], 
               color=self.colors['vwc'], linewidth=2.0, label='VWC')
        
        # 불확실성 밴드
        vwc_upper = sm_data['VWC'] + sm_data['sigma_VWC']
        vwc_lower = sm_data['VWC'] - sm_data['sigma_VWC']
        
        ax.fill_between(sm_data.index, vwc_lower, vwc_upper, 
                       color=self.colors['vwc'], alpha=0.3, 
                       label='±1σ Uncertainty')
        
        # 스타일링
        ax.set_title(f'{self.station_id} - VWC with Uncertainty', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('VWC (m³/m³)', fontsize=12)
        
        # y축 범위 설정
        y_min = max(0, (sm_data['VWC'] - sm_data['sigma_VWC']).min() * 0.9)
        y_max = min(1, (sm_data['VWC'] + sm_data['sigma_VWC']).max() * 1.1)
        ax.set_ylim(y_min, y_max)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 불확실성 통계
        avg_uncertainty = sm_data['sigma_VWC'].mean()
        max_uncertainty = sm_data['sigma_VWC'].max()
        ax.text(0.02, 0.98, f"Avg. Uncertainty: ±{avg_uncertainty:.4f}\nMax. Uncertainty: ±{max_uncertainty:.4f}", 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_sensing_depth(self, sm_data: pd.DataFrame) -> plt.Figure:
        """유효깊이 시계열 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 유효깊이 플롯
        ax.plot(sm_data.index, sm_data['sensing_depth'], 
               color=self.colors['sensing_depth'], linewidth=1.5, alpha=0.8)
        
        # 평균값 수평선
        avg_depth = sm_data['sensing_depth'].mean()
        ax.axhline(y=avg_depth, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_depth:.0f} mm')
        
        # 스타일링
        ax.set_title(f'{self.station_id} - Sensing Depth', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sensing Depth (mm)', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 통계 정보
        depth_stats = sm_data['sensing_depth'].describe()
        stats_text = f"Mean: {depth_stats['mean']:.0f} mm\nStd: {depth_stats['std']:.0f} mm\nRange: {depth_stats['min']:.0f} - {depth_stats['max']:.0f} mm"
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_storage(self, sm_data: pd.DataFrame) -> plt.Figure:
        """토양수분 저장량 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 저장량 플롯
        ax.plot(sm_data.index, sm_data['storage'], 
               color=self.colors['storage'], linewidth=1.5, alpha=0.8)
        
        # 스타일링
        ax.set_title(f'{self.station_id} - Soil Water Storage', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Storage (mm)', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 그리드
        ax.grid(True, alpha=0.3)
        
        # 통계 정보
        storage_stats = sm_data['storage'].describe()
        stats_text = f"Mean: {storage_stats['mean']:.1f} mm\nStd: {storage_stats['std']:.1f} mm\nRange: {storage_stats['min']:.1f} - {storage_stats['max']:.1f} mm"
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_vwc_with_weather(self, sm_data: pd.DataFrame, weather_data: pd.DataFrame) -> plt.Figure:
        """기상 조건과 함께 VWC 플롯"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figure_size[0], 12), sharex=True)
        
        # 1. VWC
        ax1.plot(sm_data.index, sm_data['VWC'], 
                color=self.colors['vwc'], linewidth=1.5)
        ax1.set_ylabel('VWC (m³/m³)', fontsize=11)
        ax1.set_title(f'{self.station_id} - Soil Moisture with Weather Conditions', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. 강수량 (만약 있다면)
        if 'precipitation' in weather_data.columns:
            ax2.bar(weather_data.index, weather_data['precipitation'], 
                   color=self.colors['precipitation'], alpha=0.7, width=1)
            ax2.set_ylabel('Precipitation (mm)', fontsize=11)
        elif 'RH' in weather_data.columns:
            # 강수량이 없으면 습도 사용
            ax2.plot(weather_data.index, weather_data['RH'], 
                    color=self.colors['precipitation'], linewidth=1.0)
            ax2.set_ylabel('Relative Humidity (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. 기온
        if 'Ta' in weather_data.columns:
            ax3.plot(weather_data.index, weather_data['Ta'], 
                    color=self.colors['temperature'], linewidth=1.0)
            ax3.set_ylabel('Temperature (°C)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # x축 날짜 포맷
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_soil_moisture_dashboard(self, sm_data: pd.DataFrame) -> plt.Figure:
        """토양수분 종합 대시보드"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. VWC 시계열
        axes[0, 0].plot(sm_data.index, sm_data['VWC'], 
                       color=self.colors['vwc'], linewidth=1.5)
        axes[0, 0].set_title('Volumetric Water Content')
        axes[0, 0].set_ylabel('VWC (m³/m³)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. VWC 히스토그램
        axes[0, 1].hist(sm_data['VWC'].dropna(), bins=30, 
                       color=self.colors['vwc'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('VWC Distribution')
        axes[0, 1].set_xlabel('VWC (m³/m³)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 유효깊이 (있는 경우)
        if 'sensing_depth' in sm_data.columns:
            axes[1, 0].plot(sm_data.index, sm_data['sensing_depth'], 
                           color=self.colors['sensing_depth'], linewidth=1.5)
            axes[1, 0].set_title('Sensing Depth')
            axes[1, 0].set_ylabel('Depth (mm)')
        else:
            # 대신 월별 평균 플롯
            monthly_vwc = sm_data.groupby(sm_data.index.month)['VWC'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1, 0].bar(range(1, len(monthly_vwc)+1), monthly_vwc.values, 
                          color=self.colors['vwc'], alpha=0.7)
            axes[1, 0].set_title('Monthly Average VWC')
            axes[1, 0].set_ylabel('VWC (m³/m³)')
            axes[1, 0].set_xticks(range(1, len(monthly_vwc)+1))
            axes[1, 0].set_xticklabels([month_names[i-1] for i in monthly_vwc.index])
            
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 통계 박스 (텍스트)
        axes[1, 1].axis('off')
        
        # 통계 정보 생성
        vwc_stats = sm_data['VWC'].describe()
        stats_text = f"""
VWC Statistics:
• Count: {vwc_stats['count']:.0f}
• Mean: {vwc_stats['mean']:.3f}
• Std: {vwc_stats['std']:.3f}
• Min: {vwc_stats['min']:.3f}
• 25%: {vwc_stats['25%']:.3f}
• 50%: {vwc_stats['50%']:.3f}
• 75%: {vwc_stats['75%']:.3f}
• Max: {vwc_stats['max']:.3f}

Data Period:
• Start: {sm_data.index.min().strftime('%Y-%m-%d')}
• End: {sm_data.index.max().strftime('%Y-%m-%d')}
• Days: {len(sm_data)} days
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 전체 제목
        fig.suptitle(f'{self.station_id} - Soil Moisture Dashboard', 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    def plot_seasonal_patterns(self, sm_data: pd.DataFrame, output_dir: str = None) -> Optional[str]:
        """계절별 패턴 분석 플롯"""
        
        with ProcessTimer(self.logger, "Creating seasonal patterns plot"):
            
            if len(sm_data) < 365:  # 1년 미만 데이터
                self.logger.warning("Insufficient data for seasonal analysis")
                return None
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 월별 평균
            monthly_mean = sm_data.groupby(sm_data.index.month)['VWC'].mean()
            monthly_std = sm_data.groupby(sm_data.index.month)['VWC'].std()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            axes[0, 0].errorbar(range(1, 13), monthly_mean.reindex(range(1, 13)), 
                               yerr=monthly_std.reindex(range(1, 13)), 
                               marker='o', linewidth=2, capsize=5,
                               color=self.colors['vwc'])
            axes[0, 0].set_title('Monthly VWC Pattern')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('VWC (m³/m³)')
            axes[0, 0].set_xticks(range(1, 13))
            axes[0, 0].set_xticklabels(month_names)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 계절별 박스플롯
            sm_data['season'] = sm_data.index.month % 12 // 3
            season_names = ['Winter', 'Spring', 'Summer', 'Fall']
            
            seasonal_data = []
            for season in range(4):
                seasonal_data.append(sm_data[sm_data['season'] == season]['VWC'].dropna())
                
            bp = axes[0, 1].boxplot(seasonal_data, labels=season_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(self.colors['vwc'])
                patch.set_alpha(0.7)
                
            axes[0, 1].set_title('Seasonal VWC Distribution')
            axes[0, 1].set_ylabel('VWC (m³/m³)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 연간 패턴 (DOY)
            sm_data['doy'] = sm_data.index.dayofyear
            doy_mean = sm_data.groupby('doy')['VWC'].mean()
            
            axes[1, 0].plot(doy_mean.index, doy_mean.values, 
                           color=self.colors['vwc'], linewidth=1.5)
            axes[1, 0].set_title('Day of Year Pattern')
            axes[1, 0].set_xlabel('Day of Year')
            axes[1, 0].set_ylabel('VWC (m³/m³)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 연도별 비교 (여러 해가 있는 경우)
            yearly_data = sm_data.groupby(sm_data.index.year)['VWC'].mean()
            
            if len(yearly_data) > 1:
                axes[1, 1].bar(yearly_data.index, yearly_data.values, 
                              color=self.colors['vwc'], alpha=0.7)
                axes[1, 1].set_title('Annual Average VWC')
                axes[1, 1].set_xlabel('Year')
                axes[1, 1].set_ylabel('VWC (m³/m³)')
            else:
                # 단일 연도인 경우 일별 변화량 히스토그램
                daily_change = sm_data['VWC'].diff()
                axes[1, 1].hist(daily_change.dropna(), bins=30, 
                               color=self.colors['vwc'], alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Daily VWC Change Distribution')
                axes[1, 1].set_xlabel('Daily Change (m³/m³)')
                axes[1, 1].set_ylabel('Frequency')
                
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{self.station_id} - Seasonal VWC Patterns', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if output_dir:
                seasonal_file = Path(output_dir) / f"{self.station_id}_seasonal_patterns.png"
                fig.savefig(seasonal_file, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                return str(seasonal_file)
            else:
                return None


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    import pandas as pd
    
    # 테스트용 데이터 생성
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    test_data = pd.DataFrame({
        'VWC': 0.3 + 0.1 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 0.02, 365),
        'sigma_VWC': np.random.normal(0.01, 0.002, 365),
        'sensing_depth': 150 + 20 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365),
        'storage': 45 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365)
    }, index=dates)
    
    # SoilMoisturePlotter 테스트
    logger = setup_logger("SoilMoisturePlotter_Test")
    plotter = SoilMoisturePlotter("TEST", logger)
    
    try:
        plot_files = plotter.plot_soil_moisture_timeseries(test_data)
        print("✅ SoilMoisturePlotter 테스트 완료!")
        print(f"생성된 플롯: {len(plot_files)}개")
        
    except Exception as e:
        print(f"❌ SoilMoisturePlotter 테스트 실패: {e}")
        import traceback
        traceback.print_exc()