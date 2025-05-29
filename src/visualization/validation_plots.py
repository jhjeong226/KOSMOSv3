# src/visualization/validation_plots.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score

from ..core.logger import CRNPLogger, ProcessTimer
from .styles import ModernPlotStyle


class ValidationPlotter:
    """CRNP vs 지점 토양수분 검증 플롯을 생성하는 클래스"""
    
    def __init__(self, station_id: str, logger: Optional[CRNPLogger] = None):
        self.station_id = station_id
        self.logger = logger or CRNPLogger("ValidationPlotter")
        
        # 스타일 설정
        self.style = ModernPlotStyle()
        
        # 플롯 설정
        self.figure_size = (15, 7)
        self.dpi = 300
        
        # 색상 설정
        self.colors = {
            'crnp': '#2E86AB',
            'field': '#A23B72',
            'fit_line': '#F18F01',
            'one_to_one': '#C73E1D',
            'confidence': '#6C757D'
        }
        
    def plot_validation_comparison(self, crnp_data: pd.DataFrame, 
                                 field_data: pd.DataFrame,
                                 output_dir: str = None) -> Dict[str, str]:
        """CRNP vs 지점 토양수분 검증 플롯 생성"""
        
        with ProcessTimer(self.logger, "Creating validation comparison plots"):
            
            # 데이터 매칭
            matched_data = self._match_crnp_field_data(crnp_data, field_data)
            
            if len(matched_data) == 0:
                self.logger.warning("No matching data found for validation")
                return {}
                
            plot_files = {}
            
            # 1. 시계열 비교 플롯
            fig_timeseries = self._plot_timeseries_comparison(matched_data)
            if output_dir:
                timeseries_file = Path(output_dir) / f"{self.station_id}_validation_timeseries.png"
                fig_timeseries.savefig(timeseries_file, dpi=self.dpi, bbox_inches='tight')
                plot_files['validation_timeseries'] = str(timeseries_file)
                plt.close(fig_timeseries)
                
            # 2. 산점도
            fig_scatter = self._plot_scatter_comparison(matched_data)
            if output_dir:
                scatter_file = Path(output_dir) / f"{self.station_id}_validation_scatter.png"
                fig_scatter.savefig(scatter_file, dpi=self.dpi, bbox_inches='tight')
                plot_files['validation_scatter'] = str(scatter_file)
                plt.close(fig_scatter)
                
            # 3. 잔차 분석
            fig_residuals = self._plot_residual_analysis(matched_data)
            if output_dir:
                residuals_file = Path(output_dir) / f"{self.station_id}_validation_residuals.png"
                fig_residuals.savefig(residuals_file, dpi=self.dpi, bbox_inches='tight')
                plot_files['validation_residuals'] = str(residuals_file)
                plt.close(fig_residuals)
                
            # 4. 성능 지표 요약
            fig_metrics = self._plot_performance_metrics(matched_data)
            if output_dir:
                metrics_file = Path(output_dir) / f"{self.station_id}_validation_metrics.png"
                fig_metrics.savefig(metrics_file, dpi=self.dpi, bbox_inches='tight')
                plot_files['validation_metrics'] = str(metrics_file)
                plt.close(fig_metrics)
                
            # 5. 깊이별 비교 (FDR 데이터가 깊이 정보를 포함하는 경우)
            if 'FDR_depth' in field_data.columns:
                fig_depth = self._plot_depth_comparison(crnp_data, field_data)
                if output_dir:
                    depth_file = Path(output_dir) / f"{self.station_id}_validation_depth.png"
                    fig_depth.savefig(depth_file, dpi=self.dpi, bbox_inches='tight')
                    plot_files['validation_depth'] = str(depth_file)
                    plt.close(fig_depth)
                    
            self.logger.info(f"Generated {len(plot_files)} validation plots")
            return plot_files
            
    def _match_crnp_field_data(self, crnp_data: pd.DataFrame, 
                              field_data: pd.DataFrame) -> pd.DataFrame:
        """CRNP와 지점 데이터 매칭"""
        
        # CRNP 데이터에서 VWC 컬럼 확인
        if 'VWC' not in crnp_data.columns:
            self.logger.error("VWC column not found in CRNP data")
            return pd.DataFrame()
            
        # 지점 데이터에서 theta_v 컬럼 확인
        if 'theta_v' not in field_data.columns:
            self.logger.error("theta_v column not found in field data")
            return pd.DataFrame()
            
        # 날짜 기준으로 매칭
        crnp_daily = crnp_data.copy()
        if hasattr(crnp_daily.index, 'date'):
            crnp_daily.index = crnp_daily.index.date
        elif 'Date' in crnp_daily.columns:
            crnp_daily = crnp_daily.set_index('Date')
            
        field_daily = field_data.copy()
        if 'Date' in field_daily.columns:
            field_daily['Date'] = pd.to_datetime(field_daily['Date']).dt.date
            # 일별 평균 계산 (여러 깊이/센서가 있는 경우)
            field_grouped = field_daily.groupby('Date')['theta_v'].mean()
        else:
            field_grouped = field_daily['theta_v']
            
        # 공통 날짜 찾기
        common_dates = crnp_daily.index.intersection(field_grouped.index)
        
        if len(common_dates) == 0:
            self.logger.warning("No common dates found between CRNP and field data")
            return pd.DataFrame()
            
        # 매칭된 데이터 생성
        matched_data = pd.DataFrame({
            'Date': common_dates,
            'CRNP_VWC': crnp_daily.loc[common_dates, 'VWC'],
            'Field_VWC': field_grouped.loc[common_dates]
        }).set_index('Date')
        
        # NaN 값 제거
        matched_data = matched_data.dropna()
        
        self.logger.info(f"Matched {len(matched_data)} data points")
        return matched_data
        
    def _plot_timeseries_comparison(self, matched_data: pd.DataFrame) -> plt.Figure:
        """시계열 비교 플롯"""
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 시계열 플롯
        ax.plot(matched_data.index, matched_data['CRNP_VWC'], 
               color=self.colors['crnp'], linewidth=2.0, label='CRNP', marker='o', markersize=4)
        ax.plot(matched_data.index, matched_data['Field_VWC'], 
               color=self.colors['field'], linewidth=2.0, label='Field Sensors', marker='s', markersize=4)
        
        # 스타일링
        ax.set_title(f'{self.station_id} - CRNP vs Field Sensor Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volumetric Water Content (m³/m³)', fontsize=12)
        
        # x축 날짜 포맷
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 성능 지표 계산 및 표시
        metrics = self._calculate_performance_metrics(
            matched_data['Field_VWC'], matched_data['CRNP_VWC']
        )
        
        metrics_text = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nBias = {metrics['bias']:.3f}"
        ax.text(0.02, 0.98, metrics_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def _plot_scatter_comparison(self, matched_data: pd.DataFrame) -> plt.Figure:
        """산점도 비교 플롯"""
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        x = matched_data['Field_VWC']
        y = matched_data['CRNP_VWC']
        
        # 산점도
        ax.scatter(x, y, color=self.colors['crnp'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 1:1 선
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               color=self.colors['one_to_one'], linestyle='--', linewidth=2, label='1:1 Line')
        
        # 최적 맞춤선
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color=self.colors['fit_line'], linewidth=2, 
               label=f'Best fit (y = {slope:.2f}x + {intercept:.3f})')
        
        # 스타일링
        ax.set_title(f'{self.station_id} - CRNP vs Field Sensor Scatter Plot', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Field Sensor VWC (m³/m³)', fontsize=12)
        ax.set_ylabel('CRNP VWC (m³/m³)', fontsize=12)
        
        # 동일한 축 범위 설정
        ax.set_xlim(min_val * 0.9, max_val * 1.1)
        ax.set_ylim(min_val * 0.9, max_val * 1.1)
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 성능 지표
        metrics = self._calculate_performance_metrics(x, y)
        
        metrics_text = f"""Performance Metrics:
R² = {metrics['r2']:.3f}
RMSE = {metrics['rmse']:.3f}
MAE = {metrics['mae']:.3f}
Bias = {metrics['bias']:.3f}
NSE = {metrics['nse']:.3f}
n = {len(x)}"""
        
        ax.text(0.02, 0.98, metrics_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        return fig
        
    def _plot_residual_analysis(self, matched_data: pd.DataFrame) -> plt.Figure:
        """잔차 분석 플롯"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        x = matched_data['Field_VWC']
        y = matched_data['CRNP_VWC']
        residuals = y - x
        
        # 1. 잔차 vs 예측값
        axes[0, 0].scatter(y, residuals, color=self.colors['crnp'], alpha=0.6, s=50)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].set_xlabel('CRNP VWC (m³/m³)')
        axes[0, 0].set_ylabel('Residuals (CRNP - Field)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 히스토그램
        axes[0, 1].hist(residuals, bins=20, color=self.colors['crnp'], alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals (CRNP - Field)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q 플롯
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 시계열 잔차
        axes[1, 1].plot(matched_data.index, residuals, 
                       color=self.colors['crnp'], linewidth=1.0, marker='o', markersize=3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Residuals Time Series')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Residuals (CRNP - Field)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 잔차 통계
        residual_stats = f"""Residual Statistics:
Mean: {residuals.mean():.4f}
Std: {residuals.std():.4f}
Min: {residuals.min():.4f}
Max: {residuals.max():.4f}"""
        
        axes[1, 1].text(0.02, 0.98, residual_stats, 
                        transform=axes[1, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{self.station_id} - Residual Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _plot_performance_metrics(self, matched_data: pd.DataFrame) -> plt.Figure:
        """성능 지표 요약 플롯"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        x = matched_data['Field_VWC']
        y = matched_data['CRNP_VWC']
        
        # 1. 성능 지표 막대 그래프
        metrics = self._calculate_performance_metrics(x, y)
        metric_names = ['R²', 'NSE', 'Index of Agreement']
        metric_values = [metrics['r2'], metrics['nse'], metrics['ioa']]
        
        colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in metric_values]
        bars = axes[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Model Performance Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 값 라벨 추가
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 오차 지표 막대 그래프
        error_names = ['RMSE', 'MAE', 'Bias']
        error_values = [metrics['rmse'], metrics['mae'], abs(metrics['bias'])]
        
        axes[0, 1].bar(error_names, error_values, color=self.colors['field'], alpha=0.7)
        axes[0, 1].set_title('Error Metrics')
        axes[0, 1].set_ylabel('Value (m³/m³)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 월별 성능 (데이터가 충분한 경우)
        if len(matched_data) > 30:
            monthly_metrics = []
            months = []
            
            for month in range(1, 13):
                monthly_data = matched_data[matched_data.index.month == month]
                if len(monthly_data) >= 5:  # 최소 5개 데이터
                    month_metrics = self._calculate_performance_metrics(
                        monthly_data['Field_VWC'], monthly_data['CRNP_VWC']
                    )
                    monthly_metrics.append(month_metrics['r2'])
                    months.append(month)
                    
            if monthly_metrics:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                axes[1, 0].plot([month_names[m-1] for m in months], monthly_metrics, 
                               'o-', color=self.colors['crnp'], linewidth=2, markersize=6)
                axes[1, 0].set_title('Monthly R² Values')
                axes[1, 0].set_ylabel('R²')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 0].set_title('Monthly Analysis')
            
        # 4. 성능 등급 표시
        axes[1, 1].axis('off')
        
        # 성능 등급 결정
        r2 = metrics['r2']
        rmse = metrics['rmse']
        
        if r2 >= 0.8 and rmse <= 0.05:
            grade = "Excellent"
            grade_color = "green"
        elif r2 >= 0.6 and rmse <= 0.1:
            grade = "Good"
            grade_color = "orange"
        else:
            grade = "Needs Improvement"
            grade_color = "red"
            
        performance_text = f"""
Model Performance Grade:

{grade}

Detailed Metrics:
• R² = {metrics['r2']:.3f}
• RMSE = {metrics['rmse']:.3f} m³/m³
• MAE = {metrics['mae']:.3f} m³/m³
• Bias = {metrics['bias']:.3f} m³/m³
• NSE = {metrics['nse']:.3f}
• Index of Agreement = {metrics['ioa']:.3f}

Data Points: {len(x)}
        """
        
        axes[1, 1].text(0.1, 0.9, performance_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.2))
        
        plt.suptitle(f'{self.station_id} - Performance Metrics Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _plot_depth_comparison(self, crnp_data: pd.DataFrame, 
                             field_data: pd.DataFrame) -> plt.Figure:
        """깊이별 비교 플롯"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 깊이별 데이터 그룹화
        depths = sorted(field_data['FDR_depth'].unique())
        
        if len(depths) > 3:
            depths = depths[:3]  # 최대 3개 깊이만 표시
            
        for i, depth in enumerate(depths):
            depth_data = field_data[field_data['FDR_depth'] == depth]
            
            # 일별 평균 계산
            if 'Date' in depth_data.columns:
                depth_daily = depth_data.groupby('Date')['theta_v'].mean()
            else:
                depth_daily = depth_data['theta_v']
                
            # CRNP 데이터와 매칭
            common_dates = crnp_data.index.intersection(depth_daily.index)
            
            if len(common_dates) > 0:
                crnp_matched = crnp_data.loc[common_dates, 'VWC']
                field_matched = depth_daily.loc[common_dates]
                
                # 산점도
                axes[i].scatter(field_matched, crnp_matched, 
                               color=self.colors['crnp'], alpha=0.6, s=30)
                
                # 1:1 선
                min_val = min(field_matched.min(), crnp_matched.min())
                max_val = max(field_matched.max(), crnp_matched.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 
                            'r--', linewidth=2, label='1:1 Line')
                
                # 성능 지표
                metrics = self._calculate_performance_metrics(field_matched, crnp_matched)
                
                axes[i].set_title(f'{depth}cm Depth\nR² = {metrics["r2"]:.3f}')
                axes[i].set_xlabel(f'Field Sensor VWC at {depth}cm (m³/m³)')
                axes[i].set_ylabel('CRNP VWC (m³/m³)')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                
        plt.suptitle(f'{self.station_id} - Depth-wise Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _calculate_performance_metrics(self, observed: np.ndarray, 
                                     predicted: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        
        if len(observed) == 0 or len(predicted) == 0:
            return {}
            
        try:
            # R²
            r2 = r2_score(observed, predicted)
            
            # RMSE
            rmse = np.sqrt(np.mean((observed - predicted) ** 2))
            
            # MAE
            mae = np.mean(np.abs(observed - predicted))
            
            # Bias
            bias = np.mean(predicted - observed)
            
            # NSE (Nash-Sutcliffe Efficiency)
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Index of Agreement
            numerator = np.sum((observed - predicted) ** 2)
            denominator = np.sum((np.abs(predicted - np.mean(observed)) + 
                                np.abs(observed - np.mean(observed))) ** 2)
            ioa = 1 - (numerator / denominator) if denominator != 0 else 0
            
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'nse': nse,
                'ioa': ioa
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    import pandas as pd
    
    # 테스트용 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # CRNP 데이터
    crnp_data = pd.DataFrame({
        'VWC': 0.3 + 0.1 * np.sin(2 * np.pi * np.arange(100) / 365) + np.random.normal(0, 0.02, 100)
    }, index=dates)
    
    # 지점 데이터 (약간의 노이즈와 편향 추가)
    field_data = pd.DataFrame({
        'Date': dates,
        'theta_v': crnp_data['VWC'] * 1.1 + np.random.normal(0, 0.03, 100),
        'FDR_depth': np.random.choice([10, 30, 60], 100)
    })
    
    # ValidationPlotter 테스트
    logger = setup_logger("ValidationPlotter_Test")
    plotter = ValidationPlotter("TEST", logger)
    
    try:
        plot_files = plotter.plot_validation_comparison(crnp_data, field_data)
        print("✅ ValidationPlotter 테스트 완료!")
        print(f"생성된 플롯: {len(plot_files)}개")
        
    except Exception as e:
        print(f"❌ ValidationPlotter 테스트 실패: {e}")
        import traceback
        traceback.print_exc()