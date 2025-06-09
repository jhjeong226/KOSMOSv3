# src/visualization/validation_plots.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import seaborn as sns
from scipy.stats import pearsonr, linregress

plt.style.use('seaborn-v0_8-whitegrid')

class ValidationPlotter:
    """검증 플롯 클래스"""
    
    def __init__(self, station_id: str, logger):
        self.station_id = station_id
        self.logger = logger
        self.figure_size = (12, 6)
        self.dpi = 300
        
    def plot_validation_comparison(self, sm_data: pd.DataFrame, fdr_data: pd.DataFrame, 
                                 output_dir: str) -> Dict[str, str]:
        """검증 비교 플롯"""
        plots = {}
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 데이터 병합 및 정리
            validation_data = self._prepare_validation_data(sm_data, fdr_data)
            
            if validation_data is None or len(validation_data) == 0:
                self.logger.warning("No validation data available for plotting")
                return {}
            
            # 시계열 비교 플롯
            timeseries_plot = self._plot_validation_timeseries(validation_data, output_path)
            if timeseries_plot:
                plots['validation_timeseries'] = timeseries_plot
            
            # 산점도 플롯
            scatter_plot = self._plot_validation_scatter(validation_data, output_path)
            if scatter_plot:
                plots['validation_scatter'] = scatter_plot
            
            # 잔차 분석 플롯
            residuals_plot = self._plot_residuals_analysis(validation_data, output_path)
            if residuals_plot:
                plots['validation_residuals'] = residuals_plot
            
            # 성능지표 요약 플롯
            metrics_plot = self._plot_performance_metrics(validation_data, output_path)
            if metrics_plot:
                plots['validation_metrics'] = metrics_plot
            
            return plots
            
        except Exception as e:
            self.logger.warning(f"Error creating validation plots: {e}")
            return {}
    
    def _prepare_validation_data(self, sm_data: pd.DataFrame, fdr_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """검증용 데이터 준비"""
        try:
            # FDR 데이터 날짜 인덱스 설정
            if 'Date' in fdr_data.columns:
                fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
                fdr_data = fdr_data.set_index('Date')
            
            # VWC 컬럼 확인
            if 'VWC' not in sm_data.columns:
                return None
            
            # FDR 데이터에서 토양수분 컬럼 찾기
            soil_moisture_cols = []
            for col in fdr_data.columns:
                if any(keyword in col.lower() for keyword in ['vwc', 'moisture', 'theta']):
                    soil_moisture_cols.append(col)
            
            if not soil_moisture_cols:
                # 기본적으로 숫자 컬럼들을 토양수분으로 간주
                numeric_cols = fdr_data.select_dtypes(include=[np.number]).columns
                soil_moisture_cols = [col for col in numeric_cols if col not in ['Year', 'Month', 'Day']]
            
            if not soil_moisture_cols:
                return None
            
            # 평균 FDR 값 계산
            fdr_data['field_sm'] = fdr_data[soil_moisture_cols].mean(axis=1)
            
            # 데이터 병합
            validation_data = pd.DataFrame()
            validation_data['crnp_vwc'] = sm_data['VWC']
            validation_data['field_sm'] = fdr_data['field_sm']
            
            # 유효한 데이터만 선택
            validation_data = validation_data.dropna()
            
            return validation_data
            
        except Exception as e:
            self.logger.warning(f"Error preparing validation data: {e}")
            return None
    
    def _plot_validation_timeseries(self, data: pd.DataFrame, output_path: Path) -> Optional[str]:
        """시계열 비교 플롯"""
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ax.plot(data.index, data['field_sm'], 
                   color='#F18F01', linewidth=2, marker='o', markersize=4,
                   label='Field Sensors', alpha=0.8)
            ax.plot(data.index, data['crnp_vwc'], 
                   color='#2E86AB', linewidth=2, marker='s', markersize=4,
                   label='CRNP', alpha=0.8)
            
            ax.set_title(f'{self.station_id} - Soil Moisture Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volumetric Water Content (m³/m³)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 성능 지표 계산 및 표시
            try:
                r_value, p_value = pearsonr(data['field_sm'], data['crnp_vwc'])
                rmse = np.sqrt(np.mean((data['field_sm'] - data['crnp_vwc']) ** 2))
                
                metrics_text = f"R = {r_value:.3f}\nRMSE = {rmse:.3f}\nn = {len(data)}"
                ax.text(0.02, 0.98, metrics_text, 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
            
            plt.tight_layout()
            plot_file = output_path / f"{self.station_id}_validation_timeseries.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating validation timeseries plot: {e}")
            return None
    
    def _plot_validation_scatter(self, data: pd.DataFrame, output_path: Path) -> Optional[str]:
        """산점도 플롯"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            x = data['field_sm']
            y = data['crnp_vwc']
            
            # 산점도
            ax.scatter(x, y, color='#2E86AB', alpha=0.6, s=50, 
                      edgecolors='black', linewidth=0.5)
            
            # 1:1 선
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='1:1 Line')
            
            # 최적 맞춤선
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                line_x = np.array([min_val, max_val])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, color='orange', linewidth=2, 
                       label=f'Best fit (y = {slope:.2f}x + {intercept:.3f})')
            except:
                pass
            
            ax.set_title(f'{self.station_id} - Soil Moisture Scatter Plot')
            ax.set_xlabel('Field Sensor VWC (m³/m³)')
            ax.set_ylabel('CRNP VWC (m³/m³)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 동일한 축 범위
            ax.set_xlim(min_val * 0.9, max_val * 1.1)
            ax.set_ylim(min_val * 0.9, max_val * 1.1)
            
            # 성능 지표
            try:
                r_value, p_value = pearsonr(x, y)
                rmse = np.sqrt(np.mean((x - y) ** 2))
                mae = np.mean(np.abs(x - y))
                bias = np.mean(y - x)
                
                metrics_text = f"""Performance Metrics:
R = {r_value:.3f}
RMSE = {rmse:.3f}
MAE = {mae:.3f}
Bias = {bias:.3f}
n = {len(x)}"""
                
                ax.text(0.02, 0.98, metrics_text, 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            except:
                pass
            
            plt.tight_layout()
            plot_file = output_path / f"{self.station_id}_validation_scatter.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating validation scatter plot: {e}")
            return None
    
    def _plot_residuals_analysis(self, data: pd.DataFrame, output_path: Path) -> Optional[str]:
        """잔차 분석 플롯"""
        try:
            residuals = data['crnp_vwc'] - data['field_sm']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 잔차 시계열
            axes[0,0].plot(data.index, residuals, 'o-', alpha=0.7)
            axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0,0].set_title('Residuals Time Series')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Residuals (CRNP - Field)')
            axes[0,0].grid(True, alpha=0.3)
            
            # 잔차 히스토그램
            axes[0,1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[0,1].set_title('Residuals Distribution')
            axes[0,1].set_xlabel('Residuals')
            axes[0,1].set_ylabel('Frequency')
            
            # 잔차 vs 예측값
            axes[1,0].scatter(data['crnp_vwc'], residuals, alpha=0.6)
            axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1,0].set_title('Residuals vs Predicted')
            axes[1,0].set_xlabel('CRNP VWC')
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].grid(True, alpha=0.3)
            
            # Q-Q 플롯 (정규성 검증)
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1,1])
            axes[1,1].set_title('Q-Q Plot (Normality Check)')
            
            plt.tight_layout()
            plot_file = output_path / f"{self.station_id}_validation_residuals.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating residuals plot: {e}")
            return None
    
    def _plot_performance_metrics(self, data: pd.DataFrame, output_path: Path) -> Optional[str]:
        """성능지표 요약 플롯"""
        try:
            # 성능 지표 계산
            x = data['field_sm']
            y = data['crnp_vwc']
            
            r_value, p_value = pearsonr(x, y)
            rmse = np.sqrt(np.mean((x - y) ** 2))
            mae = np.mean(np.abs(x - y))
            bias = np.mean(y - x)
            
            # Nash-Sutcliffe efficiency
            ss_res = np.sum((x - y) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            nse = 1 - (ss_res / ss_tot)
            
            # Percent bias
            pbias = 100 * np.sum(y - x) / np.sum(x)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['R', 'RMSE', 'MAE', 'Bias', 'NSE', 'PBIAS']
            values = [r_value, rmse, mae, bias, nse, pbias]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
            
            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{self.station_id} - Validation Performance Metrics')
            ax.set_ylabel('Metric Value')
            ax.grid(True, alpha=0.3)
            
            # 기준선 추가
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_path / f"{self.station_id}_validation_metrics.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating performance metrics plot: {e}")
            return None