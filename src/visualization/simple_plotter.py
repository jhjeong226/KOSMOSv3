# src/visualization/simple_plotter.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimpleCRNPPlotter:
    """간단한 CRNP 시각화 클래스"""
    
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.figure_size = (12, 6)
        self.dpi = 300
        
        # 색상 설정
        self.colors = {
            'raw_neutron': '#2E86AB',
            'corrected_neutron': '#A23B72',
            'vwc': '#28A745',
            'field_sm': '#F18F01',
            'crnp_sm': '#2E86AB',
            'fi': '#FF6B6B',
            'fp': '#4ECDC4', 
            'fw': '#45B7D1',
            'fb': '#96CEB4'
        }
    
    def create_all_plots(self, output_dir: str) -> Dict[str, str]:
        """모든 시각화 생성"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. 중성자 카운트 비교 그래프
            neutron_file = self._plot_neutron_comparison(output_path)
            if neutron_file:
                plot_files['neutron_comparison'] = neutron_file
                
            # 2. 보정계수 시계열 그래프
            correction_file = self._plot_correction_factors(output_path)
            if correction_file:
                plot_files['correction_factors'] = correction_file
                
            # 3. VWC 시계열 그래프
            vwc_file = self._plot_vwc_timeseries(output_path)
            if vwc_file:
                plot_files['vwc_timeseries'] = vwc_file
                
            # 4. 토양수분 시계열 비교 그래프
            sm_timeseries_file = self._plot_soil_moisture_comparison(output_path)
            if sm_timeseries_file:
                plot_files['sm_timeseries'] = sm_timeseries_file
                
            # 5. 토양수분 scatter plot
            sm_scatter_file = self._plot_soil_moisture_scatter(output_path)
            if sm_scatter_file:
                plot_files['sm_scatter'] = sm_scatter_file
                
        except Exception as e:
            print(f"Error creating plots: {e}")
            
        return plot_files
    
    def _load_soil_moisture_data(self) -> Optional[pd.DataFrame]:
        """토양수분 데이터 로드"""
        
        sm_file = Path(f"data/output/{self.station_id}/soil_moisture/{self.station_id}_soil_moisture.xlsx")
        
        if not sm_file.exists():
            print(f"Soil moisture file not found: {sm_file}")
            return None
            
        try:
            df = pd.read_excel(sm_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error loading soil moisture data: {e}")
            return None
    
    def _load_validation_data(self) -> Optional[pd.DataFrame]:
        """검증 데이터 로드"""
        
        val_file = Path(f"data/output/{self.station_id}/validation/{self.station_id}_validation_data.xlsx")
        
        if not val_file.exists():
            print(f"Validation file not found: {val_file}")
            return None
            
        try:
            df = pd.read_excel(val_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error loading validation data: {e}")
            return None
    
    def _plot_neutron_comparison(self, output_path: Path) -> Optional[str]:
        """1. 중성자 카운트 비교 그래프"""
        
        df = self._load_soil_moisture_data()
        if df is None:
            return None
            
        # 필요한 컬럼 확인
        raw_col = None
        corrected_col = None
        
        for col in ['total_raw_counts', 'N_counts']:
            if col in df.columns:
                raw_col = col
                break
                
        for col in ['total_corrected_neutrons']:
            if col in df.columns:
                corrected_col = col
                break
                
        if raw_col is None or corrected_col is None:
            print("Neutron count columns not found")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # 원시 및 보정 중성자 카운트 플롯
            ax.plot(df.index, df[raw_col], 
                   color=self.colors['raw_neutron'], linewidth=1.5, 
                   label='Raw Neutron Counts', alpha=0.8)
            ax.plot(df.index, df[corrected_col], 
                   color=self.colors['corrected_neutron'], linewidth=1.5, 
                   label='Corrected Neutron Counts', alpha=0.8)
            
            ax.set_title(f'{self.station_id} - Raw vs Corrected Neutron Counts', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Neutron Counts', fontsize=12)
            
            # 날짜 포맷
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 저장
            plot_file = output_path / f"{self.station_id}_neutron_comparison.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {plot_file.name}")
            return str(plot_file)
            
        except Exception as e:
            print(f"Error creating neutron comparison plot: {e}")
            return None
    
    def _plot_correction_factors(self, output_path: Path) -> Optional[str]:
        """2. 보정계수 시계열 그래프"""
        
        df = self._load_soil_moisture_data()
        if df is None:
            return None
        
        # 보정계수 컬럼 확인
        correction_factors = ['fi', 'fp', 'fw', 'fb']
        available_factors = [f for f in correction_factors if f in df.columns]
        
        if not available_factors:
            print("No correction factors found")
            return None
        
        try:
            n_factors = len(available_factors)
            fig, axes = plt.subplots(n_factors, 1, figsize=(self.figure_size[0], 3*n_factors), sharex=True)
            
            if n_factors == 1:
                axes = [axes]
            
            factor_labels = {
                'fi': 'Incoming Flux Correction (fi)',
                'fp': 'Pressure Correction (fp)',
                'fw': 'Humidity Correction (fw)',
                'fb': 'Biomass Correction (fb)'
            }
            
            for i, factor in enumerate(available_factors):
                ax = axes[i]
                
                # 보정계수 플롯
                ax.plot(df.index, df[factor], 
                       color=self.colors.get(factor, '#666666'), linewidth=1.5)
                
                # 1.0 기준선
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                
                ax.set_ylabel(factor_labels.get(factor, factor), fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # 통계 정보
                mean_val = df[factor].mean()
                std_val = df[factor].std()
                ax.text(0.02, 0.95, f'Mean: {mean_val:.3f}±{std_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 제목 및 x축
            fig.suptitle(f'{self.station_id} - Neutron Correction Factors', 
                        fontsize=14, fontweight='bold')
            axes[-1].set_xlabel('Date', fontsize=12)
            
            # 날짜 포맷
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 저장
            plot_file = output_path / f"{self.station_id}_correction_factors.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {plot_file.name}")
            return str(plot_file)
            
        except Exception as e:
            print(f"Error creating correction factors plot: {e}")
            return None
    
    def _plot_vwc_timeseries(self, output_path: Path) -> Optional[str]:
        """3. VWC 시계열 그래프"""
        
        df = self._load_soil_moisture_data()
        if df is None or 'VWC' not in df.columns:
            print("VWC data not found")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # VWC 플롯
            ax.plot(df.index, df['VWC'], 
                   color=self.colors['vwc'], linewidth=1.5, alpha=0.8)
            
            # 불확실성이 있는 경우 추가
            if 'sigma_VWC' in df.columns:
                vwc_upper = df['VWC'] + df['sigma_VWC']
                vwc_lower = df['VWC'] - df['sigma_VWC']
                ax.fill_between(df.index, vwc_lower, vwc_upper, 
                               color=self.colors['vwc'], alpha=0.2, label='±1σ Uncertainty')
            
            ax.set_title(f'{self.station_id} - Volumetric Water Content', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('VWC (m³/m³)', fontsize=12)
            
            # y축 범위
            ax.set_ylim(0, max(0.8, df['VWC'].max() * 1.1))
            
            # 날짜 포맷
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            ax.grid(True, alpha=0.3)
            
            # 통계 정보
            vwc_stats = df['VWC'].describe()
            stats_text = f"Mean: {vwc_stats['mean']:.3f}\nStd: {vwc_stats['std']:.3f}\nRange: {vwc_stats['min']:.3f} - {vwc_stats['max']:.3f}"
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 저장
            plot_file = output_path / f"{self.station_id}_vwc_timeseries.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {plot_file.name}")
            return str(plot_file)
            
        except Exception as e:
            print(f"Error creating VWC timeseries plot: {e}")
            return None
    
    def _plot_soil_moisture_comparison(self, output_path: Path) -> Optional[str]:
        """4. 토양수분 시계열 비교 그래프"""
        
        val_df = self._load_validation_data()
        if val_df is None:
            print("Validation data not found")
            return None
        
        # 컬럼 확인
        field_col = None
        crnp_col = None
        
        for col in ['field_sm', 'theta_v']:
            if col in val_df.columns:
                field_col = col
                break
                
        for col in ['crnp_vwc', 'VWC']:
            if col in val_df.columns:
                crnp_col = col
                break
        
        if field_col is None or crnp_col is None:
            print("Required columns not found in validation data")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # 시계열 비교
            ax.plot(val_df.index, val_df[field_col], 
                   color=self.colors['field_sm'], linewidth=2, 
                   label='Field Sensors', marker='o', markersize=4)
            ax.plot(val_df.index, val_df[crnp_col], 
                   color=self.colors['crnp_sm'], linewidth=2, 
                   label='CRNP', marker='s', markersize=4)
            
            ax.set_title(f'{self.station_id} - Soil Moisture Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Volumetric Water Content (m³/m³)', fontsize=12)
            
            # 날짜 포맷
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 성능 지표 계산
            try:
                from scipy.stats import pearsonr
                r_value, p_value = pearsonr(val_df[field_col].dropna(), val_df[crnp_col].dropna())
                rmse = np.sqrt(np.mean((val_df[field_col] - val_df[crnp_col]) ** 2))
                
                metrics_text = f"R = {r_value:.3f}\nRMSE = {rmse:.3f}\nn = {len(val_df)}"
                ax.text(0.02, 0.98, metrics_text, 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 저장
            plot_file = output_path / f"{self.station_id}_soil_moisture_comparison.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {plot_file.name}")
            return str(plot_file)
            
        except Exception as e:
            print(f"Error creating soil moisture comparison plot: {e}")
            return None
    
    def _plot_soil_moisture_scatter(self, output_path: Path) -> Optional[str]:
        """5. 토양수분 scatter plot"""
        
        val_df = self._load_validation_data()
        if val_df is None:
            return None
        
        # 컬럼 확인
        field_col = None
        crnp_col = None
        
        for col in ['field_sm', 'theta_v']:
            if col in val_df.columns:
                field_col = col
                break
                
        for col in ['crnp_vwc', 'VWC']:
            if col in val_df.columns:
                crnp_col = col
                break
        
        if field_col is None or crnp_col is None:
            return None
        
        try:
            # 유효한 데이터만 선택
            valid_data = val_df[[field_col, crnp_col]].dropna()
            
            if len(valid_data) == 0:
                print("No valid data for scatter plot")
                return None
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            x = valid_data[field_col]
            y = valid_data[crnp_col]
            
            # 산점도
            ax.scatter(x, y, color=self.colors['crnp_sm'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # 1:1 선
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='1:1 Line')
            
            # 최적 맞춤선
            try:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                line_x = np.array([min_val, max_val])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, color='orange', linewidth=2, 
                       label=f'Best fit (y = {slope:.2f}x + {intercept:.3f})')
            except:
                pass
            
            ax.set_title(f'{self.station_id} - Soil Moisture Scatter Plot', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Field Sensor VWC (m³/m³)', fontsize=12)
            ax.set_ylabel('CRNP VWC (m³/m³)', fontsize=12)
            
            # 동일한 축 범위
            ax.set_xlim(min_val * 0.9, max_val * 1.1)
            ax.set_ylim(min_val * 0.9, max_val * 1.1)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 성능 지표
            try:
                from scipy.stats import pearsonr
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
            
            # 저장
            plot_file = output_path / f"{self.station_id}_soil_moisture_scatter.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {plot_file.name}")
            return str(plot_file)
            
        except Exception as e:
            print(f"Error creating soil moisture scatter plot: {e}")
            return None


def create_simple_visualization(station_id: str, output_dir: str = None) -> Dict[str, str]:
    """간단한 시각화 생성 메인 함수"""
    
    if output_dir is None:
        output_dir = f"data/output/{station_id}/visualization"
    
    plotter = SimpleCRNPPlotter(station_id)
    plot_files = plotter.create_all_plots(output_dir)
    
    print(f"\n📊 Visualization Summary:")
    print(f"   Generated {len(plot_files)} plots")
    print(f"   Output directory: {output_dir}")
    
    for plot_type, file_path in plot_files.items():
        print(f"   ✅ {plot_type}: {Path(file_path).name}")
    
    return plot_files


if __name__ == "__main__":
    # 테스트
    station_id = "PC"
    output_dir = f"data/output/{station_id}/visualization"
    
    try:
        plot_files = create_simple_visualization(station_id, output_dir)
        print("✅ Simple visualization test completed!")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()