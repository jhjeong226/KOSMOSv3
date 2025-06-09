# src/visualization/soil_moisture_plots.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

class SoilMoisturePlotter:
    """토양수분 시각화 클래스"""
    
    def __init__(self, station_id: str, logger):
        self.station_id = station_id
        self.logger = logger
        self.figure_size = (12, 6)
        self.dpi = 300
        
    def plot_soil_moisture_timeseries(self, sm_data: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None, 
                                    output_dir: str = "") -> Dict[str, str]:
        """토양수분 시계열 플롯"""
        plots = {}
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # VWC 시계열 플롯
            if 'VWC' in sm_data.columns:
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                ax.plot(sm_data.index, sm_data['VWC'], 
                       color='#28A745', linewidth=1.5, alpha=0.8, label='VWC')
                
                # 불확실성 포함
                if 'sigma_VWC' in sm_data.columns:
                    vwc_upper = sm_data['VWC'] + sm_data['sigma_VWC']
                    vwc_lower = sm_data['VWC'] - sm_data['sigma_VWC']
                    ax.fill_between(sm_data.index, vwc_lower, vwc_upper, 
                                   color='#28A745', alpha=0.2, label='±1σ Uncertainty')
                
                ax.set_title(f'{self.station_id} - Volumetric Water Content')
                ax.set_xlabel('Date')
                ax.set_ylabel('VWC (m³/m³)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, max(0.8, sm_data['VWC'].max() * 1.1))
                
                plt.tight_layout()
                plot_file = output_path / f"{self.station_id}_vwc_timeseries.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                plots['vwc_timeseries'] = str(plot_file)
            
            # 센싱 깊이 플롯
            if 'sensing_depth' in sm_data.columns:
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                ax.plot(sm_data.index, sm_data['sensing_depth'], 
                       color='#F18F01', linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'{self.station_id} - CRNP Sensing Depth')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sensing Depth (cm)')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / f"{self.station_id}_sensing_depth.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                plots['sensing_depth'] = str(plot_file)
            
            # 토양수분 저장량 플롯
            if 'storage' in sm_data.columns:
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                ax.plot(sm_data.index, sm_data['storage'], 
                       color='#2E86AB', linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'{self.station_id} - Soil Moisture Storage')
                ax.set_xlabel('Date')
                ax.set_ylabel('Storage (mm)')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / f"{self.station_id}_storage.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                plots['storage'] = str(plot_file)
            
            # 기상조건과 함께 표시 (weather_data가 있는 경우)
            if weather_data is not None and 'VWC' in sm_data.columns:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                
                # VWC
                ax1.plot(sm_data.index, sm_data['VWC'], color='#28A745', linewidth=1.5)
                ax1.set_ylabel('VWC (m³/m³)')
                ax1.set_title(f'{self.station_id} - Soil Moisture with Weather')
                ax1.grid(True, alpha=0.3)
                
                # 온도
                if 'Ta' in weather_data.columns:
                    ax2.plot(weather_data.index, weather_data['Ta'], color='red', linewidth=1)
                    ax2.set_ylabel('Temperature (°C)')
                    ax2.grid(True, alpha=0.3)
                
                # 강수량 (있다면)
                if 'Pa' in weather_data.columns:
                    ax3.plot(weather_data.index, weather_data['Pa'], color='blue', linewidth=1)
                    ax3.set_ylabel('Pressure (hPa)')
                    ax3.set_xlabel('Date')
                    ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / f"{self.station_id}_vwc_weather.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                plots['vwc_weather'] = str(plot_file)
            
            return plots
            
        except Exception as e:
            self.logger.warning(f"Error creating soil moisture plots: {e}")
            return {}
    
    def plot_seasonal_patterns(self, sm_data: pd.DataFrame, output_dir: str) -> Optional[str]:
        """계절별 패턴 분석"""
        try:
            if 'VWC' not in sm_data.columns:
                return None
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 월별 박스플롯
            sm_data['month'] = sm_data.index.month
            monthly_data = [sm_data[sm_data['month'] == month]['VWC'].dropna() 
                           for month in range(1, 13)]
            axes[0,0].boxplot(monthly_data, labels=[f'{i}' for i in range(1, 13)])
            axes[0,0].set_title('Monthly VWC Distribution')
            axes[0,0].set_xlabel('Month')
            axes[0,0].set_ylabel('VWC (m³/m³)')
            
            # 일별 평균
            sm_data['day_of_year'] = sm_data.index.dayofyear
            daily_mean = sm_data.groupby('day_of_year')['VWC'].mean()
            axes[0,1].plot(daily_mean.index, daily_mean.values)
            axes[0,1].set_title('Seasonal VWC Pattern')
            axes[0,1].set_xlabel('Day of Year')
            axes[0,1].set_ylabel('VWC (m³/m³)')
            
            # 연도별 비교 (데이터가 여러 해에 걸쳐 있는 경우)
            sm_data['year'] = sm_data.index.year
            years = sm_data['year'].unique()
            if len(years) > 1:
                for year in years:
                    year_data = sm_data[sm_data['year'] == year]
                    axes[1,0].plot(year_data.index.dayofyear, year_data['VWC'], 
                                  label=f'{year}', alpha=0.7)
                axes[1,0].set_title('Year-to-Year Comparison')
                axes[1,0].set_xlabel('Day of Year')
                axes[1,0].set_ylabel('VWC (m³/m³)')
                axes[1,0].legend()
            
            # VWC 변화율
            sm_data['vwc_change'] = sm_data['VWC'].diff()
            axes[1,1].hist(sm_data['vwc_change'].dropna(), bins=30, alpha=0.7)
            axes[1,1].set_title('VWC Daily Change Distribution')
            axes[1,1].set_xlabel('Daily VWC Change')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plot_file = Path(output_dir) / f"{self.station_id}_seasonal_patterns.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating seasonal patterns plot: {e}")
            return None