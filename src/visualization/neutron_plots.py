# src/visualization/neutron_plots.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

class NeutronPlotter:
    """중성자 데이터 시각화 클래스"""
    
    def __init__(self, station_id: str, logger):
        self.station_id = station_id
        self.logger = logger
        self.figure_size = (12, 6)
        self.dpi = 300
        
    def plot_neutron_timeseries(self, data: pd.DataFrame, output_dir: str, show_corrections: bool = True) -> Dict[str, str]:
        """중성자 시계열 플롯"""
        plots = {}
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 원시 vs 보정 중성자 비교
            if 'total_raw_counts' in data.columns and 'total_corrected_neutrons' in data.columns:
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                ax.plot(data.index, data['total_raw_counts'], 
                       label='Raw Neutron Counts', linewidth=1.5, alpha=0.8)
                ax.plot(data.index, data['total_corrected_neutrons'], 
                       label='Corrected Neutron Counts', linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'{self.station_id} - Neutron Count Comparison')
                ax.set_xlabel('Date')
                ax.set_ylabel('Neutron Counts')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = output_path / f"{self.station_id}_neutron_comparison.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                plots['neutron_comparison'] = str(plot_file)
                
            return plots
            
        except Exception as e:
            self.logger.warning(f"Error creating neutron plots: {e}")
            return {}
    
    def plot_neutron_statistics(self, data: pd.DataFrame, output_dir: str) -> Optional[str]:
        """중성자 통계 플롯"""
        try:
            if 'total_corrected_neutrons' not in data.columns:
                return None
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 히스토그램
            axes[0,0].hist(data['total_corrected_neutrons'].dropna(), bins=30, alpha=0.7)
            axes[0,0].set_title('Neutron Count Distribution')
            axes[0,0].set_xlabel('Neutron Counts')
            axes[0,0].set_ylabel('Frequency')
            
            # 시계열 플롯
            axes[0,1].plot(data.index, data['total_corrected_neutrons'])
            axes[0,1].set_title('Neutron Count Time Series')
            axes[0,1].set_xlabel('Date')
            axes[0,1].set_ylabel('Neutron Counts')
            
            # 박스플롯
            data['month'] = data.index.month
            monthly_data = [data[data['month'] == month]['total_corrected_neutrons'].dropna() 
                           for month in range(1, 13)]
            axes[1,0].boxplot(monthly_data, labels=[f'M{i}' for i in range(1, 13)])
            axes[1,0].set_title('Monthly Distribution')
            axes[1,0].set_xlabel('Month')
            axes[1,0].set_ylabel('Neutron Counts')
            
            # 일일 변화
            data['daily_change'] = data['total_corrected_neutrons'].diff()
            axes[1,1].plot(data.index, data['daily_change'])
            axes[1,1].set_title('Daily Change in Neutron Counts')
            axes[1,1].set_xlabel('Date')
            axes[1,1].set_ylabel('Change in Counts')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plot_file = Path(output_dir) / f"{self.station_id}_neutron_statistics.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating neutron statistics plot: {e}")
            return None
    
    def plot_correction_summary(self, data: pd.DataFrame, output_dir: str) -> Optional[str]:
        """보정계수 요약 플롯"""
        try:
            correction_factors = ['fi', 'fp', 'fw', 'fb']
            available_factors = [f for f in correction_factors if f in data.columns]
            
            if not available_factors:
                return None
                
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            for factor in available_factors:
                ax.plot(data.index, data[factor], label=factor, linewidth=1.5)
            
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Reference (1.0)')
            ax.set_title(f'{self.station_id} - Correction Factors')
            ax.set_xlabel('Date')
            ax.set_ylabel('Correction Factor')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = Path(output_dir) / f"{self.station_id}_correction_summary.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"Error creating correction summary plot: {e}")
            return None