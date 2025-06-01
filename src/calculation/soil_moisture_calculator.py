# src/calculation/soil_moisture_calculator.py

import pandas as pd
import numpy as np
import crnpy
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler
from ..calibration.neutron_correction import NeutronCorrector


class SoilMoistureCalculator:
    """CRNP 토양수분 계산을 담당하는 클래스 - 간소화된 버전"""
    
    def __init__(self, station_config: Dict, processing_config: Dict,
                 calibration_params: Dict, logger: Optional[CRNPLogger] = None):
        self.station_config = station_config
        self.processing_config = processing_config
        self.calibration_params = calibration_params
        self.logger = logger or CRNPLogger("SoilMoistureCalculator")
        
        # 종속 모듈 초기화
        self.file_handler = FileHandler(self.logger)
        self.neutron_corrector = NeutronCorrector(station_config, processing_config, self.logger)
        
        # 계산 설정
        self.calculation_config = self.processing_config.get('calculation', {})
        self.smoothing_config = self.calculation_config.get('smoothing', {})
        
        # 캘리브레이션 매개변수
        self.N0 = calibration_params['N0_rdt']
        self.bulk_density = calibration_params['soil_bulk_density']
        self.lattice_water = calibration_params.get('lattice_water', 0.03)
        
    def calculate_soil_moisture(self, crnp_data_path: str, 
                              calculation_start: Optional[str] = None,
                              calculation_end: Optional[str] = None,
                              output_dir: str = None) -> Dict[str, Any]:
        """토양수분 계산 전체 프로세스"""
        
        with ProcessTimer(self.logger, "Soil Moisture Calculation"):
            
            try:
                # 1. CRNP 데이터 로드
                crnp_data = self._load_crnp_data(crnp_data_path)
                
                # 2. 계산 기간 설정
                calc_start, calc_end = self._determine_calculation_period(
                    crnp_data, calculation_start, calculation_end
                )
                
                # 3. 기간별 데이터 필터링
                filtered_data = self._filter_calculation_period(crnp_data, calc_start, calc_end)
                
                # 4. 중성자 보정 적용
                corrected_data = self._apply_neutron_corrections(filtered_data)
                
                # 5. 제외 기간 적용
                cleaned_data = self._apply_exclusion_periods(corrected_data)
                
                # 6. 일평균 계산
                daily_data = self._calculate_daily_averages(cleaned_data)
                
                # 7. 토양수분 계산
                soil_moisture_data = self._calculate_vwc(daily_data)
                
                # 8. 토양수분 저장량 계산 (고정 깊이 사용)
                soil_moisture_data = self._calculate_storage_simple(soil_moisture_data)
                
                # 9. 불확실성 계산
                soil_moisture_data = self._calculate_uncertainty(soil_moisture_data)
                
                # 10. 스무딩 적용 (선택사항)
                if self.smoothing_config.get('enabled', False):
                    soil_moisture_data = self._apply_smoothing(soil_moisture_data)
                    
                # 11. 결과 저장
                if output_dir:
                    self._save_results(soil_moisture_data, output_dir)
                    
                self.logger.info(f"Soil moisture calculation completed: {len(soil_moisture_data)} days processed")
                return {
                    'soil_moisture_data': soil_moisture_data,
                    'calculation_period': {'start': calc_start, 'end': calc_end},
                    'data_summary': self._create_data_summary(soil_moisture_data)
                }
                
            except Exception as e:
                self.logger.log_error_with_context(e, "Soil moisture calculation")
                raise
                
    def _load_crnp_data(self, data_path: str) -> pd.DataFrame:
        """CRNP 데이터 로드"""
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"CRNP data file not found: {data_path}")
            
        # 데이터 로드
        df = pd.read_excel(data_path)
        
        # 타임스탬프 처리
        timestamp_col = None
        for col in ['timestamp', 'Timestamp', 'Date']:
            if col in df.columns:
                timestamp_col = col
                break
                
        if timestamp_col is None:
            timestamp_col = df.columns[0]
            
        # 타임스탬프 변환
        try:
            df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            if df['timestamp'].isna().all():
                raise ValueError(f"All timestamp values are invalid in column {timestamp_col}")
                
        except Exception as e:
            raise ValueError(f"Cannot convert timestamp column {timestamp_col}")
            
        # 유효한 데이터만 유지
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after timestamp processing")
            
        self.logger.info(f"Loaded CRNP data: {len(df)} records")
        return df
        
    def _determine_calculation_period(self, df: pd.DataFrame,
                                    start_str: Optional[str], 
                                    end_str: Optional[str]) -> Tuple[str, str]:
        """계산 기간 결정"""
        
        if start_str and end_str:
            calc_start = start_str
            calc_end = end_str
        else:
            calc_start = df['timestamp'].min().strftime('%Y-%m-%d')
            calc_end = df['timestamp'].max().strftime('%Y-%m-%d')
            
        self.logger.info(f"Calculation period: {calc_start} to {calc_end}")
        return calc_start, calc_end
        
    def _filter_calculation_period(self, df: pd.DataFrame, 
                                 calc_start: str, calc_end: str) -> pd.DataFrame:
        """계산 기간으로 데이터 필터링"""
        
        start_date = pd.to_datetime(calc_start)
        end_date = pd.to_datetime(calc_end)
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df[mask].copy()
        
        return filtered_df
        
    def _apply_neutron_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """중성자 보정 적용"""
        
        # 원시 중성자 카운트 설정
        neutron_col = None
        for col in ['N_counts', 'total_raw_counts']:
            if col in df.columns:
                neutron_col = col
                break
                
        if neutron_col is None:
            raise ValueError("No neutron counts column found")
            
        df['total_raw_counts'] = df[neutron_col]
        
        # 중성자 보정 적용
        try:
            corrected_data = self.neutron_corrector.apply_corrections(df, self.calibration_params)
        except Exception as e:
            self.logger.warning(f"Neutron correction failed, using raw counts: {e}")
            
            # 기본 보정만 적용
            corrected_data = df.copy()
            corrected_data['fi'] = 1.0
            corrected_data['fp'] = 1.0 
            corrected_data['fw'] = 1.0
            corrected_data['fb'] = 1.0
            corrected_data['total_corrected_neutrons'] = corrected_data['total_raw_counts']
            
        return corrected_data
        
    def _apply_exclusion_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """제외 기간 적용"""
        
        exclude_config = self.calculation_config.get('exclude_periods', {})
        
        if not exclude_config:
            return df
            
        exclude_mask = pd.Series(False, index=df.index)
        
        # 특정 월 제외
        exclude_months = exclude_config.get('months', [])
        if exclude_months:
            month_mask = df['timestamp'].dt.month.isin(exclude_months)
            exclude_mask |= month_mask
            
        # 특정 날짜 범위 제외
        exclude_dates = exclude_config.get('custom_dates', [])
        for date_range in exclude_dates:
            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                
                date_mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                exclude_mask |= date_mask
                
        # 제외 기간 적용
        if exclude_mask.any():
            df_cleaned = df[~exclude_mask].copy()
            excluded_count = exclude_mask.sum()
            self.logger.info(f"Excluded {excluded_count} records")
            return df_cleaned
        else:
            return df
            
    def _calculate_daily_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """일평균 데이터 계산"""
        
        # 날짜별 그룹화를 위한 date 컬럼 생성
        df['date'] = df['timestamp'].dt.date
        
        # 수치 컬럼들만 선택
        numeric_columns = ['total_raw_counts', 'total_corrected_neutrons', 'Pa', 'fi', 'fp', 'fw']
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        
        if not existing_numeric:
            raise ValueError("No numeric columns found for daily averaging")
            
        # 일별 그룹화 및 평균 계산
        daily_data = df.groupby('date')[existing_numeric].mean().reset_index()
        
        # 인덱스를 날짜로 설정
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.set_index('date')
        
        # NaN이 있는 행 제거
        daily_data = daily_data.dropna()
        
        self.logger.info(f"Daily averages calculated: {len(daily_data)} days")
        return daily_data
        
    def _calculate_vwc(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """체적수분함량 계산"""
        
        # VWC 계산
        daily_data['VWC'] = crnpy.counts_to_vwc(
            daily_data['total_corrected_neutrons'],
            N0=self.N0,
            bulk_density=self.bulk_density,
            Wlat=self.lattice_water,
            Wsoc=0.01
        )
        
        # 물리적으로 불가능한 값 제거
        valid_mask = (daily_data['VWC'] >= 0) & (daily_data['VWC'] <= 1)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Removed {invalid_count} invalid VWC values")
            daily_data.loc[~valid_mask, 'VWC'] = np.nan
            
        valid_vwc = daily_data['VWC'].dropna()
        if len(valid_vwc) > 0:
            self.logger.info(f"VWC range: {valid_vwc.min():.3f} - {valid_vwc.max():.3f} (mean: {valid_vwc.mean():.3f})")
        
        return daily_data
        
    def _calculate_storage_simple(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """토양수분 저장량 계산 - 고정 깊이 사용"""
        
        # 고정 깊이 200mm 사용 (sensing depth 계산 제거)
        fixed_depth = 200  # mm
        
        try:
            # 토양수분 저장량 계산 (mm)
            daily_data['storage'] = daily_data['VWC'] * fixed_depth
            
            valid_storage = daily_data['storage'].dropna()
            if len(valid_storage) > 0:
                self.logger.info(f"Storage calculated using fixed depth ({fixed_depth}mm)")
                
        except Exception as e:
            self.logger.warning(f"Storage calculation failed: {e}")
            daily_data['storage'] = np.nan
            
        return daily_data
        
    def _calculate_uncertainty(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """토양수분 불확실성 계산"""
        
        try:
            daily_data['sigma_VWC'] = crnpy.uncertainty_vwc(
                daily_data['total_raw_counts'],
                N0=self.N0,
                bulk_density=self.bulk_density,
                fp=daily_data['fp'],
                fi=daily_data['fi'],
                fw=daily_data['fw']
            )
            
        except Exception as e:
            self.logger.warning(f"Uncertainty calculation failed: {e}")
            daily_data['sigma_VWC'] = np.nan
            
        return daily_data
        
    def _apply_smoothing(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """스무딩 적용"""
        
        method = self.smoothing_config.get('method', 'savitzky_golay')
        
        if method == 'savitzky_golay':
            window = self.smoothing_config.get('window', 11)
            order = self.smoothing_config.get('order', 3)
            
            try:
                valid_vwc = daily_data['VWC'].dropna()
                if len(valid_vwc) >= window:
                    smoothed_vwc = crnpy.smooth_1d(
                        valid_vwc,
                        window=window,
                        order=order,
                        method="savitzky_golay"
                    )
                    
                    daily_data['VWC_smoothed'] = smoothed_vwc
                    self.logger.info(f"Applied smoothing (window={window})")
                else:
                    daily_data['VWC_smoothed'] = daily_data['VWC']
                
            except Exception as e:
                self.logger.warning(f"Smoothing failed: {e}")
                daily_data['VWC_smoothed'] = daily_data['VWC']
                
        return daily_data
        
    def _save_results(self, daily_data: pd.DataFrame, output_dir: str) -> None:
        """결과 저장"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = self.station_config['station_info']['id']
        output_file = output_path / f"{station_id}_soil_moisture.xlsx"
        
        # 날짜 범위를 포함한 완전한 인덱스 생성
        if len(daily_data) > 0:
            full_date_range = pd.date_range(
                start=daily_data.index.min(),
                end=daily_data.index.max(),
                freq='D'
            )
            daily_data_complete = daily_data.reindex(full_date_range)
        else:
            daily_data_complete = daily_data
        
        # Excel 파일로 저장
        self.file_handler.save_dataframe(daily_data_complete, str(output_file), index=True)
        self.logger.info(f"Results saved: {output_file}")
        
    def _create_data_summary(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 생성"""
        
        if len(daily_data) == 0:
            return {
                'total_days': 0,
                'valid_vwc_days': 0,
                'date_range': {'start': None, 'end': None},
                'vwc_statistics': {}
            }
        
        valid_vwc = daily_data['VWC'].dropna()
        
        summary = {
            'total_days': len(daily_data),
            'valid_vwc_days': len(valid_vwc),
            'date_range': {
                'start': str(daily_data.index.min().date()),
                'end': str(daily_data.index.max().date())
            },
            'vwc_statistics': {},
            'storage': None
        }
        
        # VWC 통계
        if len(valid_vwc) > 0:
            summary['vwc_statistics'] = {
                'mean': float(valid_vwc.mean()),
                'std': float(valid_vwc.std()),
                'min': float(valid_vwc.min()),
                'max': float(valid_vwc.max()),
                'q25': float(valid_vwc.quantile(0.25)),
                'q75': float(valid_vwc.quantile(0.75))
            }
        
        # 저장량 통계
        if 'storage' in daily_data.columns:
            valid_storage = daily_data['storage'].dropna()
            if len(valid_storage) > 0:
                summary['storage'] = {
                    'mean': float(valid_storage.mean()),
                    'std': float(valid_storage.std())
                }
        
        return summary