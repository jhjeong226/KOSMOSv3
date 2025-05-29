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
    """CRNP 토양수분 계산을 담당하는 클래스"""
    
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
        
        # 유효깊이 계산을 위한 설정
        self.z_surface = 144  # mm (표면층 깊이)
        self.z_subsurface = 350  # mm (하부층 깊이)
        
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
                
                # 8. 유효깊이 계산
                soil_moisture_data = self._calculate_sensing_depth(soil_moisture_data)
                
                # 9. 토양수분 저장량 계산
                soil_moisture_data = self._calculate_storage(soil_moisture_data)
                
                # 10. 불확실성 계산
                soil_moisture_data = self._calculate_uncertainty(soil_moisture_data)
                
                # 11. 스무딩 적용 (선택사항)
                if self.smoothing_config.get('enabled', False):
                    soil_moisture_data = self._apply_smoothing(soil_moisture_data)
                    
                # 12. 결과 저장
                if output_dir:
                    self._save_results(soil_moisture_data, output_dir)
                    
                self.logger.info(f"Soil moisture calculation completed for {len(soil_moisture_data)} days")
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
        
        with ProcessTimer(self.logger, "Loading CRNP data"):
            
            if not Path(data_path).exists():
                raise FileNotFoundError(f"CRNP data file not found: {data_path}")
                
            # 데이터 로드
            crnp_columns = ['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
            df = pd.read_excel(data_path, names=crnp_columns)
            
            # 타임스탬프 처리
            df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # 유효한 데이터만 유지
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.log_data_summary("CRNP_Raw", len(df),
                                       date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
    def _determine_calculation_period(self, df: pd.DataFrame,
                                    start_str: Optional[str], 
                                    end_str: Optional[str]) -> Tuple[datetime, datetime]:
        """계산 기간 결정"""
        
        if start_str and end_str:
            calc_start = pd.to_datetime(start_str)
            calc_end = pd.to_datetime(end_str)
        else:
            # 전체 데이터 기간 사용
            calc_start = df['timestamp'].min()
            calc_end = df['timestamp'].max()
            
        self.logger.info(f"Calculation period: {calc_start.date()} to {calc_end.date()}")
        return calc_start, calc_end
        
    def _filter_calculation_period(self, df: pd.DataFrame, 
                                 calc_start: datetime, calc_end: datetime) -> pd.DataFrame:
        """계산 기간으로 데이터 필터링"""
        
        mask = (df['timestamp'] >= calc_start) & (df['timestamp'] <= calc_end)
        filtered_df = df[mask].copy()
        
        self.logger.log_data_summary("CRNP_Filtered", len(filtered_df))
        return filtered_df
        
    def _apply_neutron_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """중성자 보정 적용"""
        
        with ProcessTimer(self.logger, "Applying neutron corrections"):
            
            # 원시 중성자 카운트 설정
            df['total_raw_counts'] = df['N_counts']
            
            # 중성자 보정 적용 (캘리브레이션 매개변수 사용)
            corrected_data = self.neutron_corrector.apply_corrections(df, self.calibration_params)
            
            return corrected_data
            
    def _apply_exclusion_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """제외 기간 적용"""
        
        exclude_config = self.calculation_config.get('exclude_periods', {})
        
        if not exclude_config:
            return df
            
        with ProcessTimer(self.logger, "Applying exclusion periods"):
            
            exclude_mask = pd.Series(False, index=df.index)
            
            # 특정 월 제외
            exclude_months = exclude_config.get('months', [])
            if exclude_months:
                month_mask = df['timestamp'].dt.month.isin(exclude_months)
                exclude_mask |= month_mask
                excluded_months_count = month_mask.sum()
                self.logger.info(f"Excluding {excluded_months_count} records from months: {exclude_months}")
                
            # 특정 날짜 범위 제외
            exclude_dates = exclude_config.get('custom_dates', [])
            for date_range in exclude_dates:
                if len(date_range) == 2:
                    start_date = pd.to_datetime(date_range[0])
                    end_date = pd.to_datetime(date_range[1])
                    
                    date_mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                    exclude_mask |= date_mask
                    excluded_dates_count = date_mask.sum()
                    self.logger.info(f"Excluding {excluded_dates_count} records from {start_date.date()} to {end_date.date()}")
                    
            # 제외 기간 적용
            if exclude_mask.any():
                df_cleaned = df[~exclude_mask].copy()
                total_excluded = exclude_mask.sum()
                self.logger.info(f"Total excluded records: {total_excluded} ({total_excluded/len(df)*100:.1f}%)")
                return df_cleaned
            else:
                return df
                
    def _calculate_daily_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """일평균 데이터 계산"""
        
        with ProcessTimer(self.logger, "Calculating daily averages"):
            
            # 수치 컬럼들만 선택
            numeric_columns = ['total_raw_counts', 'total_corrected_neutrons', 'Pa', 'fi', 'fp', 'fw']
            existing_numeric = [col for col in numeric_columns if col in df.columns]
            
            # 일별 그룹화 및 평균 계산
            daily_data = df.resample('D', on='timestamp')[existing_numeric].mean()
            
            # NaN이 있는 행 제거
            daily_data = daily_data.dropna()
            
            self.logger.log_data_summary("CRNP_Daily", len(daily_data))
            return daily_data
            
    def _calculate_vwc(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """체적수분함량 계산"""
        
        with ProcessTimer(self.logger, "Calculating VWC"):
            
            # VWC 계산
            daily_data['VWC'] = crnpy.counts_to_vwc(
                N=daily_data['total_corrected_neutrons'],
                N0=self.N0,
                bulk_density=self.bulk_density,
                Wlat=self.lattice_water,
                Wsoc=0.01  # 토양 유기탄소 (기본값)
            )
            
            # 물리적으로 불가능한 값 제거
            valid_mask = (daily_data['VWC'] >= 0) & (daily_data['VWC'] <= 1)
            invalid_count = (~valid_mask).sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Removing {invalid_count} invalid VWC values")
                daily_data.loc[~valid_mask, 'VWC'] = np.nan
                
            self.logger.info(f"VWC calculated: mean={daily_data['VWC'].mean():.3f}, "
                           f"std={daily_data['VWC'].std():.3f}")
            
            return daily_data
            
    def _calculate_sensing_depth(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """유효깊이 계산"""
        
        with ProcessTimer(self.logger, "Calculating sensing depth"):
            
            try:
                # 유효깊이 계산 (Franz_2012 방법)
                daily_data['sensing_depth'] = crnpy.sensing_depth(
                    theta_v=daily_data['VWC'],
                    pressure=daily_data['Pa'],
                    pressure_ref=daily_data['Pa'].mean(),
                    bulk_density=self.bulk_density,
                    Wlat=self.lattice_water,
                    method="Franz_2012"
                )
                
                avg_depth = daily_data['sensing_depth'].mean()
                self.logger.info(f"Average sensing depth: {avg_depth:.1f} mm")
                
            except Exception as e:
                self.logger.warning(f"Sensing depth calculation failed: {e}")
                daily_data['sensing_depth'] = self.z_surface  # 기본값 사용
                
            return daily_data
            
    def _calculate_storage(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """토양수분 저장량 계산"""
        
        with ProcessTimer(self.logger, "Calculating soil water storage"):
            
            try:
                # 표면층과 하부층 토양수분 계산
                surface_sm = daily_data['VWC']
                subsurface_sm = crnpy.exp_filter(daily_data['VWC'])
                
                # 토양수분 저장량 계산 (mm)
                daily_data['storage'] = (surface_sm * self.z_surface + 
                                       subsurface_sm * self.z_subsurface)
                
                avg_storage = daily_data['storage'].mean()
                self.logger.info(f"Average soil water storage: {avg_storage:.1f} mm")
                
            except Exception as e:
                self.logger.warning(f"Storage calculation failed: {e}")
                daily_data['storage'] = daily_data['VWC'] * self.z_surface  # 단순 계산
                
            return daily_data
            
    def _calculate_uncertainty(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """토양수분 불확실성 계산"""
        
        with ProcessTimer(self.logger, "Calculating uncertainty"):
            
            try:
                # VWC 불확실성 계산
                daily_data['sigma_VWC'] = crnpy.uncertainty_vwc(
                    N=daily_data['total_raw_counts'],
                    N0=self.N0,
                    bulk_density=self.bulk_density,
                    fp=daily_data['fp'],
                    fi=daily_data['fi'],
                    fw=daily_data['fw']
                )
                
                avg_uncertainty = daily_data['sigma_VWC'].mean()
                self.logger.info(f"Average VWC uncertainty: ±{avg_uncertainty:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Uncertainty calculation failed: {e}")
                daily_data['sigma_VWC'] = np.nan
                
            return daily_data
            
    def _apply_smoothing(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """스무딩 적용 (선택사항)"""
        
        with ProcessTimer(self.logger, "Applying smoothing"):
            
            method = self.smoothing_config.get('method', 'savitzky_golay')
            
            if method == 'savitzky_golay':
                window = self.smoothing_config.get('window', 11)
                order = self.smoothing_config.get('order', 3)
                
                try:
                    # VWC에 Savitzky-Golay 필터 적용
                    smoothed_vwc = crnpy.smooth_1d(
                        daily_data['VWC'].dropna(),
                        window=window,
                        order=order,
                        method="savitzky_golay"
                    )
                    
                    daily_data['VWC_smoothed'] = smoothed_vwc
                    self.logger.info(f"Applied Savitzky-Golay smoothing (window={window}, order={order})")
                    
                except Exception as e:
                    self.logger.warning(f"Smoothing failed: {e}")
                    daily_data['VWC_smoothed'] = daily_data['VWC']
                    
            else:
                self.logger.warning(f"Unknown smoothing method: {method}")
                daily_data['VWC_smoothed'] = daily_data['VWC']
                
            return daily_data
            
    def _save_results(self, daily_data: pd.DataFrame, output_dir: str) -> None:
        """결과 저장"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = self.station_config['station_info']['id']
        
        # 결과 파일 저장
        output_file = output_path / f"{station_id}_soil_moisture.xlsx"
        
        # 날짜 범위를 포함한 완전한 인덱스 생성
        full_date_range = pd.date_range(
            start=daily_data.index.min(),
            end=daily_data.index.max(),
            freq='D'
        )
        
        # 완전한 날짜 범위로 재인덱싱 (누락된 날짜는 NaN)
        daily_data_complete = daily_data.reindex(full_date_range)
        
        # Excel 파일로 저장
        self.file_handler.save_dataframe(daily_data_complete, str(output_file), index=True)
        
        self.logger.log_file_operation("save", str(output_file), "success")
        
    def _create_data_summary(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 생성"""
        
        summary = {
            'total_days': len(daily_data),
            'valid_vwc_days': daily_data['VWC'].notna().sum(),
            'date_range': {
                'start': str(daily_data.index.min().date()),
                'end': str(daily_data.index.max().date())
            },
            'vwc_statistics': {
                'mean': float(daily_data['VWC'].mean()),
                'std': float(daily_data['VWC'].std()),
                'min': float(daily_data['VWC'].min()),
                'max': float(daily_data['VWC'].max()),
                'q25': float(daily_data['VWC'].quantile(0.25)),
                'q75': float(daily_data['VWC'].quantile(0.75))
            },
            'sensing_depth': {
                'mean': float(daily_data['sensing_depth'].mean()),
                'min': float(daily_data['sensing_depth'].min()),
                'max': float(daily_data['sensing_depth'].max())
            } if 'sensing_depth' in daily_data.columns else None,
            'storage': {
                'mean': float(daily_data['storage'].mean()),
                'std': float(daily_data['storage'].std())
            } if 'storage' in daily_data.columns else None
        }
        
        return summary
        
    def get_calculation_status(self, output_dir: str) -> Dict[str, Any]:
        """토양수분 계산 상태 확인"""
        
        output_path = Path(output_dir)
        station_id = self.station_config['station_info']['id']
        result_file = output_path / f"{station_id}_soil_moisture.xlsx"
        
        status = {
            'station_id': station_id,
            'calculation_available': result_file.exists(),
            'result_file': str(result_file) if result_file.exists() else None,
            'file_size_mb': round(result_file.stat().st_size / (1024*1024), 2) if result_file.exists() else 0
        }
        
        if result_file.exists():
            try:
                # 결과 파일에서 기본 정보 읽기
                df = pd.read_excel(result_file, index_col=0)
                
                status.update({
                    'data_records': len(df),
                    'date_range': {
                        'start': str(df.index.min().date()) if len(df) > 0 else None,
                        'end': str(df.index.max().date()) if len(df) > 0 else None
                    },
                    'vwc_available': 'VWC' in df.columns,
                    'uncertainty_available': 'sigma_VWC' in df.columns,
                    'storage_available': 'storage' in df.columns
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading result file: {e}")
                
        return status


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    
    # 테스트용 설정
    test_station_config = {
        'station_info': {'id': 'HC'},
        'calibration': {'neutron_monitor': 'ATHN', 'utc_offset': 9}
    }
    
    test_processing_config = {
        'calculation': {
            'exclude_periods': {
                'months': [12, 1, 2],
                'custom_dates': []
            },
            'smoothing': {
                'enabled': False,
                'method': 'savitzky_golay',
                'window': 11,
                'order': 3
            }
        },
        'corrections': {
            'incoming_flux': True,
            'pressure': True,
            'humidity': True,
            'biomass': False
        }
    }
    
    test_calibration_params = {
        'N0_rdt': 1757.86,
        'Pref': 962.93,
        'Aref': 12.57,
        'Iref': 1000.0,
        'soil_bulk_density': 1.44,
        'lattice_water': 0.03
    }
    
    # SoilMoistureCalculator 테스트
    logger = setup_logger("SoilMoistureCalculator_Test")
    calculator = SoilMoistureCalculator(
        test_station_config, test_processing_config, test_calibration_params, logger
    )
    
    print("✅ SoilMoistureCalculator 구현 완료!")
    print("주요 기능:")
    print("  - 중성자 보정 적용")
    print("  - 제외 기간 관리")
    print("  - VWC 계산")
    print("  - 유효깊이 계산")
    print("  - 토양수분 저장량 계산")
    print("  - 불확실성 계산")
    print("  - 스무딩 적용 (선택사항)")