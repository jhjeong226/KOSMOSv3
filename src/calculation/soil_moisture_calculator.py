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
        
    def _determine_calculation_period(self, crnp_data: pd.DataFrame,
                                          start_str: Optional[str], 
                                          end_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """계산 기간 결정 - 수정된 버전"""
        
        if start_str and end_str:
            # 1. 스크립트에서 직접 지정 (최우선)
            self.logger.info(f"Using script-specified calculation period: {start_str} to {end_str}")
            return start_str, end_str
            
        # 2. YAML 설정에서 가져오기
        calc_config = self.calculation_config
        yaml_start = calc_config.get('default_start_date')
        yaml_end = calc_config.get('default_end_date')
        
        if yaml_start and yaml_end:
            self.logger.info(f"Using YAML calculation period: {yaml_start} to {yaml_end}")
            return yaml_start, yaml_end
            
        # 3. 전체 데이터 기간 사용
        self.logger.info("Using full data period for calculation")
        return None, None
        
    def _filter_calculation_period(self, df: pd.DataFrame, 
                                       calc_start: Optional[str], 
                                       calc_end: Optional[str]) -> pd.DataFrame:
        """계산 기간으로 데이터 필터링 - 수정된 버전"""
        
        if calc_start is None or calc_end is None:
            self.logger.info("No specific calculation period, using all data")
            return df
            
        try:
            start_date = pd.to_datetime(calc_start)
            end_date = pd.to_datetime(calc_end)
            
            # 타임스탬프 컬럼 확인
            if 'timestamp' not in df.columns:
                raise ValueError("timestamp column not found")
                
            # 기간 필터링
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df[mask].copy()
            
            initial_count = len(df)
            filtered_count = len(filtered_df)
            
            self.logger.info(f"Filtered to calculation period: {filtered_count}/{initial_count} records retained")
            self.logger.info(f"Calculation period: {start_date.date()} to {end_date.date()}")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering calculation period: {e}")
            self.logger.warning("Using all available data")
            return df
        
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
        """제외 기간 마킹 - 수정된 버전 (데이터는 유지하되 표시만)"""
        
        exclude_config = self.calculation_config.get('exclude_periods', {})
        
        if not exclude_config:
            df['exclude_from_calculation'] = False
            self.logger.info("No exclusion periods configured")
            return df
            
        exclude_mask = pd.Series(False, index=df.index)
        exclusion_details = []
        
        # 특정 월 제외
        exclude_months = exclude_config.get('months', [])
        if exclude_months:
            month_mask = df['timestamp'].dt.month.isin(exclude_months)
            exclude_mask |= month_mask
            month_count = month_mask.sum()
            if month_count > 0:
                month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                             7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
                excluded_month_names = [month_names[m] for m in exclude_months if m in month_names]
                exclusion_details.append(f"Months {excluded_month_names}: {month_count} records")
                
        # 특정 날짜 범위 제외
        exclude_dates = exclude_config.get('custom_dates', [])
        for date_range in exclude_dates:
            if len(date_range) == 2:
                try:
                    start_date = pd.to_datetime(date_range[0])
                    end_date = pd.to_datetime(date_range[1])
                    
                    date_mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                    exclude_mask |= date_mask
                    
                    range_count = date_mask.sum()
                    if range_count > 0:
                        exclusion_details.append(f"Date range {date_range[0]} to {date_range[1]}: {range_count} records")
                        
                except Exception as e:
                    self.logger.warning(f"Invalid date range {date_range}: {e}")
                    
        # 제외 기간 마킹
        df['exclude_from_calculation'] = exclude_mask
        
        total_excluded = exclude_mask.sum()
        total_records = len(df)
        
        if total_excluded > 0:
            exclusion_pct = (total_excluded / total_records) * 100
            self.logger.info(f"Marked {total_excluded}/{total_records} records for exclusion ({exclusion_pct:.1f}%)")
            for detail in exclusion_details:
                self.logger.info(f"  - {detail}")
        else:
            self.logger.info("No records marked for exclusion")
            
        return df
            
    def _calculate_daily_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """일평균 데이터 계산 - 수정된 버전 (제외 기간도 포함하여 계산)"""
        
        # 날짜별 그룹화를 위한 date 컬럼 생성
        df['date'] = df['timestamp'].dt.date
        
        # 기본 수치 컬럼들
        numeric_columns = ['total_raw_counts', 'total_corrected_neutrons', 'Pa', 'fi', 'fp', 'fw']
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        
        # 제외 여부도 함께 그룹화
        groupby_columns = existing_numeric + ['exclude_from_calculation']
        
        if not existing_numeric:
            raise ValueError("No numeric columns found for daily averaging")
            
        # 일별 그룹화 및 평균 계산
        agg_dict = {}
        for col in existing_numeric:
            agg_dict[col] = 'mean'
        # 제외 여부는 any() 사용 (하나라도 제외되면 해당 일은 제외)
        agg_dict['exclude_from_calculation'] = 'any'
        
        daily_data = df.groupby('date').agg(agg_dict).reset_index()
        
        # 인덱스를 날짜로 설정
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.set_index('date')
        
        # NaN이 있는 행도 유지 (제외 기간을 위해)
        total_days = len(daily_data)
        excluded_days = daily_data['exclude_from_calculation'].sum()
        
        self.logger.info(f"Daily averages calculated: {total_days} days ({excluded_days} marked for exclusion)")
        return daily_data
    
    def _calculate_vwc(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """체적수분함량 계산 - 수정된 버전 (제외 기간 처리 개선)"""
        
        # 모든 날짜에 대해 일단 VWC 계산
        daily_data['VWC'] = crnpy.counts_to_vwc(
            daily_data['total_corrected_neutrons'],
            N0=self.N0,
            bulk_density=self.bulk_density,
            Wlat=self.lattice_water,
            Wsoc=0.01
        )
        
        # 제외 기간에는 VWC를 NaN으로 설정
        if 'exclude_from_calculation' in daily_data.columns:
            exclude_mask = daily_data['exclude_from_calculation']
            excluded_count = exclude_mask.sum()
            
            if excluded_count > 0:
                # 제외된 날짜들 확인
                excluded_dates = daily_data[exclude_mask].index
                self.logger.info(f"Setting VWC to NaN for {excluded_count} excluded days:")
                for date in excluded_dates[:5]:  # 처음 5개만 표시
                    self.logger.info(f"  - {date.date()}")
                if excluded_count > 5:
                    self.logger.info(f"  ... and {excluded_count - 5} more dates")
                
                # 제외 기간 VWC를 NaN으로 설정
                daily_data.loc[exclude_mask, 'VWC'] = np.nan
        
        # 물리적으로 불가능한 값도 NaN 처리
        physical_mask = (daily_data['VWC'] < 0) | (daily_data['VWC'] > 1)
        invalid_count = physical_mask.sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Set {invalid_count} physically invalid VWC values to NaN")
            daily_data.loc[physical_mask, 'VWC'] = np.nan
            
        # 유효한 VWC 통계
        valid_vwc = daily_data['VWC'].dropna()
        if len(valid_vwc) > 0:
            self.logger.info(f"Valid VWC: {len(valid_vwc)} days, range: {valid_vwc.min():.3f} - {valid_vwc.max():.3f} (mean: {valid_vwc.mean():.3f})")
        else:
            self.logger.warning("No valid VWC values calculated!")
        
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
        """결과 저장 - 수정된 버전 (연속적인 날짜 범위 보장)"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = self.station_config['station_info']['id']
        output_file = output_path / f"{station_id}_soil_moisture.xlsx"
        
        # 연속적인 날짜 범위 생성
        if len(daily_data) > 0:
            min_date = daily_data.index.min()
            max_date = daily_data.index.max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # 기존 데이터를 연속적인 인덱스로 reindex
            daily_data_complete = daily_data.reindex(full_date_range)
            
            # 결과 통계
            total_days = len(full_date_range)
            original_days = len(daily_data)
            excluded_days = daily_data['exclude_from_calculation'].sum() if 'exclude_from_calculation' in daily_data.columns else 0
            valid_vwc_days = daily_data_complete['VWC'].notna().sum()
            
            self.logger.info(f"Save results summary:")
            self.logger.info(f"  Total date range: {total_days} days ({min_date.date()} to {max_date.date()})")
            self.logger.info(f"  Original data: {original_days} days")
            self.logger.info(f"  Excluded from calculation: {excluded_days} days")
            self.logger.info(f"  Valid VWC values: {valid_vwc_days} days")
            
        else:
            daily_data_complete = daily_data
            self.logger.warning("No data to save!")
        
        # Excel 파일로 저장
        try:
            self.file_handler.save_dataframe(daily_data_complete, str(output_file), index=True)
            self.logger.info(f"Results saved: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
        
        # 제외 기간 정보를 별도 파일로 저장
        if 'exclude_from_calculation' in daily_data_complete.columns:
            exclude_summary = {
                'calculation_period': {
                    'start': str(daily_data_complete.index.min().date()),
                    'end': str(daily_data_complete.index.max().date()),
                    'total_days': len(daily_data_complete)
                },
                'exclusion_summary': {
                    'excluded_days': int(daily_data_complete['exclude_from_calculation'].sum()),
                    'valid_vwc_days': int(daily_data_complete['VWC'].notna().sum()),
                    'exclusion_percentage': float(daily_data_complete['exclude_from_calculation'].sum() / len(daily_data_complete) * 100)
                },
                'exclude_periods_config': self.calculation_config.get('exclude_periods', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_file = output_path / f"{station_id}_calculation_exclusions.json"
            try:
                import json
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(exclude_summary, f, indent=2, ensure_ascii=False, default=str)
                
                self.logger.info(f"Exclusion summary saved: {summary_file}")
            except Exception as e:
                self.logger.warning(f"Could not save exclusion summary: {e}")
        
    def _create_data_summary(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 생성 - 수정된 버전"""
        
        if len(daily_data) == 0:
            return {
                'total_days': 0,
                'excluded_days': 0,
                'valid_vwc_days': 0,
                'date_range': {'start': None, 'end': None},
                'vwc_statistics': {},
                'exclusion_info': {}
            }
        
        # 제외 기간 정보
        excluded_days = 0
        exclusion_info = {}
        
        if 'exclude_from_calculation' in daily_data.columns:
            excluded_days = int(daily_data['exclude_from_calculation'].sum())
            exclude_config = self.calculation_config.get('exclude_periods', {})
            
            exclusion_info = {
                'excluded_days': excluded_days,
                'excluded_months': exclude_config.get('months', []),
                'custom_date_ranges': exclude_config.get('custom_dates', []),
                'exclusion_percentage': float(excluded_days / len(daily_data) * 100) if len(daily_data) > 0 else 0
            }
        
        # 유효한 VWC 데이터 (NaN 제외)
        valid_vwc = daily_data['VWC'].dropna()
        
        summary = {
            'total_days': len(daily_data),
            'excluded_days': excluded_days,
            'valid_vwc_days': len(valid_vwc),
            'date_range': {
                'start': str(daily_data.index.min().date()),
                'end': str(daily_data.index.max().date())
            },
            'vwc_statistics': {},
            'storage': None,
            'exclusion_info': exclusion_info
        }
        
        # VWC 통계 (제외 기간 및 NaN 제외)
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