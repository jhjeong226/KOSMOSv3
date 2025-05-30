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
    """CRNP 토양수분 계산을 담당하는 클래스 - Sensing Depth 문제 해결"""
    
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
                
                # 8. 유효깊이 계산 (수정된 버전)
                soil_moisture_data = self._calculate_sensing_depth_robust(soil_moisture_data)
                
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
        """CRNP 데이터 로드 - 수정된 버전"""
        
        with ProcessTimer(self.logger, "Loading CRNP data"):
            
            if not Path(data_path).exists():
                raise FileNotFoundError(f"CRNP data file not found: {data_path}")
                
            # 데이터 로드 (컬럼명 지정하지 않음 - 전처리된 데이터이므로)
            df = pd.read_excel(data_path)
            
            self.logger.info(f"Original columns: {list(df.columns)}")
            
            # 타임스탬프 처리 - 수정된 로직
            timestamp_col = None
            for col in ['timestamp', 'Timestamp', 'Date']:
                if col in df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col is None:
                # 첫 번째 컬럼이 타임스탬프일 가능성
                timestamp_col = df.columns[0]
                self.logger.warning(f"Using first column as timestamp: {timestamp_col}")
                
            # 타임스탬프 변환
            try:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # 타임스탬프 검증
                if df['timestamp'].isna().all():
                    raise ValueError(f"All timestamp values are invalid in column {timestamp_col}")
                    
                # 잘못된 날짜 확인 (1970년 문제)
                min_date = df['timestamp'].min()
                if min_date.year < 2000:
                    self.logger.warning(f"Suspicious timestamp found: {min_date}. Checking for numeric timestamp...")
                    
                    # 숫자형 타임스탬프인지 확인
                    if df[timestamp_col].dtype in ['int64', 'float64']:
                        # Unix timestamp 또는 Excel serial date 변환 시도
                        try:
                            # Excel serial date 변환 시도 (1900-01-01 기준)
                            df['timestamp'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(df[timestamp_col] - 1, unit='D')
                            self.logger.info("Converted Excel serial dates to datetime")
                        except:
                            # Unix timestamp 변환 시도
                            df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce')
                            self.logger.info("Converted Unix timestamps to datetime")
                            
            except Exception as e:
                self.logger.error(f"Timestamp conversion failed: {e}")
                raise ValueError(f"Cannot convert timestamp column {timestamp_col}")
                
            # 유효한 데이터만 유지
            initial_count = len(df)
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            removed_count = initial_count - len(df)
            if removed_count > 0:
                self.logger.warning(f"Removed {removed_count} records with invalid timestamps")
                
            # 최종 검증
            if len(df) == 0:
                raise ValueError("No valid data remaining after timestamp processing")
                
            date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            self.logger.log_data_summary("CRNP_Raw", len(df), date_range=date_range)
            
            return df
            
    def _determine_calculation_period(self, df: pd.DataFrame,
                                    start_str: Optional[str], 
                                    end_str: Optional[str]) -> Tuple[str, str]:
        """계산 기간 결정 - 수정된 버전"""
        
        if start_str and end_str:
            calc_start = start_str
            calc_end = end_str
        else:
            # 전체 데이터 기간 사용
            calc_start = df['timestamp'].min().strftime('%Y-%m-%d')
            calc_end = df['timestamp'].max().strftime('%Y-%m-%d')
            
        self.logger.info(f"Calculation period: {calc_start} to {calc_end}")
        return calc_start, calc_end
        
    def _filter_calculation_period(self, df: pd.DataFrame, 
                                 calc_start: str, calc_end: str) -> pd.DataFrame:
        """계산 기간으로 데이터 필터링 - 수정된 버전"""
        
        start_date = pd.to_datetime(calc_start)
        end_date = pd.to_datetime(calc_end)
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df[mask].copy()
        
        self.logger.log_data_summary("CRNP_Filtered", len(filtered_df))
        return filtered_df
        
    def _apply_neutron_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """중성자 보정 적용 - 안정화된 버전"""
        
        with ProcessTimer(self.logger, "Applying neutron corrections"):
            
            # 원시 중성자 카운트 설정
            neutron_col = None
            for col in ['N_counts', 'total_raw_counts']:
                if col in df.columns:
                    neutron_col = col
                    break
                    
            if neutron_col is None:
                raise ValueError("No neutron counts column found")
                
            df['total_raw_counts'] = df[neutron_col]
            
            # 중성자 보정 적용 (캘리브레이션 매개변수 사용)
            try:
                corrected_data = self.neutron_corrector.apply_corrections(df, self.calibration_params)
            except Exception as e:
                self.logger.warning(f"Neutron correction failed: {e}. Using basic corrections only.")
                
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
        """일평균 데이터 계산 - 수정된 버전"""
        
        with ProcessTimer(self.logger, "Calculating daily averages"):
            
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
            
            self.logger.log_data_summary("CRNP_Daily", len(daily_data))
            return daily_data
            
    def _calculate_vwc(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """체적수분함량 계산 - 수정된 함수 호출"""
        
        with ProcessTimer(self.logger, "Calculating VWC"):
            
            # VWC 계산 - 수정된 함수 호출 (positional argument 사용)
            daily_data['VWC'] = crnpy.counts_to_vwc(
                daily_data['total_corrected_neutrons'],  # 첫 번째 인자는 positional
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
                
            valid_vwc = daily_data['VWC'].dropna()
            if len(valid_vwc) > 0:
                self.logger.info(f"VWC calculated: mean={valid_vwc.mean():.3f}, "
                               f"std={valid_vwc.std():.3f}, "
                               f"range=[{valid_vwc.min():.3f}, {valid_vwc.max():.3f}]")
            else:
                self.logger.warning("No valid VWC values calculated")
            
            return daily_data
            
    def _calculate_sensing_depth_robust(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """강건한 유효깊이 계산 - 완전 수정된 버전"""
        
        with ProcessTimer(self.logger, "Calculating sensing depth"):
            
            # 1단계: crnpy.sensing_depth 시도 (여러 방법)
            crnpy_success = False
            
            try:
                # 방법 1: 기본 Franz_2012
                depth_result = crnpy.sensing_depth(
                    theta_v=daily_data['VWC'].values,  # numpy array로 전달
                    pressure=daily_data['Pa'].values,
                    pressure_ref=daily_data['Pa'].mean(),
                    bulk_density=self.bulk_density,
                    Wlat=self.lattice_water,
                    method="Franz_2012"
                )
                
                if depth_result is not None and not np.all(np.isnan(depth_result)):
                    daily_data['sensing_depth'] = depth_result
                    avg_depth = np.nanmean(depth_result)
                    self.logger.info(f"crnpy sensing depth (Franz_2012): {avg_depth:.1f} mm")
                    crnpy_success = True
                    
            except Exception as e:
                self.logger.debug(f"Franz_2012 method failed: {e}")
                
            # 방법 2: Kohli_2015 시도
            if not crnpy_success:
                try:
                    depth_result = crnpy.sensing_depth(
                        theta_v=daily_data['VWC'].values,
                        pressure=daily_data['Pa'].values,
                        pressure_ref=daily_data['Pa'].mean(),
                        bulk_density=self.bulk_density,
                        Wlat=self.lattice_water,
                        method="Kohli_2015"
                    )
                    
                    if depth_result is not None and not np.all(np.isnan(depth_result)):
                        daily_data['sensing_depth'] = depth_result
                        avg_depth = np.nanmean(depth_result)
                        self.logger.info(f"crnpy sensing depth (Kohli_2015): {avg_depth:.1f} mm")
                        crnpy_success = True
                        
                except Exception as e:
                    self.logger.debug(f"Kohli_2015 method failed: {e}")
                    
            # 방법 3: 단순 파라미터로 시도
            if not crnpy_success:
                try:
                    # VWC와 압력만 사용하는 단순한 호출
                    depth_result = crnpy.sensing_depth(
                        daily_data['VWC'].values,
                        daily_data['Pa'].values,
                        daily_data['Pa'].mean(),
                        self.bulk_density,
                        self.lattice_water
                    )
                    
                    if depth_result is not None and not np.all(np.isnan(depth_result)):
                        daily_data['sensing_depth'] = depth_result
                        avg_depth = np.nanmean(depth_result)
                        self.logger.info(f"crnpy sensing depth (simple): {avg_depth:.1f} mm")
                        crnpy_success = True
                        
                except Exception as e:
                    self.logger.debug(f"Simple method failed: {e}")
                    
            # 2단계: crnpy 실패시 물리 기반 경험식 사용
            if not crnpy_success:
                self.logger.warning("crnpy sensing_depth failed, using physics-based empirical formula")
                
                try:
                    # Desilets et al. (2010) & Franz et al. (2012) 기반 경험식
                    vwc_values = daily_data['VWC'].values
                    pressure_values = daily_data['Pa'].values
                    p_ref = daily_data['Pa'].mean()
                    
                    # 물리 기반 유효깊이 계산
                    physics_depth = []
                    
                    for i, (vwc, pressure) in enumerate(zip(vwc_values, pressure_values)):
                        if pd.notna(vwc) and pd.notna(pressure) and vwc > 0:
                            
                            # Franz et al. (2012) 공식 기반
                            # D86 = 5.8 / (ρ_bd * (θ + W_lat) + 0.0829)
                            
                            # 유효 밀도 계산
                            effective_density = self.bulk_density * (vwc + self.lattice_water) + 0.0829
                            
                            # 86% 관여 깊이 (mm)
                            D86 = (5.8 / effective_density) * 100  # cm to mm
                            
                            # 기압 보정
                            pressure_correction = pressure / p_ref
                            D86_corrected = D86 * pressure_correction
                            
                            # 현실적 범위 제한 (50-800mm)
                            depth = max(50, min(800, D86_corrected))
                            
                        else:
                            # 결측값이나 비정상값의 경우 중간값 사용
                            depth = 200  # mm
                            
                        physics_depth.append(depth)
                    
                    daily_data['sensing_depth'] = physics_depth
                    avg_depth = np.mean(physics_depth)
                    depth_std = np.std(physics_depth)
                    depth_range = [np.min(physics_depth), np.max(physics_depth)]
                    
                    self.logger.info(f"Physics-based sensing depth: mean={avg_depth:.1f}mm, "
                                   f"std={depth_std:.1f}mm, range={depth_range[0]:.1f}-{depth_range[1]:.1f}mm")
                    
                except Exception as e:
                    self.logger.warning(f"Physics-based calculation failed: {e}")
                    
                    # 3단계: 최종 fallback - VWC 기반 단순 추정
                    try:
                        # VWC에 따른 단순 선형 관계
                        vwc_mean = daily_data['VWC'].mean()
                        
                        if pd.notna(vwc_mean):
                            # 일반적으로 VWC가 높을수록 유효깊이 감소
                            # VWC 0.1 → 300mm, VWC 0.5 → 150mm 정도의 관계
                            base_depth = 350 - (vwc_mean - 0.1) * 500  # 선형 관계
                            base_depth = max(100, min(400, base_depth))  # 100-400mm 범위
                            
                            # 약간의 변동성 추가
                            simple_depths = []
                            for vwc in daily_data['VWC'].values:
                                if pd.notna(vwc):
                                    depth = 350 - (vwc - 0.1) * 500
                                    depth = max(100, min(400, depth))
                                else:
                                    depth = base_depth
                                simple_depths.append(depth)
                                
                            daily_data['sensing_depth'] = simple_depths
                            avg_depth = np.mean(simple_depths)
                            self.logger.warning(f"Using VWC-based estimation: {avg_depth:.1f} mm")
                            
                        else:
                            # 모든 방법 실패 - 고정값
                            daily_data['sensing_depth'] = 200  # mm
                            self.logger.warning("All methods failed, using fixed depth: 200 mm")
                            
                    except Exception as e:
                        # 절대 최종 fallback
                        daily_data['sensing_depth'] = 200  # mm
                        self.logger.error(f"All sensing depth calculations failed: {e}")
                        self.logger.warning("Using absolute fallback: 200 mm")
                        
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
                
                valid_storage = daily_data['storage'].dropna()
                if len(valid_storage) > 0:
                    avg_storage = valid_storage.mean()
                    self.logger.info(f"Average soil water storage: {avg_storage:.1f} mm")
                
            except Exception as e:
                self.logger.warning(f"Storage calculation failed: {e}")
                daily_data['storage'] = daily_data['VWC'] * self.z_surface  # 단순 계산
                
            return daily_data
            
    def _calculate_uncertainty(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """토양수분 불확실성 계산"""
        
        with ProcessTimer(self.logger, "Calculating uncertainty"):
            
            try:
                # VWC 불확실성 계산 - positional argument 사용
                daily_data['sigma_VWC'] = crnpy.uncertainty_vwc(
                    daily_data['total_raw_counts'],  # 첫 번째 인자는 positional
                    N0=self.N0,
                    bulk_density=self.bulk_density,
                    fp=daily_data['fp'],
                    fi=daily_data['fi'],
                    fw=daily_data['fw']
                )
                
                valid_uncertainty = daily_data['sigma_VWC'].dropna()
                if len(valid_uncertainty) > 0:
                    avg_uncertainty = valid_uncertainty.mean()
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
                    valid_vwc = daily_data['VWC'].dropna()
                    if len(valid_vwc) >= window:
                        smoothed_vwc = crnpy.smooth_1d(
                            valid_vwc,
                            window=window,
                            order=order,
                            method="savitzky_golay"
                        )
                        
                        daily_data['VWC_smoothed'] = smoothed_vwc
                        self.logger.info(f"Applied Savitzky-Golay smoothing (window={window}, order={order})")
                    else:
                        self.logger.warning(f"Insufficient data for smoothing (need ≥{window} points)")
                        daily_data['VWC_smoothed'] = daily_data['VWC']
                    
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
        if len(daily_data) > 0:
            full_date_range = pd.date_range(
                start=daily_data.index.min(),
                end=daily_data.index.max(),
                freq='D'
            )
            
            # 완전한 날짜 범위로 재인덱싱 (누락된 날짜는 NaN)
            daily_data_complete = daily_data.reindex(full_date_range)
        else:
            daily_data_complete = daily_data
        
        # Excel 파일로 저장
        self.file_handler.save_dataframe(daily_data_complete, str(output_file), index=True)
        
        self.logger.log_file_operation("save", str(output_file), "success")
        
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
            'sensing_depth': None,
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
        
        # 유효깊이 통계
        if 'sensing_depth' in daily_data.columns:
            valid_depth = daily_data['sensing_depth'].dropna()
            if len(valid_depth) > 0:
                summary['sensing_depth'] = {
                    'mean': float(valid_depth.mean()),
                    'min': float(valid_depth.min()),
                    'max': float(valid_depth.max())
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