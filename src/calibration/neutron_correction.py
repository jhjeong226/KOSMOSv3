# src/calibration/neutron_correction.py

import pandas as pd
import numpy as np
import crnpy
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler


class NeutronCorrector:
    """중성자 카운트 보정을 담당하는 클래스"""
    
    def __init__(self, station_config: Dict, processing_config: Dict, 
                 logger: Optional[CRNPLogger] = None):
        self.station_config = station_config
        self.processing_config = processing_config
        self.logger = logger or CRNPLogger("NeutronCorrector")
        
        # 보정 옵션 설정
        self.corrections_enabled = processing_config.get('corrections', {
            'incoming_flux': True,
            'pressure': True,
            'humidity': True,
            'biomass': False
        })
        
        # 중성자 모니터 설정
        self.neutron_monitor = station_config.get('calibration', {}).get('neutron_monitor', 'ATHN')
        self.utc_offset = station_config.get('calibration', {}).get('utc_offset', 9)
        
        # 보정 매개변수 기본값
        self.correction_params = {
            'pressure_L': 130,  # 기압 보정 상수
        }
        
    def apply_corrections(self, df_crnp: pd.DataFrame, 
                         calibration_params: Optional[Dict] = None) -> pd.DataFrame:
        """선택적 중성자 보정 적용"""
        
        with ProcessTimer(self.logger, "Neutron Corrections"):
            
            df_corrected = df_crnp.copy()
            
            # 기본 전처리
            df_corrected = self._preprocess_neutron_data(df_corrected)
            
            # 보정 계수 초기화
            df_corrected['fi'] = 1.0  # incoming flux
            df_corrected['fp'] = 1.0  # pressure  
            df_corrected['fw'] = 1.0  # humidity
            df_corrected['fb'] = 1.0  # biomass
            
            # 들어오는 중성자 플럭스 보정 (fi)
            if self.corrections_enabled.get('incoming_flux', True):
                df_corrected = self._apply_incoming_flux_correction(df_corrected, calibration_params)
                
            # 기압 보정 (fp)
            if self.corrections_enabled.get('pressure', True):
                df_corrected = self._apply_pressure_correction(df_corrected, calibration_params)
                
            # 습도 보정 (fw)
            if self.corrections_enabled.get('humidity', True):
                df_corrected = self._apply_humidity_correction(df_corrected, calibration_params)
                
            # 바이오매스 보정 (fb) - 선택사항
            if self.corrections_enabled.get('biomass', False):
                df_corrected = self._apply_biomass_correction(df_corrected, calibration_params)
                
            # 최종 보정된 중성자 수 계산
            df_corrected['total_corrected_neutrons'] = (
                df_corrected['total_raw_counts'] * df_corrected['fw'] * df_corrected['fb'] / 
                (df_corrected['fp'] * df_corrected['fi'])
            )
            
            # 보정 통계 로깅
            self._log_correction_statistics(df_corrected)
            
            return df_corrected
            
    def _preprocess_neutron_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """중성자 데이터 전처리"""
        
        # 필수 컬럼 확인
        required_columns = ['timestamp', 'N_counts', 'Ta', 'RH', 'Pa']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # 원시 중성자 카운트 설정
        if 'total_raw_counts' not in df.columns:
            df['total_raw_counts'] = df['N_counts']
            
        # 기상 데이터 수치 변환 및 결측치 보간
        numeric_columns = ['Ta', 'RH', 'Pa', 'total_raw_counts']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 결측치 보간 (선형 보간, 최대 24시간)
        df[['Pa', 'RH', 'Ta']] = df[['Pa', 'RH', 'Ta']].interpolate(
            method='linear', limit=24, limit_direction='both'
        )
        
        return df
        
    def _apply_incoming_flux_correction(self, df: pd.DataFrame, 
                                      calibration_params: Optional[Dict] = None) -> pd.DataFrame:
        """들어오는 중성자 플럭스 보정 (fi)"""
        
        try:
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            self.logger.info(f"Downloading neutron flux data from {self.neutron_monitor}")
            
            # NMDB에서 중성자 플럭스 데이터 다운로드
            nmdb = crnpy.get_incoming_neutron_flux(
                start_date, end_date, 
                station=self.neutron_monitor, 
                utc_offset=self.utc_offset
            )
            
            # 관측소 데이터 시간에 맞춰 보간
            df['incoming_flux'] = crnpy.interpolate_incoming_flux(
                nmdb['timestamp'], nmdb['counts'], df['timestamp']
            )
            
            # 참조 값 설정 (캘리브레이션 매개변수 또는 평균값)
            if calibration_params and 'Iref' in calibration_params:
                Iref = calibration_params['Iref']
            else:
                Iref = df['incoming_flux'].mean()
                
            # fi 보정 계수 계산
            df['fi'] = crnpy.correction_incoming_flux(
                incoming_neutrons=df['incoming_flux'], 
                incoming_Ref=Iref
            )
            
            self.logger.info(f"Incoming flux correction applied (Iref={Iref:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to apply incoming flux correction: {e}")
            df['fi'] = 1.0
            df['incoming_flux'] = np.nan
            
        return df
        
    def _apply_pressure_correction(self, df: pd.DataFrame, 
                                 calibration_params: Optional[Dict] = None) -> pd.DataFrame:
        """기압 보정 (fp)"""
        
        try:
            # 참조 기압 설정
            if calibration_params and 'Pref' in calibration_params:
                Pref = calibration_params['Pref']
            else:
                Pref = df['Pa'].mean()
                
            # fp 보정 계수 계산
            df['fp'] = crnpy.correction_pressure(
                pressure=df['Pa'], 
                Pref=Pref, 
                L=self.correction_params['pressure_L']
            )
            
            self.logger.info(f"Pressure correction applied (Pref={Pref:.2f} hPa)")
            
        except Exception as e:
            self.logger.error(f"Failed to apply pressure correction: {e}")
            df['fp'] = 1.0
            
        return df
        
    def _apply_humidity_correction(self, df: pd.DataFrame, 
                                 calibration_params: Optional[Dict] = None) -> pd.DataFrame:
        """습도 보정 (fw)"""
        
        try:
            # 절대습도 계산
            df['abs_humidity'] = crnpy.abs_humidity(df['RH'], df['Ta'])
            
            # 참조 절대습도 설정
            if calibration_params and 'Aref' in calibration_params:
                Aref = calibration_params['Aref']
            else:
                Aref = df['abs_humidity'].mean()
                
            # fw 보정 계수 계산
            df['fw'] = crnpy.correction_humidity(
                abs_humidity=df['abs_humidity'], 
                Aref=Aref
            )
            
            self.logger.info(f"Humidity correction applied (Aref={Aref:.4f} g/cm³)")
            
        except Exception as e:
            self.logger.error(f"Failed to apply humidity correction: {e}")
            df['fw'] = 1.0
            df['abs_humidity'] = np.nan
            
        return df
        
    def _apply_biomass_correction(self, df: pd.DataFrame, 
                                calibration_params: Optional[Dict] = None) -> pd.DataFrame:
        """바이오매스 보정 (fb) - 선택사항"""
        
        try:
            # 바이오매스 데이터가 있는 경우에만 적용
            if 'biomass' in df.columns:
                # 바이오매스 보정 구현 (추후 확장)
                # 현재는 기본값 1.0 사용
                df['fb'] = 1.0
                self.logger.info("Biomass correction applied (default=1.0)")
            else:
                df['fb'] = 1.0
                self.logger.debug("No biomass data, fb=1.0")
                
        except Exception as e:
            self.logger.error(f"Failed to apply biomass correction: {e}")
            df['fb'] = 1.0
            
        return df
        
    def _log_correction_statistics(self, df: pd.DataFrame) -> None:
        """보정 통계 로깅"""
        
        correction_factors = ['fi', 'fp', 'fw', 'fb']
        
        self.logger.info("Neutron correction statistics:")
        for factor in correction_factors:
            if factor in df.columns:
                values = df[factor].dropna()
                if len(values) > 0:
                    self.logger.info(f"  {factor}: mean={values.mean():.4f}, "
                                   f"std={values.std():.4f}, "
                                   f"range=[{values.min():.4f}, {values.max():.4f}]")
                    
        # 보정 전후 중성자 카운트 비교
        if 'total_raw_counts' in df.columns and 'total_corrected_neutrons' in df.columns:
            raw_mean = df['total_raw_counts'].mean()
            corrected_mean = df['total_corrected_neutrons'].mean()
            change_percent = ((corrected_mean - raw_mean) / raw_mean) * 100
            
            self.logger.info(f"Neutron counts: raw={raw_mean:.1f}, "
                           f"corrected={corrected_mean:.1f}, "
                           f"change={change_percent:+.2f}%")
                           
    def calculate_reference_values(self, df: pd.DataFrame, 
                                 period_start: Optional[datetime] = None,
                                 period_end: Optional[datetime] = None) -> Dict[str, float]:
        """캘리브레이션 기간 동안의 참조값 계산"""
        
        # 기간 필터링
        if period_start and period_end:
            mask = (df['timestamp'] >= period_start) & (df['timestamp'] <= period_end)
            df_period = df[mask]
        else:
            df_period = df
            
        reference_values = {}
        
        # 기본 참조값들 계산
        if 'Pa' in df_period.columns:
            reference_values['Pref'] = df_period['Pa'].mean()
            
        if 'abs_humidity' in df_period.columns:
            reference_values['Aref'] = df_period['abs_humidity'].mean()
        elif 'RH' in df_period.columns and 'Ta' in df_period.columns:
            abs_humidity = crnpy.abs_humidity(df_period['RH'], df_period['Ta'])
            reference_values['Aref'] = abs_humidity.mean()
            
        if 'incoming_flux' in df_period.columns:
            reference_values['Iref'] = df_period['incoming_flux'].mean()
            
        self.logger.info(f"Calculated reference values: {reference_values}")
        return reference_values
        
    def remove_outliers(self, df: pd.DataFrame, column: str = 'total_corrected_neutrons',
                       method: str = 'mad', threshold: float = 3.0) -> pd.DataFrame:
        """이상값 제거"""
        
        if column not in df.columns:
            return df
            
        data = df[column].dropna()
        
        if len(data) == 0:
            return df
            
        if method == 'mad':
            # Median Absolute Deviation
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            
            if mad > 0:
                mad_scores = np.abs(data - median) / mad
                outlier_mask = mad_scores > threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    self.logger.warning(f"Removing {outlier_count} outliers from {column}")
                    
                    # 이상값을 NaN으로 설정
                    df_cleaned = df.copy()
                    outlier_indices = data[outlier_mask].index
                    df_cleaned.loc[outlier_indices, column] = np.nan
                    
                    return df_cleaned
                    
        return df


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    
    # 테스트용 설정
    test_station_config = {
        'calibration': {
            'neutron_monitor': 'ATHN',
            'utc_offset': 9
        }
    }
    
    test_processing_config = {
        'corrections': {
            'incoming_flux': True,
            'pressure': True,
            'humidity': True,
            'biomass': False
        }
    }
    
    # NeutronCorrector 테스트
    logger = setup_logger("NeutronCorrector_Test")
    corrector = NeutronCorrector(test_station_config, test_processing_config, logger)
    
    # 테스트 데이터 생성
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-08-17', periods=24, freq='H'),
        'N_counts': np.random.normal(1000, 50, 24),
        'Ta': np.random.normal(25, 5, 24),
        'RH': np.random.normal(60, 10, 24),
        'Pa': np.random.normal(1013, 10, 24)
    })
    
    try:
        corrected_data = corrector.apply_corrections(test_data)
        print("✅ 중성자 보정 테스트 완료!")
        print(f"보정 계수 컬럼: {[col for col in corrected_data.columns if col in ['fi', 'fp', 'fw', 'fb']]}")
        
    except Exception as e:
        print(f"❌ 중성자 보정 테스트 실패: {e}")
        import traceback
        traceback.print_exc()