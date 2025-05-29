# src/calibration/calibration_engine.py

import pandas as pd
import numpy as np
import crnpy
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler
from .neutron_correction import NeutronCorrector


class CalibrationEngine:
    """CRNP 캘리브레이션을 담당하는 엔진 클래스"""
    
    def __init__(self, station_config: Dict, processing_config: Dict, 
                 logger: Optional[CRNPLogger] = None):
        self.station_config = station_config
        self.processing_config = processing_config
        self.logger = logger or CRNPLogger("CalibrationEngine")
        
        # 종속 모듈 초기화
        self.file_handler = FileHandler(self.logger)
        self.neutron_corrector = NeutronCorrector(station_config, processing_config, self.logger)
        
        # 캘리브레이션 설정
        self.calibration_config = self.processing_config.get('calibration', {})
        self.depths = self.calibration_config.get('reference_depths', [10, 30, 60])
        self.weighting_method = self.calibration_config.get('weighting_method', 'Schron_2017')
        self.optimization_method = self.calibration_config.get('optimization_method', 'Nelder-Mead')
        self.initial_N0 = self.calibration_config.get('initial_N0', 1000)
        
        # 토양 특성
        soil_props = self.station_config.get('soil_properties', {})
        self.bulk_density = soil_props.get('bulk_density', 1.44)
        self.clay_content = soil_props.get('clay_content', 0.35)
        self.lattice_water = None  # 자동 계산
        
    def run_calibration(self, calibration_start: str, calibration_end: str,
                       fdr_data_path: str, crnp_data_path: str,
                       output_dir: str) -> Dict[str, Any]:
        """전체 캘리브레이션 프로세스 실행"""
        
        with ProcessTimer(self.logger, "CRNP Calibration",
                         period=f"{calibration_start} to {calibration_end}"):
            
            try:
                # 1. 캘리브레이션 기간 설정
                cal_start = pd.to_datetime(calibration_start)
                cal_end = pd.to_datetime(calibration_end)
                
                # 2. 데이터 로드
                fdr_data, crnp_data = self._load_calibration_data(
                    fdr_data_path, crnp_data_path, cal_start, cal_end
                )
                
                # 3. 중성자 보정 적용
                corrected_crnp = self._apply_neutron_corrections(crnp_data)
                
                # 4. 일별 데이터 매칭
                matched_data = self._match_daily_data(fdr_data, corrected_crnp, cal_start, cal_end)
                
                # 5. N0 최적화
                optimization_result = self._optimize_N0(matched_data)
                
                # 6. 캘리브레이션 결과 생성
                calibration_result = self._create_calibration_result(
                    optimization_result, corrected_crnp, cal_start, cal_end
                )
                
                # 7. 결과 저장
                self._save_calibration_results(calibration_result, output_dir)
                
                self.logger.info(f"Calibration completed successfully. N0 = {calibration_result['N0_rdt']:.2f}")
                return calibration_result
                
            except Exception as e:
                self.logger.log_error_with_context(e, "Calibration process")
                raise
                
    def _load_calibration_data(self, fdr_path: str, crnp_path: str,
                              cal_start: datetime, cal_end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """캘리브레이션 데이터 로드"""
        
        with ProcessTimer(self.logger, "Loading calibration data"):
            
            # FDR 데이터 로드
            self.logger.info(f"Loading FDR data from {fdr_path}")
            fdr_data = pd.read_excel(fdr_path)
            
            # 날짜 컬럼 처리
            if 'Date' in fdr_data.columns:
                fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
            else:
                raise ValueError("Date column not found in FDR data")
                
            # CRNP 데이터 로드
            self.logger.info(f"Loading CRNP data from {crnp_path}")
            crnp_columns = ['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
            crnp_data = pd.read_excel(crnp_path, names=crnp_columns)
            crnp_data['timestamp'] = pd.to_datetime(crnp_data['Timestamp'], errors='coerce')
            
            # 캘리브레이션 기간으로 필터링
            fdr_mask = (fdr_data['Date'] >= cal_start) & (fdr_data['Date'] <= cal_end)
            crnp_mask = (crnp_data['timestamp'] >= cal_start) & (crnp_data['timestamp'] <= cal_end)
            
            fdr_filtered = fdr_data[fdr_mask].copy()
            crnp_filtered = crnp_data[crnp_mask].copy()
            
            self.logger.log_data_summary("FDR_Calibration", len(fdr_filtered))
            self.logger.log_data_summary("CRNP_Calibration", len(crnp_filtered))
            
            return fdr_filtered, crnp_filtered
            
    def _apply_neutron_corrections(self, crnp_data: pd.DataFrame) -> pd.DataFrame:
        """중성자 보정 적용"""
        
        with ProcessTimer(self.logger, "Applying neutron corrections"):
            
            # 원시 중성자 카운트 설정
            crnp_data['total_raw_counts'] = crnp_data['N_counts']
            
            # 중성자 보정 적용
            corrected_data = self.neutron_corrector.apply_corrections(crnp_data)
            
            # 이상값 제거
            cleaned_data = self.neutron_corrector.remove_outliers(
                corrected_data, 'total_corrected_neutrons', threshold=3.0
            )
            
            return cleaned_data
            
    def _match_daily_data(self, fdr_data: pd.DataFrame, crnp_data: pd.DataFrame,
                         cal_start: datetime, cal_end: datetime) -> pd.DataFrame:
        """일별 FDR과 CRNP 데이터 매칭"""
        
        with ProcessTimer(self.logger, "Matching daily data"):
            
            # 지리정보 로드
            geo_info = self._load_geo_info()
            
            # CRNP 일별 평균 계산
            crnp_data['date'] = crnp_data['timestamp'].dt.date
            daily_crnp = crnp_data.groupby('date').agg({
                'total_corrected_neutrons': 'mean',
                'abs_humidity': 'mean',
                'Pa': 'mean'
            }).reset_index()
            
            results = []
            
            # 일별로 데이터 매칭 및 가중평균 계산
            for single_date in pd.date_range(start=cal_start, end=cal_end, freq='D'):
                date_key = single_date.date()
                
                # 해당 날짜의 CRNP 데이터
                daily_crnp_data = daily_crnp[daily_crnp['date'] == date_key]
                
                if daily_crnp_data.empty:
                    continue
                    
                # 해당 날짜의 FDR 데이터
                fdr_mask = fdr_data['Date'].dt.date == date_key
                daily_fdr = fdr_data[fdr_mask]
                
                if daily_fdr.empty:
                    continue
                    
                # 지점 토양수분 가중평균 계산
                field_sm = self._calculate_weighted_soil_moisture(
                    daily_fdr, daily_crnp_data.iloc[0], geo_info
                )
                
                if field_sm is not None:
                    results.append({
                        'date': single_date,
                        'Daily_N': daily_crnp_data.iloc[0]['total_corrected_neutrons'],
                        'Field_SM': field_sm
                    })
                    
            matched_df = pd.DataFrame(results)
            self.logger.log_data_summary("Matched_Daily", len(matched_df))
            
            return matched_df
            
    def _calculate_weighted_soil_moisture(self, fdr_data: pd.DataFrame, 
                                        crnp_data: pd.Series, geo_info: Dict) -> Optional[float]:
        """가중평균 지점 토양수분 계산"""
        
        try:
            # 깊이별 토양수분 데이터 필터링
            depth_mask = fdr_data['FDR_depth'].isin(self.depths)
            fdr_filtered = fdr_data[depth_mask]
            
            if fdr_filtered.empty:
                return None
                
            # ID별로 프로파일 생성
            fdr_filtered['ID'] = (fdr_filtered['latitude'].astype(str) + '_' + 
                                fdr_filtered['longitude'].astype(str))
            
            # 가중평균 계산
            if self.weighting_method == "Schron_2017":
                field_sm, _ = crnpy.nrad_weight(
                    abs_humidity=crnp_data['abs_humidity'],
                    theta_v=fdr_filtered['theta_v'],
                    distances=fdr_filtered['distance_from_station'],
                    depths=fdr_filtered['FDR_depth'],
                    profiles=fdr_filtered['ID'],
                    rhob=self.bulk_density,
                    p=crnp_data['Pa'],
                    method="Schron_2017"
                )
            else:
                # 기본 가중평균 (Kohli_2015)
                field_sm, _ = crnpy.nrad_weight(
                    abs_humidity=crnp_data['abs_humidity'],
                    theta_v=fdr_filtered['theta_v'],
                    distances=fdr_filtered['distance_from_station'],
                    depths=fdr_filtered['FDR_depth'],
                    rhob=self.bulk_density,
                    method="Kohli_2015"
                )
                
            return field_sm
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate weighted soil moisture: {e}")
            return None
            
    def _optimize_N0(self, matched_data: pd.DataFrame) -> Dict[str, Any]:
        """N0 최적화"""
        
        with ProcessTimer(self.logger, "N0 Optimization"):
            
            if len(matched_data) == 0:
                raise ValueError("No matched data available for optimization")
                
            # 격자수 계산
            if self.lattice_water is None:
                self.lattice_water = crnpy.lattice_water(clay_content=self.clay_content)
                
            # 목적함수 정의 (RMSE 최소화)
            def objective(N0):
                try:
                    crnp_sm = crnpy.counts_to_vwc(
                        N=matched_data['Daily_N'], 
                        N0=N0[0], 
                        bulk_density=self.bulk_density, 
                        Wlat=self.lattice_water, 
                        Wsoc=0.01
                    )
                    
                    # NaN 값 제거
                    valid_mask = ~(np.isnan(crnp_sm) | np.isnan(matched_data['Field_SM']))
                    if valid_mask.sum() == 0:
                        return 1e6  # 큰 값 반환
                        
                    crnp_clean = crnp_sm[valid_mask]
                    field_clean = matched_data['Field_SM'].values[valid_mask]
                    
                    rmse = np.sqrt(np.mean((crnp_clean - field_clean) ** 2))
                    return rmse
                    
                except Exception:
                    return 1e6
                    
            # 최적화 실행
            self.logger.info(f"Starting N0 optimization (method: {self.optimization_method})")
            
            result = minimize(
                objective, 
                x0=[self.initial_N0], 
                method=self.optimization_method,
                bounds=[(500, 3000)]  # N0 범위 제한
            )
            
            N0_optimized = result.x[0]
            final_rmse = result.fun
            
            # 최적화 결과 검증
            optimized_sm = crnpy.counts_to_vwc(
                N=matched_data['Daily_N'], 
                N0=N0_optimized, 
                bulk_density=self.bulk_density, 
                Wlat=self.lattice_water, 
                Wsoc=0.01
            )
            
            # 성능 지표 계산
            valid_mask = ~(np.isnan(optimized_sm) | np.isnan(matched_data['Field_SM']))
            crnp_clean = optimized_sm[valid_mask]
            field_clean = matched_data['Field_SM'].values[valid_mask]
            
            metrics = self._calculate_performance_metrics(field_clean, crnp_clean)
            
            optimization_result = {
                'N0_optimized': N0_optimized,
                'optimization_success': result.success,
                'final_rmse': final_rmse,
                'metrics': metrics,
                'matched_data_count': len(matched_data),
                'valid_data_count': valid_mask.sum()
            }
            
            self.logger.log_calibration_result(N0_optimized, metrics)
            
            return optimization_result
            
    def _calculate_performance_metrics(self, observed: np.ndarray, 
                                     predicted: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        
        if len(observed) == 0 or len(predicted) == 0:
            return {}
            
        try:
            # R²
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # RMSE
            rmse = np.sqrt(np.mean((observed - predicted) ** 2))
            
            # MAE
            mae = np.mean(np.abs(observed - predicted))
            
            # NSE (Nash-Sutcliffe Efficiency)
            nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Bias
            bias = np.mean(predicted - observed)
            
            return {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'NSE': nse,
                'Bias': bias,
                'n_samples': len(observed)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
            
    def _create_calibration_result(self, optimization_result: Dict, 
                                 corrected_crnp: pd.DataFrame,
                                 cal_start: datetime, cal_end: datetime) -> Dict[str, Any]:
        """캘리브레이션 결과 생성"""
        
        # 참조값들 계산
        reference_values = self.neutron_corrector.calculate_reference_values(
            corrected_crnp, cal_start, cal_end
        )
        
        # 관측소 정보
        station_info = self.station_config.get('coordinates', {})
        
        calibration_result = {
            'station_id': self.station_config['station_info']['id'],
            'calibration_period': {
                'start': cal_start.isoformat(),
                'end': cal_end.isoformat()
            },
            'coordinates': {
                'lat': station_info.get('latitude'),
                'lon': station_info.get('longitude')
            },
            'N0_rdt': optimization_result['N0_optimized'],
            'Pref': reference_values.get('Pref'),
            'Aref': reference_values.get('Aref'),
            'Iref': reference_values.get('Iref'),
            'clay_content': self.clay_content,
            'soil_bulk_density': self.bulk_density,
            'lattice_water': self.lattice_water,
            'optimization': {
                'method': self.optimization_method,
                'success': optimization_result['optimization_success'],
                'final_rmse': optimization_result['final_rmse'],
                'matched_data_count': optimization_result['matched_data_count']
            },
            'performance_metrics': optimization_result['metrics'],
            'settings': {
                'weighting_method': self.weighting_method,
                'reference_depths': self.depths,
                'corrections_enabled': self.neutron_corrector.corrections_enabled,
                'neutron_monitor': self.neutron_corrector.neutron_monitor
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return calibration_result
        
    def _save_calibration_results(self, calibration_result: Dict, output_dir: str) -> None:
        """캘리브레이션 결과 저장"""
        
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = calibration_result['station_id']
        
        # 1. JSON 형식으로 저장 (전체 결과)
        json_file = output_path / f"{station_id}_calibration_result.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_result, f, indent=2, ensure_ascii=False)
            
        # 2. Excel 형식으로 저장 (매개변수만)
        params_data = {
            'Parameter': ['lat', 'lon', 'N0_rdt', 'Pref', 'Aref', 'Iref', 'clay_content', 'soil_bulk_density'],
            'Value': [
                calibration_result['coordinates']['lat'],
                calibration_result['coordinates']['lon'],
                calibration_result['N0_rdt'],
                calibration_result['Pref'],
                calibration_result['Aref'],
                calibration_result['Iref'],
                calibration_result['clay_content'],
                calibration_result['soil_bulk_density']
            ]
        }
        
        params_df = pd.DataFrame(params_data)
        excel_file = output_path / f"{station_id}_Parameters.xlsx"
        params_df.to_excel(excel_file, index=False)
        
        self.logger.log_file_operation("save", str(json_file), "success")
        self.logger.log_file_operation("save", str(excel_file), "success")
        
    def _load_geo_info(self) -> Dict:
        """지리정보 로드"""
        from ..core.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        geo_info = config_manager.load_geo_info_from_yaml(self.station_config)
        
        return geo_info


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    
    # 테스트용 설정
    test_station_config = {
        'station_info': {'id': 'HC'},
        'coordinates': {'latitude': 37.7049111, 'longitude': 128.0316412},
        'soil_properties': {'bulk_density': 1.44, 'clay_content': 0.35},
        'calibration': {'neutron_monitor': 'ATHN', 'utc_offset': 9}
    }
    
    test_processing_config = {
        'calibration': {
            'weighting_method': 'Schron_2017',
            'optimization_method': 'Nelder-Mead',
            'initial_N0': 1000,
            'reference_depths': [10, 30, 60]
        },
        'corrections': {
            'incoming_flux': True,
            'pressure': True,
            'humidity': True,
            'biomass': False
        }
    }
    
    # CalibrationEngine 테스트
    logger = setup_logger("CalibrationEngine_Test")
    engine = CalibrationEngine(test_station_config, test_processing_config, logger)
    
    print("✅ CalibrationEngine 구현 완료!")
    print("주요 기능:")
    print("  - 중성자 보정 적용")
    print("  - 지점 토양수분 가중평균 계산")
    print("  - N0 최적화")
    print("  - 성능 지표 계산")
    print("  - 결과 저장 (JSON + Excel)")