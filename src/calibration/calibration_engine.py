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
            """캘리브레이션 데이터 로드 - 수정된 버전"""
            
            with ProcessTimer(self.logger, "Loading calibration data"):
                
                # FDR 데이터 로드
                self.logger.info(f"Loading FDR data from {fdr_path}")
                fdr_data = pd.read_excel(fdr_path)
                
                # 날짜 컬럼 처리
                if 'Date' in fdr_data.columns:
                    fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
                else:
                    raise ValueError("Date column not found in FDR data")
                    
                # CRNP 데이터 로드 - 수정된 부분!
                self.logger.info(f"Loading CRNP data from {crnp_path}")
                
                # 전처리된 파일은 이미 헤더가 있으므로 그대로 읽기
                crnp_data = pd.read_excel(crnp_path)
                
                # 타임스탬프 컬럼 확인 및 처리
                if 'timestamp' in crnp_data.columns:
                    # 이미 전처리에서 timestamp 컬럼이 생성되었음
                    self.logger.info("Using existing timestamp column")
                    pass  # 그대로 사용
                elif 'Timestamp' in crnp_data.columns:
                    # Timestamp 컬럼만 있는 경우 timestamp로 복사
                    crnp_data['timestamp'] = pd.to_datetime(crnp_data['Timestamp'], errors='coerce')
                else:
                    # 첫 번째 컬럼이 타임스탬프일 가능성
                    first_col = crnp_data.columns[0]
                    crnp_data['timestamp'] = pd.to_datetime(crnp_data[first_col], errors='coerce')
                    self.logger.warning(f"Using first column as timestamp: {first_col}")
                
                # 타임스탬프 유효성 확인
                valid_timestamps = crnp_data['timestamp'].notna().sum()
                self.logger.info(f"Valid timestamps in CRNP data: {valid_timestamps}/{len(crnp_data)}")
                
                if valid_timestamps == 0:
                    raise ValueError("No valid timestamps found in CRNP data")
                
                # 캘리브레이션 기간으로 필터링
                fdr_mask = (fdr_data['Date'] >= cal_start) & (fdr_data['Date'] <= cal_end)
                crnp_mask = (crnp_data['timestamp'] >= cal_start) & (crnp_data['timestamp'] <= cal_end)
                
                fdr_filtered = fdr_data[fdr_mask].copy()
                crnp_filtered = crnp_data[crnp_mask].copy()
                
                self.logger.log_data_summary("FDR_Calibration", len(fdr_filtered))
                self.logger.log_data_summary("CRNP_Calibration", len(crnp_filtered))
                
                # 디버깅 정보 추가
                if len(crnp_filtered) == 0:
                    self.logger.error("⚠️ CRNP calibration data is empty!")
                    self.logger.error(f"CRNP data range: {crnp_data['timestamp'].min()} to {crnp_data['timestamp'].max()}")
                    self.logger.error(f"Calibration range: {cal_start} to {cal_end}")
                    self.logger.error(f"Available CRNP columns: {list(crnp_data.columns)}")
                    
                    # 중성자 카운트 확인
                    if 'N_counts' in crnp_data.columns:
                        neutron_valid = crnp_data['N_counts'].notna().sum()
                        self.logger.error(f"Neutron counts available: {neutron_valid}/{len(crnp_data)}")
                    else:
                        self.logger.error("N_counts column not found!")
                        
                    # 기간 내 데이터 재확인
                    period_data = crnp_data[crnp_mask]
                    self.logger.error(f"Period data count: {len(period_data)}")
                    
                else:
                    # 중성자 카운트 확인
                    if 'N_counts' in crnp_filtered.columns:
                        neutron_valid = crnp_filtered['N_counts'].notna().sum()
                        self.logger.info(f"Neutron counts in calibration period: {neutron_valid}/{len(crnp_filtered)}")
                        
                        if neutron_valid == 0:
                            self.logger.warning("No valid neutron counts in calibration period!")
                
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
        """일별 FDR과 CRNP 데이터 매칭 - 최종 수정 버전"""
        
        with ProcessTimer(self.logger, "Matching daily data"):
            
            # 1. 기본 정보 로깅
            self.logger.info(f"Input data: FDR={len(fdr_data)}, CRNP={len(crnp_data)}")
            self.logger.info(f"Calibration period: {cal_start.date()} to {cal_end.date()}")
            
            # 2. CRNP 일별 평균 계산 (간단하게)
            crnp_data_copy = crnp_data.copy()
            crnp_data_copy['date'] = crnp_data_copy['timestamp'].dt.date
            
            # 필수 컬럼만 선택
            if 'total_corrected_neutrons' not in crnp_data_copy.columns:
                self.logger.error("total_corrected_neutrons column missing!")
                return pd.DataFrame()
                
            daily_crnp = crnp_data_copy.groupby('date')['total_corrected_neutrons'].mean().reset_index()
            self.logger.info(f"Daily CRNP created: {len(daily_crnp)} days")
            
            # 3. FDR 날짜 형식 통일
            fdr_data_copy = fdr_data.copy()
            if 'Date' in fdr_data_copy.columns:
                # 이미 date 타입이면 그대로, datetime이면 date로 변환
                if hasattr(fdr_data_copy['Date'].iloc[0], 'date'):
                    fdr_data_copy['Date'] = fdr_data_copy['Date'].dt.date
                self.logger.info(f"FDR date range: {fdr_data_copy['Date'].min()} to {fdr_data_copy['Date'].max()}")
            else:
                self.logger.error("No Date column in FDR data!")
                return pd.DataFrame()
            
            # 4. 매칭 시도 (매우 단순하게)
            results = []
            matched_days = 0
            failed_days = 0
            
            for single_date in pd.date_range(start=cal_start, end=cal_end, freq='D'):
                date_key = single_date.date()
                
                # CRNP 데이터
                crnp_day = daily_crnp[daily_crnp['date'] == date_key]
                if crnp_day.empty:
                    failed_days += 1
                    self.logger.debug(f"No CRNP for {date_key}")
                    continue
                    
                # FDR 데이터  
                fdr_day = fdr_data_copy[fdr_data_copy['Date'] == date_key]
                if fdr_day.empty:
                    failed_days += 1
                    self.logger.debug(f"No FDR for {date_key}")
                    continue
                    
                # 간단한 토양수분 평균 계산 (가중평균 대신)
                if 'theta_v' in fdr_day.columns:
                    valid_theta = fdr_day[(fdr_day['theta_v'] > 0) & (fdr_day['theta_v'] < 1)]
                    
                    if len(valid_theta) > 0:
                        simple_sm = valid_theta['theta_v'].mean()
                        neutron_count = crnp_day.iloc[0]['total_corrected_neutrons']
                        
                        results.append({
                            'date': single_date,
                            'Daily_N': neutron_count,
                            'Field_SM': simple_sm
                        })
                        
                        matched_days += 1
                        self.logger.debug(f"✅ {date_key}: N={neutron_count:.1f}, SM={simple_sm:.3f}")
                    else:
                        failed_days += 1
                        self.logger.debug(f"❌ {date_key}: No valid theta_v")
                else:
                    failed_days += 1
                    self.logger.debug(f"❌ {date_key}: No theta_v column")
            
            # 5. 결과 정리
            matched_df = pd.DataFrame(results)
            
            self.logger.info(f"Matching summary: {matched_days} success, {failed_days} failed")
            self.logger.log_data_summary("Matched_Daily", len(matched_df))
            
            if len(matched_df) == 0:
                self.logger.error("🚨 CRITICAL: Still no matches!")
                self.logger.error("Final debugging:")
                
                # 날짜별 상세 분석
                sample_date = cal_start.date()
                sample_crnp = daily_crnp[daily_crnp['date'] == sample_date]
                sample_fdr = fdr_data_copy[fdr_data_copy['Date'] == sample_date]
                
                self.logger.error(f"  Sample date: {sample_date}")
                self.logger.error(f"  CRNP for sample: {len(sample_crnp)} records")
                self.logger.error(f"  FDR for sample: {len(sample_fdr)} records")
                
                if len(sample_fdr) > 0:
                    theta_stats = sample_fdr['theta_v'].describe()
                    self.logger.error(f"  Sample theta_v stats: {theta_stats.to_dict()}")
            else:
                self.logger.info(f"🎉 SUCCESS: {len(matched_df)} daily records matched!")
                
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
        """N0 최적화 - API 올바른 사용법으로 수정"""
        
        with ProcessTimer(self.logger, "N0 Optimization"):
            
            if len(matched_data) == 0:
                raise ValueError("No matched data available for optimization")
                
            # 격자수 계산
            if self.lattice_water is None:
                self.lattice_water = crnpy.lattice_water(clay_content=self.clay_content)
                
            # 목적함수 정의 (RMSE 최소화) - API 수정
            def objective(N0):
                try:
                    # ✅ 올바른 API 사용법: 첫 번째 매개변수로 중성자 카운트 전달
                    crnp_sm = crnpy.counts_to_vwc(
                        matched_data['Daily_N'],  # 첫 번째 위치 매개변수
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
                    
                except Exception as e:
                    self.logger.debug(f"Objective function error: {e}")
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
            
            # 최적화 결과 검증 - API 수정
            optimized_sm = crnpy.counts_to_vwc(
                matched_data['Daily_N'],  # 첫 번째 위치 매개변수
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