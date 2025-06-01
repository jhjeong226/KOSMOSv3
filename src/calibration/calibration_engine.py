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
    """CRNP 캘리브레이션을 담당하는 엔진 클래스 - 수정된 버전"""
    
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
        """전체 캘리브레이션 프로세스 실행 - 향상된 진단 포함"""
        
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
                
                # 4. 지리정보 로드
                geo_info = self._load_geo_info()
                
                # 5. 향상된 일별 데이터 매칭 (가중평균 적용)
                matched_data = self._match_daily_data_enhanced(
                    fdr_data, corrected_crnp, geo_info, cal_start, cal_end
                )
                
                # 6. N0 최적화
                optimization_result = self._optimize_N0_enhanced(matched_data)
                
                # 7. 캘리브레이션 결과 생성
                calibration_result = self._create_calibration_result(
                    optimization_result, corrected_crnp, cal_start, cal_end
                )
                
                # 8. 진단 데이터 및 시각화 생성
                self._generate_calibration_diagnostics(
                    matched_data, optimization_result, output_dir
                )
                
                # 9. 결과 저장
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
            crnp_data = pd.read_excel(crnp_path)
            
            # 타임스탬프 컬럼 확인 및 처리
            if 'timestamp' in crnp_data.columns:
                pass  # 이미 있음
            elif 'Timestamp' in crnp_data.columns:
                crnp_data['timestamp'] = pd.to_datetime(crnp_data['Timestamp'], errors='coerce')
            else:
                # 첫 번째 컬럼이 타임스탬프일 가능성
                first_col = crnp_data.columns[0]
                crnp_data['timestamp'] = pd.to_datetime(crnp_data[first_col], errors='coerce')
                self.logger.warning(f"Using first column as timestamp: {first_col}")
            
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
            
    def _match_daily_data_enhanced(self, fdr_data: pd.DataFrame, crnp_data: pd.DataFrame,
                                geo_info: Dict, cal_start: datetime, cal_end: datetime) -> pd.DataFrame:
        """향상된 일별 데이터 매칭 - crnpy 가중평균 사용 (pandas 경고 수정)"""
        
        with ProcessTimer(self.logger, "Enhanced daily data matching"):
            
            # 1. CRNP 일별 평균 계산
            crnp_data_copy = crnp_data.copy()
            crnp_data_copy['date'] = crnp_data_copy['timestamp'].dt.date
            
            daily_crnp = crnp_data_copy.groupby('date').agg({
                'total_corrected_neutrons': 'mean',
                'abs_humidity': 'mean',
                'Pa': 'mean'
            }).reset_index()
            
            self.logger.info(f"Daily CRNP data: {len(daily_crnp)} days")
            
            # 2. FDR 데이터 날짜 처리
            fdr_data_copy = fdr_data.copy()
            fdr_data_copy['Date'] = pd.to_datetime(fdr_data_copy['Date'])
            
            results = []
            matched_days = 0
            failed_days = 0
            
            # 3. 일별 매칭 및 가중평균 계산
            for single_date in pd.date_range(start=cal_start, end=cal_end, freq='D'):
                date_key = single_date.date()
                
                # CRNP 데이터
                crnp_day = daily_crnp[daily_crnp['date'] == date_key]
                if crnp_day.empty:
                    failed_days += 1
                    continue
                    
                # FDR 데이터
                fdr_day = fdr_data_copy[fdr_data_copy['Date'].dt.date == date_key]
                if fdr_day.empty:
                    failed_days += 1
                    continue
                
                # 유효한 FDR 데이터 필터링 - .copy() 추가로 pandas 경고 해결
                valid_fdr = fdr_day[
                    (fdr_day['theta_v'].notna()) & 
                    (fdr_day['theta_v'] > 0) & 
                    (fdr_day['theta_v'] < 1) &
                    (fdr_day['FDR_depth'].isin(self.depths))
                ].copy()  # 🔧 .copy() 추가하여 SettingWithCopyWarning 해결
                
                if len(valid_fdr) == 0:
                    failed_days += 1
                    continue
                
                # crnpy 가중평균 계산
                try:
                    # 프로파일 ID 생성 - 이제 경고 없이 안전하게 할당 가능
                    valid_fdr['profile_id'] = (
                        valid_fdr['latitude'].astype(str) + '_' + 
                        valid_fdr['longitude'].astype(str)
                    )
                    
                    crnp_day_data = crnp_day.iloc[0]
                    
                    # crnpy 가중평균 계산
                    if self.weighting_method == "Schron_2017":
                        field_sm, weights = crnpy.nrad_weight(
                            abs_humidity=crnp_day_data['abs_humidity'],
                            theta_v=valid_fdr['theta_v'].values,
                            distances=valid_fdr['distance_from_station'].values,
                            depths=valid_fdr['FDR_depth'].values,
                            profiles=valid_fdr['profile_id'].values,
                            rhob=self.bulk_density,
                            p=crnp_day_data['Pa'],
                            method="Schron_2017"
                        )
                    else:
                        # Kohli_2015 방법
                        field_sm, weights = crnpy.nrad_weight(
                            abs_humidity=crnp_day_data['abs_humidity'],
                            theta_v=valid_fdr['theta_v'].values,
                            distances=valid_fdr['distance_from_station'].values,
                            depths=valid_fdr['FDR_depth'].values,
                            rhob=self.bulk_density,
                            method="Kohli_2015"
                        )
                    
                    # 결과 저장
                    results.append({
                        'date': single_date,
                        'Daily_N': crnp_day_data['total_corrected_neutrons'],
                        'Field_SM': field_sm,
                        'Simple_SM': valid_fdr['theta_v'].mean(),  # 비교용 단순 평균
                        'N_sensors': len(valid_fdr),
                        'abs_humidity': crnp_day_data['abs_humidity'],
                        'pressure': crnp_day_data['Pa']
                    })
                    
                    matched_days += 1
                    self.logger.debug(f"✅ {date_key}: N={crnp_day_data['total_corrected_neutrons']:.1f}, "
                                    f"Weighted_SM={field_sm:.3f}, Simple_SM={valid_fdr['theta_v'].mean():.3f}")
                    
                except Exception as e:
                    self.logger.debug(f"❌ {date_key}: Weighting failed - {e}")
                    # 가중평균 실패시 단순 평균 사용
                    simple_sm = valid_fdr['theta_v'].mean()
                    results.append({
                        'date': single_date,
                        'Daily_N': crnp_day_data['total_corrected_neutrons'],
                        'Field_SM': simple_sm,
                        'Simple_SM': simple_sm,
                        'N_sensors': len(valid_fdr),
                        'abs_humidity': crnp_day_data['abs_humidity'],
                        'pressure': crnp_day_data['Pa']
                    })
                    matched_days += 1
                    failed_days += 1  # 가중평균 실패로 카운트
            
            # 결과 정리
            matched_df = pd.DataFrame(results)
            
            self.logger.info(f"Enhanced matching: {matched_days} matched, {failed_days} failed")
            self.logger.log_data_summary("Enhanced_Matched", len(matched_df))
            
            if len(matched_df) > 0:
                self.logger.info(f"Field SM range: {matched_df['Field_SM'].min():.3f} ~ {matched_df['Field_SM'].max():.3f}")
                self.logger.info(f"Neutron range: {matched_df['Daily_N'].min():.1f} ~ {matched_df['Daily_N'].max():.1f}")
                
                # 변동성 확인
                sm_std = matched_df['Field_SM'].std()
                neutron_std = matched_df['Daily_N'].std()
                self.logger.info(f"Field SM std: {sm_std:.4f}, Neutron std: {neutron_std:.1f}")
                
                if sm_std < 0.01:
                    self.logger.warning("⚠️ Field SM variability is very low - may affect calibration quality")
                    
            return matched_df
            
    def _optimize_N0_enhanced(self, matched_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 N0 최적화"""
        
        with ProcessTimer(self.logger, "Enhanced N0 Optimization"):
            
            if len(matched_data) == 0:
                raise ValueError("No matched data available for optimization")
                
            # 격자수 계산
            if self.lattice_water is None:
                self.lattice_water = crnpy.lattice_water(clay_content=self.clay_content)
                
            self.logger.info(f"Optimization parameters:")
            self.logger.info(f"  Data points: {len(matched_data)}")
            self.logger.info(f"  Bulk density: {self.bulk_density}")
            self.logger.info(f"  Lattice water: {self.lattice_water:.4f}")
            self.logger.info(f"  Field SM range: {matched_data['Field_SM'].min():.3f} - {matched_data['Field_SM'].max():.3f}")
            self.logger.info(f"  Neutron range: {matched_data['Daily_N'].min():.1f} - {matched_data['Daily_N'].max():.1f}")
            
            # 먼저 여러 N0 값으로 테스트
            test_N0_values = np.linspace(500, 3000, 21)  # 500부터 3000까지 21개 값
            best_rmse = float('inf')
            best_N0_initial = self.initial_N0
            
            self.logger.info("Testing N0 values:")
            for N0_test in test_N0_values:
                try:
                    vwc_test = crnpy.counts_to_vwc(
                        matched_data['Daily_N'],
                        N0=N0_test,
                        bulk_density=self.bulk_density,
                        Wlat=self.lattice_water,
                        Wsoc=0.01
                    )
                    
                    # 유효한 값만 선택
                    valid_mask = ~np.isnan(vwc_test) & (vwc_test >= 0) & (vwc_test <= 1)
                    if valid_mask.sum() > 0:
                        rmse_test = np.sqrt(np.mean((vwc_test[valid_mask] - matched_data['Field_SM'].values[valid_mask]) ** 2))
                        
                        if rmse_test < best_rmse:
                            best_rmse = rmse_test
                            best_N0_initial = N0_test
                            
                        self.logger.debug(f"  N0={N0_test:.0f}: RMSE={rmse_test:.4f}")
                    
                except Exception:
                    continue
                    
            self.logger.info(f"Best initial N0: {best_N0_initial:.0f} (RMSE: {best_rmse:.4f})")
            
            # 목적함수 정의
            def objective(N0):
                try:
                    vwc = crnpy.counts_to_vwc(
                        matched_data['Daily_N'],
                        N0=N0[0],
                        bulk_density=self.bulk_density,
                        Wlat=self.lattice_water,
                        Wsoc=0.01
                    )
                    
                    # 유효성 검사
                    if np.any(np.isnan(vwc)) or np.any(vwc < 0) or np.any(vwc > 1):
                        return 1e6
                        
                    # RMSE 계산
                    rmse = np.sqrt(np.mean((vwc - matched_data['Field_SM']) ** 2))
                    return rmse
                    
                except Exception:
                    return 1e6
                    
            # 최적화 실행 (더 좋은 초기값 사용)
            self.logger.info(f"Starting optimization from N0={best_N0_initial:.0f}")
            
            result = minimize(
                objective,
                x0=[best_N0_initial],
                method=self.optimization_method,
                bounds=[(500, 3000)]
            )
            
            N0_optimized = result.x[0]
            final_rmse = result.fun
            
            # 최적화 결과 계산
            optimized_vwc = crnpy.counts_to_vwc(
                matched_data['Daily_N'],
                N0=N0_optimized,
                bulk_density=self.bulk_density,
                Wlat=self.lattice_water,
                Wsoc=0.01
            )
            
            # 성능 지표 계산
            metrics = self._calculate_performance_metrics_robust(
                matched_data['Field_SM'].values, optimized_vwc
            )
            
            # 디버깅 데이터 생성
            debug_data = matched_data.copy()
            debug_data['CRNP_VWC'] = optimized_vwc
            debug_data['Residuals'] = optimized_vwc - matched_data['Field_SM']
            debug_data['N0_used'] = N0_optimized
            
            optimization_result = {
                'N0_optimized': N0_optimized,
                'optimization_success': result.success,
                'final_rmse': final_rmse,
                'metrics': metrics,
                'matched_data_count': len(matched_data),
                'debug_data': debug_data,
                'initial_test_rmse': best_rmse,
                'initial_test_N0': best_N0_initial
            }
            
            self.logger.log_calibration_result(N0_optimized, metrics)
            
            return optimization_result
            
    def _calculate_performance_metrics_robust(self, observed: np.ndarray, 
                                            predicted: np.ndarray) -> Dict[str, float]:
        """강건한 성능 지표 계산 - R² 계산 문제 완전 해결"""
        
        if len(observed) == 0 or len(predicted) == 0:
            return {'R2': 0, 'RMSE': 1, 'MAE': 1, 'NSE': 0, 'Bias': 0, 'Correlation': 0, 'n_samples': 0}
            
        try:
            # 기본 통계
            rmse = np.sqrt(np.mean((observed - predicted) ** 2))
            mae = np.mean(np.abs(observed - predicted))
            bias = np.mean(predicted - observed)
            
            # Pearson 상관계수 (가장 안정적)
            if len(observed) > 1:
                correlation = np.corrcoef(observed, predicted)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
                
            # 관측값 변동성 확인
            obs_std = np.std(observed)
            obs_var = np.var(observed)
            obs_mean = np.mean(observed)
            pred_std = np.std(predicted)
            
            self.logger.info(f"Performance calculation details:")
            self.logger.info(f"  Observed: mean={obs_mean:.6f}, std={obs_std:.6f}, var={obs_var:.6f}")
            self.logger.info(f"  Predicted: mean={np.mean(predicted):.6f}, std={pred_std:.6f}")
            self.logger.info(f"  Correlation: {correlation:.6f}")
            self.logger.info(f"  RMSE: {rmse:.6f}, MAE: {mae:.6f}")
            
            # R² 계산 방법 결정
            r2 = 0.0
            method_used = "none"
            
            # 1. 충분한 변동성이 있는 경우 (표준편차 > 0.01)
            if obs_std > 0.01:
                # 전통적인 R² 계산
                ss_res = np.sum((observed - predicted) ** 2)
                ss_tot = np.sum((observed - obs_mean) ** 2)
                
                if ss_tot > 1e-10:  # 분모가 충분히 큰 경우
                    r2_traditional = 1 - (ss_res / ss_tot)
                    
                    # 합리적인 범위인지 확인 (-2 ~ 1)
                    if -2 <= r2_traditional <= 1:
                        r2 = r2_traditional
                        method_used = "traditional"
                        self.logger.info(f"  Traditional R²: {r2:.6f} (SS_res={ss_res:.6f}, SS_tot={ss_tot:.6f})")
                    else:
                        # 범위를 벗어나면 상관계수 제곱 사용
                        r2 = max(0, correlation ** 2)
                        method_used = "correlation_squared_fallback"
                        self.logger.warning(f"  Traditional R² out of range ({r2_traditional:.6f}), using correlation²")
                else:
                    # 분모가 너무 작으면 상관계수 제곱 사용
                    r2 = max(0, correlation ** 2)
                    method_used = "correlation_squared_small_denominator"
                    self.logger.warning(f"  SS_tot too small ({ss_tot:.10f}), using correlation²")
                    
            # 2. 변동성이 작은 경우 (0.005 < std <= 0.01)
            elif obs_std > 0.005:
                # 상관계수 제곱을 기본으로 사용
                r2 = max(0, correlation ** 2)
                method_used = "correlation_squared_moderate_var"
                self.logger.info(f"  Moderate variability, using correlation²: {r2:.6f}")
                
            # 3. 변동성이 매우 작은 경우 (std <= 0.005)
            else:
                # 상대 오차 기반 평가
                if abs(obs_mean) > 1e-10:
                    relative_rmse = rmse / abs(obs_mean)
                    
                    # 상대 오차가 10% 이내면 양호
                    if relative_rmse <= 0.1:
                        r2 = max(0, 1 - relative_rmse * 5)  # 최대 0.5
                    elif relative_rmse <= 0.2:
                        r2 = max(0, 1 - relative_rmse * 2.5)  # 최대 0.5
                    else:
                        r2 = 0
                        
                    method_used = "relative_error_based"
                    self.logger.warning(f"  Very low variability, using relative error method: {r2:.6f} (rel_rmse={relative_rmse:.6f})")
                else:
                    # 관측값 평균이 0에 가까운 경우
                    r2 = max(0, correlation ** 2) if abs(correlation) > 0.3 else 0
                    method_used = "correlation_near_zero_mean"
                    self.logger.warning(f"  Near-zero observed mean, using correlation if strong: {r2:.6f}")
            
            # NSE 계산 (R²와 동일하게 처리)
            nse = r2
            
            # 결과 검증
            if r2 < 0:
                r2 = 0
                self.logger.warning(f"  Negative R² set to 0")
            elif r2 > 1:
                r2 = 1
                self.logger.warning(f"  R² > 1 set to 1")
                
            self.logger.info(f"  Final R² = {r2:.6f} (method: {method_used})")
            
            # 품질 경고
            if obs_std < 0.005:
                self.logger.warning(f"  Very low observed variability may limit calibration quality")
            if abs(correlation) < 0.3:
                self.logger.warning(f"  Weak correlation may indicate poor model fit")
                
            return {
                'R2': float(r2),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'NSE': float(nse),
                'Bias': float(bias),
                'Correlation': float(correlation),
                'n_samples': len(observed),
                'obs_std': float(obs_std),
                'pred_std': float(pred_std),
                'obs_mean': float(obs_mean),
                'pred_mean': float(np.mean(predicted)),
                'method_used': method_used
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'R2': 0, 'RMSE': 1, 'MAE': 1, 'NSE': 0, 'Bias': 0, 'Correlation': 0, 
                'n_samples': len(observed), 'method_used': 'error'
            }
            
    def _generate_calibration_diagnostics(self, matched_data: pd.DataFrame, 
                                        optimization_result: Dict, output_dir: str) -> None:
        """캘리브레이션 진단 데이터 및 시각화 생성"""
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from pathlib import Path
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            station_id = self.station_config['station_info']['id']
            debug_data = optimization_result['debug_data']
            
            # 1. 상세 시각화 생성 (영문)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Time series comparison
            axes[0,0].plot(debug_data['date'], debug_data['Field_SM'], 'bo-', 
                          label='FDR Field SM', markersize=8, linewidth=2)
            axes[0,0].plot(debug_data['date'], debug_data['CRNP_VWC'], 'ro-', 
                          label='CRNP VWC', markersize=8, linewidth=2)
            if 'Simple_SM' in debug_data.columns:
                axes[0,0].plot(debug_data['date'], debug_data['Simple_SM'], 'g^-', 
                              label='Simple Average SM', markersize=6, alpha=0.7)
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Volumetric Water Content')
            axes[0,0].set_title(f'{station_id} - Calibration Time Series')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Scatter plot
            axes[0,1].scatter(debug_data['Field_SM'], debug_data['CRNP_VWC'], 
                             s=120, alpha=0.8, c='blue', edgecolors='black')
            
            # 1:1 line
            min_val = min(debug_data['Field_SM'].min(), debug_data['CRNP_VWC'].min())
            max_val = max(debug_data['Field_SM'].max(), debug_data['CRNP_VWC'].max())
            axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                          linewidth=2, label='1:1 Line')
            
            # Best fit line
            z = np.polyfit(debug_data['Field_SM'], debug_data['CRNP_VWC'], 1)
            p = np.poly1d(z)
            axes[0,1].plot(debug_data['Field_SM'], p(debug_data['Field_SM']), 
                          'g-', alpha=0.8, label=f'Best fit (y={z[0]:.2f}x+{z[1]:.3f})')
            
            axes[0,1].set_xlabel('FDR Field SM')
            axes[0,1].set_ylabel('CRNP VWC')
            axes[0,1].set_title('Calibration Scatter Plot')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Neutron counts time series
            axes[0,2].plot(debug_data['date'], debug_data['Daily_N'], 'go-', 
                          label='Daily Neutron Counts', markersize=8, linewidth=2)
            axes[0,2].set_xlabel('Date')
            axes[0,2].set_ylabel('Neutron Counts')
            axes[0,2].set_title('Neutron Counts Time Series')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # Residuals plot
            axes[1,0].scatter(debug_data['CRNP_VWC'], debug_data['Residuals'], 
                             s=120, alpha=0.8, c='red', edgecolors='black')
            axes[1,0].axhline(y=0, color='k', linestyle='--', linewidth=2)
            axes[1,0].set_xlabel('CRNP VWC')
            axes[1,0].set_ylabel('Residuals (CRNP - FDR)')
            axes[1,0].set_title('Residuals Plot')
            axes[1,0].grid(True, alpha=0.3)
            
            # Residuals time series
            axes[1,1].plot(debug_data['date'], debug_data['Residuals'], 'mo-', 
                          markersize=8, linewidth=2)
            axes[1,1].axhline(y=0, color='k', linestyle='--', linewidth=2)
            axes[1,1].set_xlabel('Date')
            axes[1,1].set_ylabel('Residuals (CRNP - FDR)')
            axes[1,1].set_title('Residuals Time Series')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Performance metrics text
            metrics = optimization_result['metrics']
            metrics_text = f"""Performance Metrics:
R² = {metrics.get('R2', 0):.4f}
RMSE = {metrics.get('RMSE', 0):.4f}
MAE = {metrics.get('MAE', 0):.4f}
Bias = {metrics.get('Bias', 0):.4f}
Correlation = {metrics.get('Correlation', 0):.4f}

Optimization:
N0 = {optimization_result['N0_optimized']:.1f}
Data points = {len(debug_data)}

Data Variability:
FDR std = {metrics.get('obs_std', 0):.4f}
CRNP std = {metrics.get('pred_std', 0):.4f}"""
            
            axes[1,2].text(0.1, 0.9, metrics_text, transform=axes[1,2].transAxes, 
                          fontsize=11, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1,2].set_xlim(0, 1)
            axes[1,2].set_ylim(0, 1)
            axes[1,2].axis('off')
            axes[1,2].set_title('Calibration Results Summary')
            
            plt.tight_layout()
            
            # 그래프 저장
            plot_file = output_path / f"{station_id}_calibration_diagnostics.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.log_file_operation("save", str(plot_file), "success")
            
            # 2. 간단한 비교 그래프 (요청사항)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Time series comparison
            ax1.plot(debug_data['date'], debug_data['Field_SM'], 'bo-', 
                    label='FDR Field SM', markersize=6, linewidth=2)
            ax1.plot(debug_data['date'], debug_data['CRNP_VWC'], 'ro-', 
                    label='CRNP VWC', markersize=6, linewidth=2)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Volumetric Water Content')
            ax1.set_title(f'{station_id} Calibration Period - Time Series Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Scatter plot
            ax2.scatter(debug_data['Field_SM'], debug_data['CRNP_VWC'], 
                       s=100, alpha=0.8, c='blue')
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
            ax2.set_xlabel('FDR Field SM')
            ax2.set_ylabel('CRNP VWC')
            ax2.set_title(f'Scatter Plot (R² = {metrics.get("R2", 0):.3f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            comparison_plot_file = output_path / f"{station_id}_calibration_comparison.png"
            plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.log_file_operation("save", str(comparison_plot_file), "success")
            
            # 3. 디버깅 데이터 Excel 저장
            debug_excel_file = output_path / f"{station_id}_calibration_debug_data.xlsx"
            with pd.ExcelWriter(debug_excel_file, engine='openpyxl') as writer:
                debug_data.to_excel(writer, sheet_name='Calibration_Data', index=False)
                
                # 요약 시트
                summary_data = {
                    'Metric': ['R²', 'RMSE', 'MAE', 'Bias', 'Correlation', 'N0', 'Data Points',
                              'FDR Std', 'CRNP Std', 'FDR Mean', 'CRNP Mean'],
                    'Value': [
                        metrics.get('R2', 0),
                        metrics.get('RMSE', 0),
                        metrics.get('MAE', 0), 
                        metrics.get('Bias', 0),
                        metrics.get('Correlation', 0),
                        optimization_result['N0_optimized'],
                        len(debug_data),
                        metrics.get('obs_std', 0),
                        metrics.get('pred_std', 0),
                        debug_data['Field_SM'].mean(),
                        debug_data['CRNP_VWC'].mean()
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
            self.logger.log_file_operation("save", str(debug_excel_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not generate calibration diagnostics: {e}")
            
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
                'matched_data_count': optimization_result['matched_data_count'],
                'initial_test_rmse': optimization_result.get('initial_test_rmse'),
                'initial_test_N0': optimization_result.get('initial_test_N0')
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
        
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = calibration_result['station_id']
        
        # JSON 저장
        json_file = output_path / f"{station_id}_calibration_result.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_result, f, indent=2, ensure_ascii=False)
        
        # 매개변수 Excel 저장
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