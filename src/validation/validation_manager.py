# src/validation/validation_manager.py - 중대한 버그 수정본

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler


class ValidationManager:
    """CRNP 검증을 담당하는 클래스 - 중대한 버그 수정본"""
    
    def __init__(self, station_id: str, config_root: str = "config"):
        self.station_id = station_id
        self.logger = CRNPLogger(f"ValidationManager_{station_id}")
        
        # 기본 경로 설정
        self.output_dir = Path(f"data/output/{station_id}")
        
    def run_validation(self) -> Dict[str, Any]:
        """검증 실행"""
        
        with ProcessTimer(self.logger, f"Validation for {self.station_id}"):
            
            try:
                # 1. 데이터 로드
                fdr_data, crnp_sm_data = self._load_validation_data()
                
                # 2. 데이터 매칭
                matched_data = self._match_validation_data(fdr_data, crnp_sm_data)
                
                if len(matched_data) == 0:
                    raise ValueError("No matching data found for validation")
                
                # 3. 전체 검증 지표 계산 (VWC vs VWC 올바른 비교)
                overall_metrics = self._calculate_validation_metrics_robust(
                    matched_data['field_sm'].values, matched_data['crnp_vwc'].values
                )
                
                # 4. 깊이별 검증
                depth_metrics = self._calculate_depth_metrics(fdr_data, crnp_sm_data)
                
                # 5. 결과 정리 (JSON 직렬화 문제 해결)
                validation_result = {
                    'station_id': self.station_id,
                    'validation_timestamp': datetime.now().isoformat(),
                    'data_summary': {
                        'total_matched_days': len(matched_data),
                        'date_range': {
                            'start': matched_data.index.min().strftime('%Y-%m-%d') if len(matched_data) > 0 else None,
                            'end': matched_data.index.max().strftime('%Y-%m-%d') if len(matched_data) > 0 else None
                        }
                    },
                    'overall_metrics': overall_metrics,
                    'depth_metrics': depth_metrics,
                    # matched_data는 JSON 직렬화 문제로 제외
                }
                
                # 6. 결과 저장
                self._save_validation_results(validation_result, matched_data)
                
                self.logger.log_validation_result(overall_metrics)
                
                return validation_result
                
            except Exception as e:
                self.logger.log_error_with_context(e, f"Validation for {self.station_id}")
                raise
                
    def _load_validation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """검증 데이터 로드"""
        
        with ProcessTimer(self.logger, "Loading validation data"):
            
            # FDR 데이터 로드
            fdr_file = self.output_dir / "preprocessed" / f"{self.station_id}_FDR_input.xlsx"
            self.logger.info(f"Loading FDR data from {fdr_file}")
            
            if not fdr_file.exists():
                raise FileNotFoundError(f"FDR data not found: {fdr_file}")
                
            fdr_data = pd.read_excel(fdr_file)
            fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
            
            # CRNP 토양수분 데이터 로드
            sm_file = self.output_dir / "soil_moisture" / f"{self.station_id}_soil_moisture.xlsx"
            self.logger.info(f"Loading CRNP SM data from {sm_file}")
            
            if not sm_file.exists():
                raise FileNotFoundError(f"CRNP soil moisture data not found: {sm_file}")
                
            crnp_sm_data = pd.read_excel(sm_file, index_col=0)
            crnp_sm_data.index = pd.to_datetime(crnp_sm_data.index)
            
            self.logger.log_data_summary("FDR_Validation", len(fdr_data))
            self.logger.log_data_summary("CRNP_SM_Validation", len(crnp_sm_data))
            
            # 데이터 내용 확인 로그 (디버깅용)
            self.logger.info(f"FDR data columns: {list(fdr_data.columns)}")
            self.logger.info(f"CRNP SM data columns: {list(crnp_sm_data.columns)}")
            
            if 'theta_v' in fdr_data.columns:
                self.logger.info(f"FDR theta_v range: {fdr_data['theta_v'].min():.3f} - {fdr_data['theta_v'].max():.3f}")
            
            if 'VWC' in crnp_sm_data.columns:
                self.logger.info(f"CRNP VWC range: {crnp_sm_data['VWC'].min():.3f} - {crnp_sm_data['VWC'].max():.3f}")
            else:
                self.logger.error(f"VWC column not found in CRNP data! Available columns: {list(crnp_sm_data.columns)}")
            
            return fdr_data, crnp_sm_data
            
    def _match_validation_data(self, fdr_data: pd.DataFrame, 
                              crnp_sm_data: pd.DataFrame) -> pd.DataFrame:
        """검증 데이터 매칭 - 제외 기간 고려하여 VWC vs VWC 비교"""
        
        with ProcessTimer(self.logger, "Matching validation data"):
            
            # FDR 일별 평균 계산
            fdr_daily = fdr_data.groupby(fdr_data['Date'].dt.date)['theta_v'].mean().reset_index()
            fdr_daily.columns = ['date', 'field_sm']
            fdr_daily['date'] = pd.to_datetime(fdr_daily['date'])
            
            # CRNP 데이터 준비 - VWC 컬럼 사용 (중성자 카운트 아님!)
            if 'VWC' not in crnp_sm_data.columns:
                raise ValueError("VWC column not found in CRNP soil moisture data")
                
            crnp_daily = crnp_sm_data[['VWC']].reset_index()
            crnp_daily.columns = ['date', 'crnp_vwc']
            crnp_daily['date'] = pd.to_datetime(crnp_daily['date'])
            
            # 제외 기간 정보도 가져오기 (있는 경우)
            exclude_info = None
            if 'exclude_from_calculation' in crnp_sm_data.columns:
                exclude_daily = crnp_sm_data[['exclude_from_calculation']].reset_index()
                exclude_daily.columns = ['date', 'excluded']
                exclude_daily['date'] = pd.to_datetime(exclude_daily['date'])
                exclude_info = exclude_daily
            
            # 날짜 기준으로 매칭
            matched_data = pd.merge(fdr_daily, crnp_daily, on='date', how='inner')
            
            # 제외 기간 정보 추가
            if exclude_info is not None:
                matched_data = pd.merge(matched_data, exclude_info, on='date', how='left')
                matched_data['excluded'] = matched_data['excluded'].fillna(False)
            else:
                matched_data['excluded'] = False
            
            # 제외 기간 데이터 필터링 (성능 평가에서 제외)
            initial_count = len(matched_data)
            excluded_count = matched_data['excluded'].sum()
            
            # 제외되지 않은 데이터만 유지 (excluded=False 또는 NaN인 데이터)
            matched_data = matched_data[~matched_data['excluded']].copy()
            
            # NaN 제거
            matched_data = matched_data.dropna(subset=['field_sm', 'crnp_vwc'])
            
            # 인덱스 설정
            matched_data = matched_data.set_index('date')
            
            self.logger.info(f"Matched {len(matched_data)} data points (excluded {excluded_count} from calculation periods)")
            
            # 매칭된 데이터 범위 확인 (디버깅)
            if len(matched_data) > 0:
                self.logger.info(f"FDR range: {matched_data['field_sm'].min():.3f} - {matched_data['field_sm'].max():.3f}")
                self.logger.info(f"CRNP VWC range: {matched_data['crnp_vwc'].min():.3f} - {matched_data['crnp_vwc'].max():.3f}")
            
            return matched_data
            
    def _calculate_validation_metrics_robust(self, observed: np.ndarray, 
                                        predicted: np.ndarray) -> Dict[str, float]:
        """강건한 검증 지표 계산 - 완전 수정본"""
        
        if len(observed) == 0 or len(predicted) == 0:
            return {
                'R2': 0.0, 'RMSE': 1.0, 'MAE': 1.0, 'NSE': 0.0, 'Bias': 0.0, 
                'Correlation': 0.0, 'P_value': 1.0, 'IOA': 0.0, 'n_samples': 0.0,
                'method_used': 'no_data'
            }
            
        try:
            # 기본 통계 (항상 계산 가능)
            rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
            mae = float(np.mean(np.abs(observed - predicted)))
            bias = float(np.mean(predicted - observed))
            n_samples = float(len(observed))
            
            # 상관계수 계산
            try:
                if len(observed) > 1:
                    correlation_matrix = np.corrcoef(observed, predicted)
                    correlation = float(correlation_matrix[0, 1])
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
            except:
                correlation = 0.0
                
            # 관측값 통계
            obs_mean = float(np.mean(observed))
            obs_std = float(np.std(observed))
            obs_var = float(np.var(observed))
            pred_std = float(np.std(predicted))
            
            # R² 및 NSE 계산
            r2 = 0.0
            nse = 0.0
            method_used = "correlation_squared"  # 기본 방법
            
            # 1. 변동성이 충분한 경우 전통적 R² 시도
            if obs_std > 0.005:
                try:
                    ss_res = np.sum((observed - predicted) ** 2)
                    ss_tot = np.sum((observed - obs_mean) ** 2)
                    
                    if ss_tot > 1e-10:
                        r2_traditional = 1 - (ss_res / ss_tot)
                        
                        # 합리적 범위 확인 (-3 ~ 1)
                        if -3 <= r2_traditional <= 1:
                            r2 = float(r2_traditional)
                            nse = float(r2_traditional)
                            method_used = "traditional"
                        else:
                            # 범위 벗어나면 상관계수 제곱 사용
                            r2 = float(max(0, correlation ** 2))
                            nse = r2
                            method_used = "correlation_squared_fallback"
                    else:
                        r2 = float(max(0, correlation ** 2))
                        nse = r2
                        method_used = "correlation_squared_small_var"
                except:
                    r2 = float(max(0, correlation ** 2))
                    nse = r2
                    method_used = "correlation_squared_error"
            else:
                # 변동성이 낮은 경우 상관계수 제곱 사용
                r2 = float(max(0, correlation ** 2))
                nse = r2
                method_used = "correlation_squared_low_var"
            
            # Index of Agreement 계산
            try:
                numerator = np.sum((observed - predicted) ** 2)
                denominator = np.sum((np.abs(predicted - obs_mean) + 
                                    np.abs(observed - obs_mean)) ** 2)
                ioa = float(1 - (numerator / denominator)) if denominator > 1e-10 else 0.0
            except:
                ioa = 0.0
                
            # P-value 계산
            try:
                from scipy.stats import pearsonr
                _, p_value = pearsonr(observed, predicted)
                p_value = float(p_value) if not np.isnan(p_value) else 1.0
            except:
                p_value = 1.0
                
            # 범위 제한
            r2 = float(max(-1, min(1, r2)))  # R²는 -1~1 범위
            nse = float(max(-5, min(1, nse)))  # NSE는 -5~1 범위 (실용적)
            ioa = float(max(0, min(1, ioa)))
            correlation = float(max(-1, min(1, correlation)))
            
            # 로깅
            self.logger.info(f"Validation metrics calculation:")
            self.logger.info(f"  Observed: mean={obs_mean:.6f}, std={obs_std:.6f}")
            self.logger.info(f"  Predicted: mean={np.mean(predicted):.6f}, std={pred_std:.6f}")
            self.logger.info(f"  Correlation: {correlation:.6f}")
            self.logger.info(f"  RMSE: {rmse:.6f}, MAE: {mae:.6f}, Bias: {bias:.6f}")
            self.logger.info(f"  Final metrics: R²={r2:.6f}, NSE={nse:.6f} (method: {method_used})")
            
            # 모든 값을 float로 확실히 변환하여 반환
            return {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'NSE': nse,
                'Bias': bias,
                'Correlation': correlation,
                'P_value': p_value,
                'IOA': ioa,
                'n_samples': n_samples,
                'obs_std': float(obs_std),
                'pred_std': pred_std,
                'method_used': method_used  # 이것만 문자열
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 시 기본값 반환 (모두 float)
            return {
                'R2': 0.0, 'RMSE': 1.0, 'MAE': 1.0, 'NSE': 0.0, 'Bias': 0.0,
                'Correlation': 0.0, 'P_value': 1.0, 'IOA': 0.0, 'n_samples': float(len(observed)),
                'obs_std': 0.0, 'pred_std': 0.0, 'method_used': 'error'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'R2': 0, 'RMSE': 1, 'MAE': 1, 'NSE': 0, 'Bias': 0, 'Correlation': 0,
                'P_value': 1, 'IOA': 0, 'n_samples': len(observed), 'method_used': 'error'
            }
            
    def _calculate_depth_metrics(self, fdr_data: pd.DataFrame, 
                               crnp_sm_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """깊이별 검증 지표 계산"""
        
        depth_metrics = {}
        
        if 'FDR_depth' not in fdr_data.columns:
            return depth_metrics
            
        depths = sorted(fdr_data['FDR_depth'].unique())
        
        for depth in depths:
            try:
                # 특정 깊이 데이터 추출
                depth_fdr = fdr_data[fdr_data['FDR_depth'] == depth].copy()
                depth_fdr.loc[:, 'date'] = depth_fdr['Date'].dt.date
                
                # 일별 평균
                depth_daily = depth_fdr.groupby('date')['theta_v'].mean().reset_index()
                depth_daily['date'] = pd.to_datetime(depth_daily['date'])
                
                # CRNP VWC 데이터와 매칭
                crnp_daily = crnp_sm_data[['VWC']].reset_index()
                crnp_daily.columns = ['date', 'crnp_vwc']
                crnp_daily['date'] = pd.to_datetime(crnp_daily['date'])
                
                matched = pd.merge(depth_daily, crnp_daily, on='date', how='inner')
                matched = matched.dropna()
                
                if len(matched) >= 5:  # 최소 5개 데이터 포인트
                    metrics = self._calculate_validation_metrics_robust(
                        matched['theta_v'].values, matched['crnp_vwc'].values
                    )
                    depth_metrics[f"{depth}cm"] = metrics
                    
            except Exception as e:
                self.logger.warning(f"Error calculating metrics for depth {depth}cm: {e}")
                continue
                
        return depth_metrics
        
    def _save_validation_results(self, validation_result: Dict, 
                               matched_data: pd.DataFrame) -> None:
        """검증 결과 저장"""
        
        validation_dir = self.output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 결과 저장 (Timestamp 문제 해결)
        json_file = validation_dir / f"{self.station_id}_validation_result.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2, ensure_ascii=False)
        self.logger.log_file_operation("save", str(json_file), "success")
        
        # 매칭된 데이터 저장 (별도 파일)
        if len(matched_data) > 0:
            data_file = validation_dir / f"{self.station_id}_validation_data.xlsx"
            # 인덱스를 문자열로 변환하여 저장
            matched_data_copy = matched_data.copy()
            matched_data_copy.index = matched_data_copy.index.strftime('%Y-%m-%d')
            matched_data_copy.to_excel(data_file)
            self.logger.log_file_operation("save", str(data_file), "success")
            
        # 보고서 생성
        report_content = self._generate_validation_report(validation_result)
        report_file = validation_dir / f"{self.station_id}_validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        self.logger.log_file_operation("save", str(report_file), "success")
        
    def _generate_validation_report(self, result: Dict) -> str:
        """검증 보고서 생성"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("CRNP VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Station: {result['station_id']}")
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 데이터 요약
        data_summary = result.get('data_summary', {})
        lines.append(f"Data Summary:")
        lines.append(f"  Matched days: {data_summary.get('total_matched_days', 0)}")
        
        date_range = data_summary.get('date_range', {})
        if date_range.get('start'):
            lines.append(f"  Date range: {date_range['start']} to {date_range['end']}")
        lines.append("")
        
        # 전체 성능 지표
        overall = result.get('overall_metrics', {})
        lines.append("Overall Performance Metrics:")
        lines.append(f"  R² = {overall.get('R2', 0):.4f}")
        lines.append(f"  RMSE = {overall.get('RMSE', 0):.4f}")
        lines.append(f"  MAE = {overall.get('MAE', 0):.4f}")
        lines.append(f"  NSE = {overall.get('NSE', 0):.4f}")
        lines.append(f"  Bias = {overall.get('Bias', 0):.4f}")
        lines.append(f"  Correlation = {overall.get('Correlation', 0):.4f}")
        lines.append(f"  P-value = {overall.get('P_value', 1):.6f}")
        lines.append(f"  Index of Agreement = {overall.get('IOA', 0):.4f}")
        lines.append(f"  Method used: {overall.get('method_used', 'unknown')}")
        lines.append("")
        
        # 깊이별 지표
        depth_metrics = result.get('depth_metrics', {})
        if depth_metrics:
            lines.append("Depth-wise Performance:")
            for depth, metrics in depth_metrics.items():
                lines.append(f"  {depth}: R² = {metrics.get('R2', 0):.4f}, "
                           f"RMSE = {metrics.get('RMSE', 0):.4f}")
                           
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

    def get_validation_status(self) -> Dict[str, Any]:
        """검증 상태 확인"""
        
        validation_dir = self.output_dir / "validation"
        json_file = validation_dir / f"{self.station_id}_validation_result.json"
        
        status = {
            'station_id': self.station_id,
            'validation_available': json_file.exists()
        }
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    
                status.update({
                    'validation_date': result.get('validation_timestamp'),
                    'overall_metrics': result.get('overall_metrics'),
                    'data_summary': result.get('data_summary')
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading validation file: {e}")
                
        return status