# src/validation/validation_manager.py

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score

from ..core.logger import CRNPLogger, ProcessTimer
from ..core.config_manager import ConfigManager
from ..utils.file_handler import FileHandler


class ValidationManager:
    """CRNP 검증을 관리하는 클래스"""
    
    def __init__(self, station_id: str, config_root: str = "config"):
        self.station_id = station_id
        self.config_manager = ConfigManager(config_root)
        self.logger = CRNPLogger(f"ValidationManager_{station_id}")
        
        # 설정 로드
        try:
            self.station_config = self.config_manager.load_station_config(station_id)
            self.processing_config = self.config_manager.load_processing_config()
        except Exception as e:
            self.logger.warning(f"Could not load configurations: {e}")
            self.station_config = {'station_info': {'id': station_id}}
            self.processing_config = {}
        
        # 기본 경로 설정
        self.data_paths = self.station_config.get('data_paths', {})
        self.output_dir = Path(f"data/output/{station_id}")
        
    def run_validation(self, fdr_data_path: Optional[str] = None,
                      crnp_sm_data_path: Optional[str] = None) -> Dict[str, Any]:
        """검증 실행"""
        
        with ProcessTimer(self.logger, f"Validation for {self.station_id}"):
            
            try:
                # 1. 데이터 파일 경로 설정
                if not fdr_data_path:
                    fdr_data_path = self.output_dir / "preprocessed" / f"{self.station_id}_FDR_input.xlsx"
                
                if not crnp_sm_data_path:
                    crnp_sm_data_path = self.output_dir / "soil_moisture" / f"{self.station_id}_soil_moisture.xlsx"
                
                # 2. 데이터 로드
                fdr_data, crnp_data = self._load_validation_data(fdr_data_path, crnp_sm_data_path)
                
                # 3. 데이터 매칭
                matched_data = self._match_validation_data(fdr_data, crnp_data)
                
                if len(matched_data) == 0:
                    raise ValueError("No matching data found for validation")
                
                # 4. 성능 지표 계산
                metrics = self._calculate_validation_metrics(matched_data)
                
                # 5. 깊이별 분석 (가능한 경우)
                depth_analysis = self._analyze_by_depth(fdr_data, crnp_data)
                
                # 6. 검증 결과 생성
                validation_result = {
                    'station_id': self.station_id,
                    'validation_timestamp': datetime.now().isoformat(),
                    'matched_data_count': len(matched_data),
                    'overall_metrics': metrics,
                    'depth_analysis': depth_analysis,
                    'data_period': {
                        'start': str(matched_data.index.min().date()),
                        'end': str(matched_data.index.max().date())
                    }
                }
                
                # 7. 결과 저장
                self._save_validation_results(validation_result, matched_data)
                
                self.logger.log_validation_result(metrics)
                
                return validation_result
                
            except Exception as e:
                self.logger.log_error_with_context(e, f"Validation for {self.station_id}")
                raise
                
    def _load_validation_data(self, fdr_path: str, crnp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """검증 데이터 로드"""
        
        with ProcessTimer(self.logger, "Loading validation data"):
            
            # FDR 데이터 로드
            if not os.path.exists(fdr_path):
                raise FileNotFoundError(f"FDR data not found: {fdr_path}")
                
            self.logger.info(f"Loading FDR data from {fdr_path}")
            fdr_data = pd.read_excel(fdr_path)
            
            # 날짜 컬럼 처리
            if 'Date' in fdr_data.columns:
                fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
            else:
                raise ValueError("Date column not found in FDR data")
                
            # CRNP 토양수분 데이터 로드
            if not os.path.exists(crnp_path):
                raise FileNotFoundError(f"CRNP soil moisture data not found: {crnp_path}")
                
            self.logger.info(f"Loading CRNP SM data from {crnp_path}")
            crnp_data = pd.read_excel(crnp_path, index_col=0)
            crnp_data.index = pd.to_datetime(crnp_data.index)
            
            self.logger.log_data_summary("FDR_Validation", len(fdr_data))
            self.logger.log_data_summary("CRNP_SM_Validation", len(crnp_data))
            
            return fdr_data, crnp_data
            
    def _match_validation_data(self, fdr_data: pd.DataFrame, 
                              crnp_data: pd.DataFrame) -> pd.DataFrame:
        """검증 데이터 매칭"""
        
        with ProcessTimer(self.logger, "Matching validation data"):
            
            # FDR 데이터 일별 평균 계산
            if 'theta_v' not in fdr_data.columns:
                self.logger.warning("theta_v column not found in FDR data")
                return pd.DataFrame()
                
            # 날짜별 FDR 평균 계산
            fdr_data['date'] = fdr_data['Date'].dt.date
            fdr_daily = fdr_data.groupby('date')['theta_v'].mean()
            
            # CRNP 데이터에서 VWC 추출
            if 'VWC' not in crnp_data.columns:
                self.logger.warning("VWC column not found in CRNP data")
                return pd.DataFrame()
                
            crnp_daily = crnp_data['VWC'].copy()
            crnp_daily.index = crnp_daily.index.date
            
            # 공통 날짜 찾기
            common_dates = set(fdr_daily.index).intersection(set(crnp_daily.index))
            
            if len(common_dates) == 0:
                self.logger.warning("No common dates found")
                return pd.DataFrame()
                
            # 매칭된 데이터 생성
            matched_data = pd.DataFrame({
                'Field_SM': fdr_daily.loc[list(common_dates)],
                'CRNP_SM': crnp_daily.loc[list(common_dates)]
            })
            
            # 인덱스를 datetime으로 변환
            matched_data.index = pd.to_datetime(matched_data.index)
            
            # NaN 값 제거
            matched_data = matched_data.dropna()
            
            self.logger.info(f"Matched {len(matched_data)} data points")
            
            return matched_data
            
    def _calculate_validation_metrics(self, matched_data: pd.DataFrame) -> Dict[str, float]:
        """검증 지표 계산"""
        
        if len(matched_data) == 0:
            return {}
            
        field_sm = matched_data['Field_SM'].values
        crnp_sm = matched_data['CRNP_SM'].values
        
        try:
            # R²
            r2 = r2_score(field_sm, crnp_sm)
            
            # RMSE
            rmse = np.sqrt(np.mean((field_sm - crnp_sm) ** 2))
            
            # MAE
            mae = np.mean(np.abs(field_sm - crnp_sm))
            
            # Bias
            bias = np.mean(crnp_sm - field_sm)
            
            # Nash-Sutcliffe Efficiency
            nse = 1 - (np.sum((field_sm - crnp_sm) ** 2) / 
                      np.sum((field_sm - np.mean(field_sm)) ** 2))
            
            # Pearson 상관계수
            correlation, p_value = stats.pearsonr(field_sm, crnp_sm)
            
            # Index of Agreement
            numerator = np.sum((field_sm - crnp_sm) ** 2)
            denominator = np.sum((np.abs(crnp_sm - np.mean(field_sm)) + 
                                np.abs(field_sm - np.mean(field_sm))) ** 2)
            ioa = 1 - (numerator / denominator) if denominator != 0 else 0
            
            metrics = {
                'R2': float(r2),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'Bias': float(bias),
                'NSE': float(nse),
                'Correlation': float(correlation),
                'P_value': float(p_value),
                'IOA': float(ioa),
                'n_samples': len(matched_data)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            return {}
            
    def _analyze_by_depth(self, fdr_data: pd.DataFrame, 
                         crnp_data: pd.DataFrame) -> Dict[str, Any]:
        """깊이별 분석"""
        
        depth_analysis = {}
        
        if 'FDR_depth' not in fdr_data.columns:
            return depth_analysis
            
        depths = sorted(fdr_data['FDR_depth'].unique())
        
        for depth in depths:
            try:
                # 특정 깊이 데이터
                depth_fdr = fdr_data[fdr_data['FDR_depth'] == depth]
                
                if len(depth_fdr) == 0:
                    continue
                    
                # 일별 평균
                depth_fdr['date'] = depth_fdr['Date'].dt.date
                depth_daily = depth_fdr.groupby('date')['theta_v'].mean()
                
                # CRNP 데이터와 매칭
                crnp_daily = crnp_data['VWC'].copy()
                crnp_daily.index = crnp_daily.index.date
                
                common_dates = set(depth_daily.index).intersection(set(crnp_daily.index))
                
                if len(common_dates) >= 5:  # 최소 5개 데이터
                    depth_matched = pd.DataFrame({
                        'Field_SM': depth_daily.loc[list(common_dates)],
                        'CRNP_SM': crnp_daily.loc[list(common_dates)]
                    }).dropna()
                    
                    if len(depth_matched) >= 5:
                        metrics = self._calculate_validation_metrics(depth_matched)
                        depth_analysis[f"{depth}cm"] = {
                            'metrics': metrics,
                            'data_count': len(depth_matched)
                        }
                        
            except Exception as e:
                self.logger.warning(f"Depth analysis failed for {depth}cm: {e}")
                
        return depth_analysis
        
    def _save_validation_results(self, result: Dict[str, Any], 
                               matched_data: pd.DataFrame) -> None:
        """검증 결과 저장"""
        
        validation_dir = self.output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 결과 저장
        json_file = validation_dir / f"{self.station_id}_validation_result.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            self.logger.log_file_operation("save", str(json_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save validation JSON: {e}")
            
        # 매칭된 데이터 저장
        data_file = validation_dir / f"{self.station_id}_validation_data.xlsx"
        
        try:
            matched_data.to_excel(data_file, index=True)
            self.logger.log_file_operation("save", str(data_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save validation data: {e}")
            
        # 검증 보고서 생성
        self._generate_validation_report(result, validation_dir)
        
    def _generate_validation_report(self, result: Dict[str, Any], output_dir: Path) -> None:
        """검증 보고서 생성"""
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CRNP VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Station: {self.station_id}")
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 검증 기간
        data_period = result.get('data_period', {})
        report_lines.append(f"Validation Period: {data_period.get('start')} to {data_period.get('end')}")
        report_lines.append(f"Matched Data Points: {result.get('matched_data_count', 0)}")
        report_lines.append("")
        
        # 전체 성능 지표
        metrics = result.get('overall_metrics', {})
        if metrics:
            report_lines.append("Overall Performance Metrics:")
            report_lines.append(f"  R² (Coefficient of Determination): {metrics.get('R2', 0):.4f}")
            report_lines.append(f"  RMSE (Root Mean Square Error): {metrics.get('RMSE', 0):.4f}")
            report_lines.append(f"  MAE (Mean Absolute Error): {metrics.get('MAE', 0):.4f}")
            report_lines.append(f"  Bias: {metrics.get('Bias', 0):.4f}")
            report_lines.append(f"  NSE (Nash-Sutcliffe Efficiency): {metrics.get('NSE', 0):.4f}")
            report_lines.append(f"  Correlation: {metrics.get('Correlation', 0):.4f}")
            report_lines.append(f"  Index of Agreement: {metrics.get('IOA', 0):.4f}")
            report_lines.append("")
            
        # 깊이별 분석
        depth_analysis = result.get('depth_analysis', {})
        if depth_analysis:
            report_lines.append("Depth-wise Analysis:")
            report_lines.append("-" * 30)
            
            for depth, analysis in depth_analysis.items():
                depth_metrics = analysis.get('metrics', {})
                data_count = analysis.get('data_count', 0)
                
                report_lines.append(f"{depth} Depth:")
                report_lines.append(f"  Data Points: {data_count}")
                report_lines.append(f"  R²: {depth_metrics.get('R2', 0):.4f}")
                report_lines.append(f"  RMSE: {depth_metrics.get('RMSE', 0):.4f}")
                report_lines.append(f"  Correlation: {depth_metrics.get('Correlation', 0):.4f}")
                report_lines.append("")
                
        # 성능 평가
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        
        report_lines.append("Performance Assessment:")
        if r2 >= 0.8 and rmse <= 0.05:
            report_lines.append("  🟢 EXCELLENT - High accuracy model")
        elif r2 >= 0.6 and rmse <= 0.1:
            report_lines.append("  🟡 GOOD - Acceptable accuracy")
        elif r2 >= 0.4:
            report_lines.append("  🟠 FAIR - Moderate accuracy")
        else:
            report_lines.append("  🔴 POOR - Low accuracy")
            
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # 보고서 저장
        report_file = output_dir / f"{self.station_id}_validation_report.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
                
            self.logger.log_file_operation("save", str(report_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save validation report: {e}")
            
    def get_validation_status(self) -> Dict[str, Any]:
        """검증 상태 확인"""
        
        validation_dir = self.output_dir / "validation"
        json_file = validation_dir / f"{self.station_id}_validation_result.json"
        
        status = {
            'station_id': self.station_id,
            'validation_available': json_file.exists(),
            'validation_file': str(json_file) if json_file.exists() else None
        }
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    validation_result = json.load(f)
                    
                status.update({
                    'validation_date': validation_result.get('validation_timestamp'),
                    'matched_data_count': validation_result.get('matched_data_count'),
                    'overall_metrics': validation_result.get('overall_metrics')
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading validation file: {e}")
                
        return status


# 사용 예시
if __name__ == "__main__":
    # ValidationManager 테스트
    validation_manager = ValidationManager("PC")
    
    try:
        result = validation_manager.run_validation()
        print("✅ 검증 완료!")
        
        overall_metrics = result.get('overall_metrics', {})
        print(f"R² = {overall_metrics.get('R2', 0):.3f}")
        print(f"RMSE = {overall_metrics.get('RMSE', 0):.3f}")
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")