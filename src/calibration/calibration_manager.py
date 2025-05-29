# src/calibration/calibration_manager.py

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from ..core.logger import CRNPLogger, ProcessTimer
from ..core.config_manager import ConfigManager
from ..utils.file_handler import FileHandler
from .calibration_engine import CalibrationEngine


class CalibrationManager:
    """CRNP 캘리브레이션 전체 프로세스를 관리하는 클래스"""
    
    def __init__(self, station_id: str, config_root: str = "config"):
        self.station_id = station_id
        self.config_manager = ConfigManager(config_root)
        self.logger = CRNPLogger(f"CalibrationManager_{station_id}")
        
        # 설정 로드
        self.station_config = self.config_manager.load_station_config(station_id)
        self.processing_config = self.config_manager.load_processing_config()
        
        # 캘리브레이션 엔진 초기화
        self.calibration_engine = CalibrationEngine(
            self.station_config, self.processing_config, self.logger
        )
        
        # 기본 경로 설정
        self.data_paths = self.station_config['data_paths']
        self.output_dir = Path(self.data_paths.get('output_folder', f'data/output/{station_id}'))
        
    def run_calibration(self, 
                       calibration_start: Optional[str] = None,
                       calibration_end: Optional[str] = None,
                       force_recalibration: bool = False) -> Dict[str, Any]:
        """캘리브레이션 실행"""
        
        with ProcessTimer(self.logger, f"Calibration for {self.station_id}"):
            
            try:
                # 1. 캘리브레이션 기간 설정
                cal_start, cal_end = self._determine_calibration_period(
                    calibration_start, calibration_end
                )
                
                # 2. 기존 캘리브레이션 결과 확인
                if not force_recalibration:
                    existing_result = self._check_existing_calibration(cal_start, cal_end)
                    if existing_result:
                        self.logger.info("Using existing calibration result")
                        return existing_result
                        
                # 3. 필요한 데이터 파일 확인
                data_files = self._validate_calibration_data()
                
                # 4. 캘리브레이션 실행
                calibration_result = self.calibration_engine.run_calibration(
                    calibration_start=cal_start.isoformat(),
                    calibration_end=cal_end.isoformat(),
                    fdr_data_path=data_files['fdr_input'],
                    crnp_data_path=data_files['crnp_input'],
                    output_dir=str(self.output_dir / "calibration")
                )
                
                # 5. 결과 검증
                self._validate_calibration_result(calibration_result)
                
                # 6. 캘리브레이션 보고서 생성
                self._generate_calibration_report(calibration_result)
                
                return calibration_result
                
            except Exception as e:
                self.logger.log_error_with_context(e, f"Calibration for {self.station_id}")
                raise
                
    def get_calibration_status(self) -> Dict[str, Any]:
        """캘리브레이션 상태 확인"""
        
        status = {
            'station_id': self.station_id,
            'calibration_available': False,
            'calibration_file': None,
            'calibration_date': None,
            'calibration_period': None,
            'data_availability': {}
        }
        
        # 캘리브레이션 파일 확인
        calibration_dir = self.output_dir / "calibration"
        json_file = calibration_dir / f"{self.station_id}_calibration_result.json"
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cal_data = json.load(f)
                    
                status.update({
                    'calibration_available': True,
                    'calibration_file': str(json_file),
                    'calibration_date': cal_data.get('timestamp'),
                    'calibration_period': cal_data.get('calibration_period'),
                    'N0_rdt': cal_data.get('N0_rdt'),
                    'performance_metrics': cal_data.get('performance_metrics')
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading calibration file: {e}")
                
        # 데이터 가용성 확인
        status['data_availability'] = self._check_data_availability()
        
        return status
        
    def load_calibration_parameters(self) -> Optional[Dict[str, Any]]:
        """저장된 캘리브레이션 매개변수 로드"""
        
        calibration_dir = self.output_dir / "calibration"
        json_file = calibration_dir / f"{self.station_id}_calibration_result.json"
        
        if not json_file.exists():
            self.logger.warning(f"No calibration file found: {json_file}")
            return None
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
                
            # 필수 매개변수 추출
            parameters = {
                'N0_rdt': calibration_data.get('N0_rdt'),
                'Pref': calibration_data.get('Pref'),
                'Aref': calibration_data.get('Aref'),
                'Iref': calibration_data.get('Iref'),
                'clay_content': calibration_data.get('clay_content'),
                'soil_bulk_density': calibration_data.get('soil_bulk_density'),
                'lattice_water': calibration_data.get('lattice_water')
            }
            
            # None 값 확인
            missing_params = [k for k, v in parameters.items() if v is None]
            if missing_params:
                self.logger.warning(f"Missing calibration parameters: {missing_params}")
                
            self.logger.info(f"Loaded calibration parameters (N0={parameters['N0_rdt']:.2f})")
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error loading calibration parameters: {e}")
            return None
            
    def update_calibration_config(self, config_updates: Dict[str, Any]) -> None:
        """캘리브레이션 설정 업데이트"""
        
        # 처리 설정 업데이트
        if 'calibration' in config_updates:
            self.processing_config['calibration'].update(config_updates['calibration'])
            
        if 'corrections' in config_updates:
            self.processing_config['corrections'].update(config_updates['corrections'])
            
        # 캘리브레이션 엔진 재초기화
        self.calibration_engine = CalibrationEngine(
            self.station_config, self.processing_config, self.logger
        )
        
        self.logger.info("Calibration configuration updated")
        
    def _determine_calibration_period(self, start_str: Optional[str], 
                                    end_str: Optional[str]) -> Tuple[datetime, datetime]:
        """캘리브레이션 기간 결정"""
        
        if start_str and end_str:
            # 사용자 지정 기간
            cal_start = pd.to_datetime(start_str)
            cal_end = pd.to_datetime(end_str)
        else:
            # 설정 파일에서 기간 가져오기
            cal_config = self.processing_config.get('calibration', {})
            default_start = cal_config.get('default_start_date', '2024-08-17')
            default_end = cal_config.get('default_end_date', '2024-08-25')
            
            cal_start = pd.to_datetime(default_start)
            cal_end = pd.to_datetime(default_end)
            
        # 기간 유효성 검증
        if cal_start >= cal_end:
            raise ValueError(f"Invalid calibration period: {cal_start} to {cal_end}")
            
        if (cal_end - cal_start).days < 3:
            self.logger.warning("Calibration period is very short (< 3 days)")
            
        self.logger.info(f"Calibration period: {cal_start.date()} to {cal_end.date()}")
        return cal_start, cal_end
        
    def _check_existing_calibration(self, cal_start: datetime, 
                                  cal_end: datetime) -> Optional[Dict[str, Any]]:
        """기존 캘리브레이션 결과 확인"""
        
        calibration_dir = self.output_dir / "calibration"
        json_file = calibration_dir / f"{self.station_id}_calibration_result.json"
        
        if not json_file.exists():
            return None
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_result = json.load(f)
                
            # 기간 비교
            existing_start = pd.to_datetime(existing_result['calibration_period']['start'])
            existing_end = pd.to_datetime(existing_result['calibration_period']['end'])
            
            if existing_start == cal_start and existing_end == cal_end:
                return existing_result
                
        except Exception as e:
            self.logger.debug(f"Error checking existing calibration: {e}")
            
        return None
        
    def _validate_calibration_data(self) -> Dict[str, str]:
        """캘리브레이션에 필요한 데이터 파일 확인"""
        
        data_files = {}
        
        # FDR 입력 데이터 확인
        fdr_input = self.output_dir / "preprocessed" / f"{self.station_id}_FDR_input.xlsx"
        if not fdr_input.exists():
            raise FileNotFoundError(f"FDR input data not found: {fdr_input}")
        data_files['fdr_input'] = str(fdr_input)
        
        # CRNP 입력 데이터 확인
        crnp_input = self.output_dir / "preprocessed" / f"{self.station_id}_CRNP_input.xlsx"
        if not crnp_input.exists():
            raise FileNotFoundError(f"CRNP input data not found: {crnp_input}")
        data_files['crnp_input'] = str(crnp_input)
        
        self.logger.info("All required data files found")
        return data_files
        
    def _validate_calibration_result(self, result: Dict[str, Any]) -> None:
        """캘리브레이션 결과 검증"""
        
        # 필수 매개변수 확인
        required_params = ['N0_rdt', 'Pref', 'Aref']
        missing_params = [p for p in required_params if result.get(p) is None]
        
        if missing_params:
            raise ValueError(f"Missing calibration parameters: {missing_params}")
            
        # N0 값 범위 확인
        N0 = result['N0_rdt']
        if not (500 <= N0 <= 3000):
            self.logger.warning(f"N0 value outside typical range: {N0:.2f}")
            
        # 성능 지표 확인
        metrics = result.get('performance_metrics', {})
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 1)
        
        if r2 < 0.5:
            self.logger.warning(f"Low R² value: {r2:.3f}")
            
        if rmse > 0.1:
            self.logger.warning(f"High RMSE value: {rmse:.3f}")
            
        self.logger.info("Calibration result validation passed")
        
    def _check_data_availability(self) -> Dict[str, Any]:
        """데이터 가용성 확인"""
        
        availability = {}
        
        # 전처리된 데이터 확인
        preprocessed_dir = self.output_dir / "preprocessed"
        
        files_to_check = [
            f"{self.station_id}_FDR_input.xlsx",
            f"{self.station_id}_CRNP_input.xlsx"
        ]
        
        for filename in files_to_check:
            file_path = preprocessed_dir / filename
            availability[filename] = {
                'exists': file_path.exists(),
                'path': str(file_path),
                'size_mb': round(file_path.stat().st_size / (1024*1024), 2) if file_path.exists() else 0
            }
            
        return availability
        
    def _generate_calibration_report(self, calibration_result: Dict[str, Any]) -> str:
        """캘리브레이션 보고서 생성"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CRNP CALIBRATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Station: {self.station_id}")
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 캘리브레이션 기간
        period = calibration_result['calibration_period']
        report_lines.append(f"Calibration Period: {period['start']} to {period['end']}")
        report_lines.append("")
        
        # 주요 매개변수
        report_lines.append("Calibration Parameters:")
        report_lines.append(f"  N0 (Reference neutron count): {calibration_result['N0_rdt']:.2f}")
        report_lines.append(f"  Pref (Reference pressure): {calibration_result['Pref']:.2f} hPa")
        report_lines.append(f"  Aref (Reference humidity): {calibration_result['Aref']:.4f} g/cm³")
        report_lines.append(f"  Iref (Reference incoming flux): {calibration_result['Iref']:.2f}")
        report_lines.append("")
        
        # 성능 지표
        metrics = calibration_result['performance_metrics']
        report_lines.append("Performance Metrics:")
        report_lines.append(f"  R² (Coefficient of determination): {metrics.get('R2', 0):.4f}")
        report_lines.append(f"  RMSE (Root Mean Square Error): {metrics.get('RMSE', 0):.4f}")
        report_lines.append(f"  MAE (Mean Absolute Error): {metrics.get('MAE', 0):.4f}")
        report_lines.append(f"  NSE (Nash-Sutcliffe Efficiency): {metrics.get('NSE', 0):.4f}")
        report_lines.append(f"  Bias: {metrics.get('Bias', 0):.4f}")
        report_lines.append(f"  Sample size: {metrics.get('n_samples', 0)}")
        report_lines.append("")
        
        # 설정 정보
        settings = calibration_result['settings']
        report_lines.append("Configuration:")
        report_lines.append(f"  Weighting method: {settings['weighting_method']}")
        report_lines.append(f"  Reference depths: {settings['reference_depths']} cm")
        report_lines.append(f"  Neutron monitor: {settings['neutron_monitor']}")
        
        corrections = settings['corrections_enabled']
        enabled_corrections = [k for k, v in corrections.items() if v]
        report_lines.append(f"  Enabled corrections: {', '.join(enabled_corrections)}")
        report_lines.append("")
        
        # 최적화 정보
        optimization = calibration_result['optimization']
        report_lines.append("Optimization Results:")
        report_lines.append(f"  Method: {optimization['method']}")
        report_lines.append(f"  Success: {optimization['success']}")
        report_lines.append(f"  Final RMSE: {optimization['final_rmse']:.4f}")
        report_lines.append(f"  Matched data points: {optimization['matched_data_count']}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 보고서 파일 저장
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "calibration" / f"{self.station_id}_calibration_report.txt"
        
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            self.logger.log_file_operation("save", str(report_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save calibration report: {e}")
            
        return report_content


# 사용 예시 및 실행 스크립트
def main():
    """캘리브레이션 실행 예시"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CRNP 캘리브레이션 실행")
    parser.add_argument("--station", "-s", required=True, help="관측소 ID")
    parser.add_argument("--start", help="캘리브레이션 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="캘리브레이션 종료일 (YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true", help="기존 결과 무시하고 재실행")
    parser.add_argument("--status", action="store_true", help="캘리브레이션 상태만 확인")
    
    args = parser.parse_args()
    
    try:
        # CalibrationManager 초기화
        calibration_manager = CalibrationManager(args.station)
        
        if args.status:
            # 상태 확인만
            status = calibration_manager.get_calibration_status()
            
            print(f"🔍 {args.station} 관측소 캘리브레이션 상태")
            print("=" * 50)
            print(f"캘리브레이션 가능: {'✅' if status['calibration_available'] else '❌'}")
            
            if status['calibration_available']:
                print(f"N0 값: {status.get('N0_rdt', 'N/A'):.2f}")
                print(f"캘리브레이션 일자: {status.get('calibration_date', 'N/A')}")
                
                metrics = status.get('performance_metrics', {})
                if metrics:
                    print(f"R²: {metrics.get('R2', 0):.3f}")
                    print(f"RMSE: {metrics.get('RMSE', 0):.3f}")
                    
        else:
            # 캘리브레이션 실행
            print(f"🚀 {args.station} 관측소 캘리브레이션 시작")
            print("=" * 50)
            
            result = calibration_manager.run_calibration(
                calibration_start=args.start,
                calibration_end=args.end,
                force_recalibration=args.force
            )
            
            print("✅ 캘리브레이션 완료!")
            print(f"N0 = {result['N0_rdt']:.2f}")
            
            metrics = result['performance_metrics']
            print(f"R² = {metrics.get('R2', 0):.3f}")
            print(f"RMSE = {metrics.get('RMSE', 0):.3f}")
            
    except Exception as e:
        print(f"❌ 캘리브레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())