# src/calculation/soil_moisture_manager.py

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
from ..calibration.calibration_manager import CalibrationManager
from .soil_moisture_calculator import SoilMoistureCalculator


class SoilMoistureManager:
    """CRNP 토양수분 계산 전체 프로세스를 관리하는 클래스"""
    
    def __init__(self, station_id: str, config_root: str = "config"):
        self.station_id = station_id
        self.config_manager = ConfigManager(config_root)
        self.logger = CRNPLogger(f"SoilMoistureManager_{station_id}")
        
        # 설정 로드
        self.station_config = self.config_manager.load_station_config(station_id)
        self.processing_config = self.config_manager.load_processing_config()
        
        # 기본 경로 설정
        self.data_paths = self.station_config['data_paths']
        self.output_dir = Path(self.data_paths.get('output_folder', f'data/output/{station_id}'))
        
        # 캘리브레이션 매니저
        self.calibration_manager = CalibrationManager(station_id, config_root)
        
    def calculate_soil_moisture(self, 
                              calculation_start: Optional[str] = None,
                              calculation_end: Optional[str] = None,
                              force_recalculation: bool = False) -> Dict[str, Any]:
        """토양수분 계산 실행"""
        
        with ProcessTimer(self.logger, f"Soil Moisture Calculation for {self.station_id}"):
            
            try:
                # 1. 캘리브레이션 매개변수 로드
                calibration_params = self._load_calibration_parameters()
                
                # 2. 계산 기간 설정
                calc_start, calc_end = self._determine_calculation_period(
                    calculation_start, calculation_end
                )
                
                # 3. 기존 결과 확인
                if not force_recalculation:
                    existing_result = self._check_existing_calculation(calc_start, calc_end)
                    if existing_result:
                        self.logger.info("Using existing soil moisture calculation")
                        return existing_result
                        
                # 4. 필요한 데이터 파일 확인
                data_files = self._validate_calculation_data()
                
                # 5. 토양수분 계산기 초기화
                calculator = SoilMoistureCalculator(
                    self.station_config, 
                    self.processing_config, 
                    calibration_params, 
                    self.logger
                )
                
                # 6. 토양수분 계산 실행
                calculation_result = calculator.calculate_soil_moisture(
                    crnp_data_path=data_files['crnp_input'],
                    calculation_start=calc_start.isoformat() if calc_start else None,
                    calculation_end=calc_end.isoformat() if calc_end else None,
                    output_dir=str(self.output_dir / "soil_moisture")
                )
                
                # 7. 결과 검증
                self._validate_calculation_result(calculation_result)
                
                # 8. 계산 결과 메타데이터 저장
                self._save_calculation_metadata(calculation_result, calibration_params)
                
                # 9. 계산 보고서 생성
                self._generate_calculation_report(calculation_result)
                
                return calculation_result
                
            except Exception as e:
                self.logger.log_error_with_context(e, f"Soil moisture calculation for {self.station_id}")
                raise
                
    def get_calculation_status(self) -> Dict[str, Any]:
        """토양수분 계산 상태 확인"""
        
        status = {
            'station_id': self.station_id,
            'calculation_available': False,
            'calculation_file': None,
            'calculation_date': None,
            'data_summary': None,
            'calibration_status': {},
            'data_availability': {}
        }
        
        # 토양수분 결과 파일 확인
        sm_dir = self.output_dir / "soil_moisture"
        sm_file = sm_dir / f"{self.station_id}_soil_moisture.xlsx"
        metadata_file = sm_dir / f"{self.station_id}_calculation_metadata.json"
        
        if sm_file.exists():
            status.update({
                'calculation_available': True,
                'calculation_file': str(sm_file),
                'file_size_mb': round(sm_file.stat().st_size / (1024*1024), 2)
            })
            
            # 메타데이터 읽기
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    status.update({
                        'calculation_date': metadata.get('timestamp'),
                        'calculation_period': metadata.get('calculation_period'),
                        'data_summary': metadata.get('data_summary')
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error reading calculation metadata: {e}")
                    
            # 결과 데이터 기본 정보
            try:
                df = pd.read_excel(sm_file, index_col=0)
                status['data_records'] = len(df)
                status['vwc_available'] = 'VWC' in df.columns
                status['uncertainty_available'] = 'sigma_VWC' in df.columns
                
            except Exception as e:
                self.logger.warning(f"Error reading result file: {e}")
                
        # 캘리브레이션 상태 확인
        status['calibration_status'] = self.calibration_manager.get_calibration_status()
        
        # 데이터 가용성 확인
        status['data_availability'] = self._check_data_availability()
        
        return status
        
    def load_soil_moisture_data(self) -> Optional[pd.DataFrame]:
        """계산된 토양수분 데이터 로드"""
        
        sm_dir = self.output_dir / "soil_moisture"
        sm_file = sm_dir / f"{self.station_id}_soil_moisture.xlsx"
        
        if not sm_file.exists():
            self.logger.warning(f"No soil moisture file found: {sm_file}")
            return None
            
        try:
            df = pd.read_excel(sm_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            self.logger.info(f"Loaded soil moisture data: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading soil moisture data: {e}")
            return None
            
    def update_calculation_config(self, config_updates: Dict[str, Any]) -> None:
        """토양수분 계산 설정 업데이트"""
        
        # 처리 설정 업데이트
        if 'calculation' in config_updates:
            self.processing_config['calculation'].update(config_updates['calculation'])
            
        if 'corrections' in config_updates:
            self.processing_config['corrections'].update(config_updates['corrections'])
            
        self.logger.info("Soil moisture calculation configuration updated")
        
    def _load_calibration_parameters(self) -> Dict[str, Any]:
        """캘리브레이션 매개변수 로드"""
        
        calibration_params = self.calibration_manager.load_calibration_parameters()
        
        if not calibration_params:
            raise ValueError(f"No calibration parameters found for {self.station_id}. "
                           "Please run calibration first.")
                           
        # 필수 매개변수 확인
        required_params = ['N0_rdt', 'Pref', 'Aref', 'soil_bulk_density']
        missing_params = [p for p in required_params if calibration_params.get(p) is None]
        
        if missing_params:
            raise ValueError(f"Missing calibration parameters: {missing_params}")
            
        self.logger.info(f"Loaded calibration parameters (N0={calibration_params['N0_rdt']:.2f})")
        return calibration_params
        
    def _determine_calculation_period(self, start_str: Optional[str], 
                                    end_str: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """계산 기간 결정"""
        
        if start_str and end_str:
            calc_start = pd.to_datetime(start_str)
            calc_end = pd.to_datetime(end_str)
            
            # 기간 유효성 검증
            if calc_start >= calc_end:
                raise ValueError(f"Invalid calculation period: {calc_start} to {calc_end}")
                
            self.logger.info(f"Calculation period: {calc_start.date()} to {calc_end.date()}")
            return calc_start, calc_end
        else:
            self.logger.info("Using full data period for calculation")
            return None, None
            
    def _check_existing_calculation(self, calc_start: Optional[datetime], 
                                  calc_end: Optional[datetime]) -> Optional[Dict[str, Any]]:
        """기존 계산 결과 확인"""
        
        sm_dir = self.output_dir / "soil_moisture"
        metadata_file = sm_dir / f"{self.station_id}_calculation_metadata.json"
        
        if not metadata_file.exists():
            return None
            
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
                
            # 기간 비교 (기간이 지정되지 않은 경우는 항상 기존 결과 사용)
            if calc_start is None and calc_end is None:
                # 토양수분 데이터 로드
                sm_data = self.load_soil_moisture_data()
                if sm_data is not None:
                    return {
                        'soil_moisture_data': sm_data,
                        'calculation_period': existing_metadata.get('calculation_period'),
                        'data_summary': existing_metadata.get('data_summary')
                    }
                    
            elif calc_start and calc_end:
                existing_period = existing_metadata.get('calculation_period', {})
                existing_start = pd.to_datetime(existing_period.get('start'))
                existing_end = pd.to_datetime(existing_period.get('end'))
                
                if existing_start == calc_start and existing_end == calc_end:
                    sm_data = self.load_soil_moisture_data()
                    if sm_data is not None:
                        return {
                            'soil_moisture_data': sm_data,
                            'calculation_period': existing_metadata.get('calculation_period'),
                            'data_summary': existing_metadata.get('data_summary')
                        }
                        
        except Exception as e:
            self.logger.debug(f"Error checking existing calculation: {e}")
            
        return None
        
    def _validate_calculation_data(self) -> Dict[str, str]:
        """계산에 필요한 데이터 파일 확인"""
        
        data_files = {}
        
        # CRNP 입력 데이터 확인
        crnp_input = self.output_dir / "preprocessed" / f"{self.station_id}_CRNP_input.xlsx"
        if not crnp_input.exists():
            raise FileNotFoundError(f"CRNP input data not found: {crnp_input}")
        data_files['crnp_input'] = str(crnp_input)
        
        self.logger.info("All required data files found")
        return data_files
        
    def _validate_calculation_result(self, result: Dict[str, Any]) -> None:
        """계산 결과 검증"""
        
        soil_moisture_data = result.get('soil_moisture_data')
        
        if soil_moisture_data is None or len(soil_moisture_data) == 0:
            raise ValueError("No soil moisture data calculated")
            
        # VWC 값 범위 확인
        if 'VWC' in soil_moisture_data.columns:
            vwc_data = soil_moisture_data['VWC'].dropna()
            
            if len(vwc_data) == 0:
                raise ValueError("No valid VWC values calculated")
                
            # 물리적 범위 확인
            out_of_range = ((vwc_data < 0) | (vwc_data > 1)).sum()
            if out_of_range > len(vwc_data) * 0.1:  # 10% 이상 범위 밖
                self.logger.warning(f"High number of out-of-range VWC values: {out_of_range}")
                
            # 통계 확인
            vwc_mean = vwc_data.mean()
            vwc_std = vwc_data.std()
            
            if vwc_mean < 0.05 or vwc_mean > 0.8:
                self.logger.warning(f"Unusual VWC mean value: {vwc_mean:.3f}")
                
            if vwc_std > 0.3:
                self.logger.warning(f"High VWC variability: std={vwc_std:.3f}")
                
        self.logger.info("Calculation result validation passed")
        
    def _save_calculation_metadata(self, result: Dict[str, Any], 
                                 calibration_params: Dict[str, Any]) -> None:
        """계산 메타데이터 저장"""
        
        sm_dir = self.output_dir / "soil_moisture"
        sm_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'station_id': self.station_id,
            'timestamp': datetime.now().isoformat(),
            'calculation_period': result.get('calculation_period'),
            'data_summary': result.get('data_summary'),
            'calibration_parameters': calibration_params,
            'processing_config': self.processing_config.get('calculation', {}),
            'corrections_enabled': self.processing_config.get('corrections', {})
        }
        
        metadata_file = sm_dir / f"{self.station_id}_calculation_metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            self.logger.log_file_operation("save", str(metadata_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save calculation metadata: {e}")
            
    def _check_data_availability(self) -> Dict[str, Any]:
        """데이터 가용성 확인"""
        
        availability = {}
        
        # 전처리된 데이터 확인
        preprocessed_dir = self.output_dir / "preprocessed"
        crnp_file = preprocessed_dir / f"{self.station_id}_CRNP_input.xlsx"
        
        availability['crnp_input'] = {
            'exists': crnp_file.exists(),
            'path': str(crnp_file),
            'size_mb': round(crnp_file.stat().st_size / (1024*1024), 2) if crnp_file.exists() else 0
        }
        
        return availability
        
    def _generate_calculation_report(self, result: Dict[str, Any]) -> str:
        """계산 보고서 생성"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CRNP SOIL MOISTURE CALCULATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Station: {self.station_id}")
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 계산 기간
        period = result.get('calculation_period', {})
        if period:
            report_lines.append(f"Calculation Period: {period.get('start')} to {period.get('end')}")
        else:
            report_lines.append("Calculation Period: Full data period")
        report_lines.append("")
        
        # 데이터 요약
        data_summary = result.get('data_summary', {})
        if data_summary:
            report_lines.append("Data Summary:")
            report_lines.append(f"  Total days: {data_summary.get('total_days', 0)}")
            report_lines.append(f"  Valid VWC days: {data_summary.get('valid_vwc_days', 0)}")
            
            date_range = data_summary.get('date_range', {})
            report_lines.append(f"  Date range: {date_range.get('start')} to {date_range.get('end')}")
            report_lines.append("")
            
            # VWC 통계
            vwc_stats = data_summary.get('vwc_statistics', {})
            if vwc_stats:
                report_lines.append("VWC Statistics:")
                report_lines.append(f"  Mean: {vwc_stats.get('mean', 0):.4f}")
                report_lines.append(f"  Std: {vwc_stats.get('std', 0):.4f}")
                report_lines.append(f"  Min: {vwc_stats.get('min', 0):.4f}")
                report_lines.append(f"  Max: {vwc_stats.get('max', 0):.4f}")
                report_lines.append(f"  Q25: {vwc_stats.get('q25', 0):.4f}")
                report_lines.append(f"  Q75: {vwc_stats.get('q75', 0):.4f}")
                report_lines.append("")
                
            # 유효깊이 정보
            sensing_depth = data_summary.get('sensing_depth')
            if sensing_depth:
                report_lines.append("Sensing Depth:")
                report_lines.append(f"  Mean: {sensing_depth.get('mean', 0):.1f} mm")
                report_lines.append(f"  Range: {sensing_depth.get('min', 0):.1f} - {sensing_depth.get('max', 0):.1f} mm")
                report_lines.append("")
                
            # 저장량 정보
            storage = data_summary.get('storage')
            if storage:
                report_lines.append("Soil Water Storage:")
                report_lines.append(f"  Mean: {storage.get('mean', 0):.1f} mm")
                report_lines.append(f"  Std: {storage.get('std', 0):.1f} mm")
                report_lines.append("")
                
        # 설정 정보
        calc_config = self.processing_config.get('calculation', {})
        exclude_periods = calc_config.get('exclude_periods', {})
        
        if exclude_periods:
            report_lines.append("Exclusion Periods:")
            
            exclude_months = exclude_periods.get('months', [])
            if exclude_months:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                excluded_month_names = [month_names[m-1] for m in exclude_months]
                report_lines.append(f"  Excluded months: {', '.join(excluded_month_names)}")
                
            exclude_dates = exclude_periods.get('custom_dates', [])
            if exclude_dates:
                report_lines.append(f"  Excluded date ranges: {len(exclude_dates)}")
                for date_range in exclude_dates:
                    report_lines.append(f"    {date_range[0]} to {date_range[1]}")
                    
            report_lines.append("")
            
        # 스무딩 정보
        smoothing = calc_config.get('smoothing', {})
        if smoothing.get('enabled', False):
            report_lines.append("Smoothing Applied:")
            report_lines.append(f"  Method: {smoothing.get('method', 'N/A')}")
            report_lines.append(f"  Window: {smoothing.get('window', 'N/A')}")
            report_lines.append(f"  Order: {smoothing.get('order', 'N/A')}")
            report_lines.append("")
            
        report_lines.append("=" * 80)
        
        # 보고서 파일 저장
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "soil_moisture" / f"{self.station_id}_calculation_report.txt"
        
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            self.logger.log_file_operation("save", str(report_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save calculation report: {e}")
            
        return report_content


# 사용 예시 및 실행 스크립트
def main():
    """토양수분 계산 실행 예시"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CRNP 토양수분 계산 실행")
    parser.add_argument("--station", "-s", required=True, help="관측소 ID")
    parser.add_argument("--start", help="계산 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="계산 종료일 (YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true", help="기존 결과 무시하고 재실행")
    parser.add_argument("--status", action="store_true", help="계산 상태만 확인")
    
    args = parser.parse_args()
    
    try:
        # SoilMoistureManager 초기화
        sm_manager = SoilMoistureManager(args.station)
        
        if args.status:
            # 상태 확인만
            status = sm_manager.get_calculation_status()
            
            print(f"🔍 {args.station} 관측소 토양수분 계산 상태")
            print("=" * 50)
            print(f"계산 결과 가능: {'✅' if status['calculation_available'] else '❌'}")
            
            if status['calculation_available']:
                print(f"데이터 레코드: {status.get('data_records', 'N/A')}개")
                print(f"계산 일자: {status.get('calculation_date', 'N/A')}")
                
                data_summary = status.get('data_summary', {})
                if data_summary:
                    vwc_stats = data_summary.get('vwc_statistics', {})
                    print(f"VWC 평균: {vwc_stats.get('mean', 0):.3f}")
                    print(f"VWC 표준편차: {vwc_stats.get('std', 0):.3f}")
                    
            # 캘리브레이션 상태 확인
            cal_status = status.get('calibration_status', {})
            print(f"캘리브레이션: {'✅' if cal_status.get('calibration_available') else '❌'}")
            
        else:
            # 토양수분 계산 실행
            print(f"🚀 {args.station} 관측소 토양수분 계산 시작")
            print("=" * 50)
            
            result = sm_manager.calculate_soil_moisture(
                calculation_start=args.start,
                calculation_end=args.end,
                force_recalculation=args.force
            )
            
            print("✅ 토양수분 계산 완료!")
            
            data_summary = result.get('data_summary', {})
            if data_summary:
                print(f"총 데이터: {data_summary.get('total_days', 0)}일")
                print(f"유효 VWC: {data_summary.get('valid_vwc_days', 0)}일")
                
                vwc_stats = data_summary.get('vwc_statistics', {})
                print(f"VWC 평균: {vwc_stats.get('mean', 0):.3f}")
                print(f"VWC 범위: {vwc_stats.get('min', 0):.3f} - {vwc_stats.get('max', 0):.3f}")
                
    except Exception as e:
        print(f"❌ 토양수분 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())