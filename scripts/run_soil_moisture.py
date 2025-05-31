# scripts/run_soil_moisture.py

"""
CRNP 토양수분 계산 실행 스크립트

사용법:
    python scripts/run_soil_moisture.py --station HC
    python scripts/run_soil_moisture.py --station HC --start 2024-08-01 --end 2024-12-31
    python scripts/run_soil_moisture.py --station HC --status
    python scripts/run_soil_moisture.py --station HC --force
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calculation.soil_moisture_manager import SoilMoistureManager
from src.calibration.calibration_manager import CalibrationManager
from src.core.logger import setup_logger


class SoilMoistureRunner:
    """토양수분 계산 실행 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("SoilMoistureRunner")
        
    def run_soil_moisture_calculation(self, station_id: str,
                                    start_date: str = None, end_date: str = None,
                                    force: bool = False, status_only: bool = False,
                                    with_validation: bool = True) -> bool:
        """토양수분 계산 실행"""
        
        print(f"💧 CRNP Soil Moisture Calculation - {station_id} Station")
        print("=" * 70)
        
        try:
            # SoilMoistureManager 초기화
            sm_manager = SoilMoistureManager(station_id)
            
            if status_only:
                # 상태 확인만
                return self._show_calculation_status(sm_manager)
            
            # 1. 캘리브레이션 상태 확인
            print("🔍 1단계: 캘리브레이션 확인")
            calibration_status = self._check_calibration_status(station_id)
            
            if not calibration_status['available']:
                print("❌ Calibration result not found")
                print("💡 Please run calibration first:")
                print(f"   python scripts/run_calibration.py --station {station_id}")
                return False
                
            print("   ✅ Calibration result found")
            print(f"   📊 N0 = {calibration_status['N0']:.2f}")
            print(f"   📈 R² = {calibration_status['R2']:.3f}")
            
            # 2. 데이터 가용성 확인
            print("\n📁 2단계: 데이터 가용성 확인")
            data_status = self._check_data_availability(station_id, start_date, end_date)
            
            if not data_status['sufficient']:
                print("❌ Insufficient data for calculation")
                for issue in data_status['issues']:
                    print(f"   ⚠️  {issue}")
                return False
                
            print("   ✅ CRNP data available")
            print(f"   📊 Records: {data_status['total_records']}")
            print(f"   📅 Period: {data_status['date_range']['start']} ~ {data_status['date_range']['end']}")
            
            # 3. 기존 결과 확인
            if not force:
                existing_status = sm_manager.get_calculation_status()
                if existing_status['calculation_available']:
                    print("\n📋 Existing calculation result found.")
                    
                    # 기간 확인
                    if start_date and end_date:
                        # 기간 비교 로직 (필요시)
                        pass
                        
                    user_input = input("Overwrite existing result? (y/N): ")
                    if user_input.lower() != 'y':
                        print("Calculation cancelled.")
                        return False
                        
            # 4. 토양수분 계산 실행
            print(f"\n🔄 3단계: 토양수분 계산 실행")
            if start_date and end_date:
                print(f"   기간: {start_date} ~ {end_date}")
            else:
                print("   기간: 전체 데이터 기간")
                
            result = sm_manager.calculate_soil_moisture(
                calculation_start=start_date,
                calculation_end=end_date,
                force_recalculation=True
            )
            
            # 5. 결과 분석 및 출력
            self._analyze_and_display_results(result, station_id, data_status)
            
            # 6. 검증 (선택사항)
            if with_validation:
                self._run_optional_validation(station_id)
                
            return True
            
        except Exception as e:
            print(f"❌ Soil moisture calculation failed: {e}")
            self.logger.error(f"Calculation failed for {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _check_calibration_status(self, station_id: str) -> Dict[str, Any]:
        """캘리브레이션 상태 확인"""
        
        try:
            calibration_manager = CalibrationManager(station_id)
            status = calibration_manager.get_calibration_status()
            
            return {
                'available': status['calibration_available'],
                'N0': status.get('N0_rdt', 0),
                'R2': status.get('performance_metrics', {}).get('R2', 0),
                'file_path': status.get('calibration_file')
            }
            
        except Exception as e:
            self.logger.warning(f"Error checking calibration status: {e}")
            return {'available': False}
            
    def _check_data_availability(self, station_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """데이터 가용성 확인"""
        
        status = {
            'sufficient': False,
            'issues': [],
            'total_records': 0,
            'date_range': {}
        }
        
        try:
            # CRNP 전처리 데이터 확인
            crnp_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_CRNP_input.xlsx"
            
            if not crnp_file.exists():
                status['issues'].append("CRNP preprocessed data not found")
                status['issues'].append("Run preprocessing first")
                return status
                
            # CRNP 데이터 로드 및 분석
            crnp_data = pd.read_excel(crnp_file)
            
            if 'timestamp' not in crnp_data.columns:
                # 첫 번째 컬럼이 타임스탬프일 가능성
                crnp_data['timestamp'] = pd.to_datetime(crnp_data.iloc[:, 0], errors='coerce')
                
            if 'timestamp' in crnp_data.columns:
                crnp_data['timestamp'] = pd.to_datetime(crnp_data['timestamp'], errors='coerce')
                crnp_data = crnp_data.dropna(subset=['timestamp'])
                
                if len(crnp_data) == 0:
                    status['issues'].append("No valid timestamp data")
                    return status
                    
                status['total_records'] = len(crnp_data)
                status['date_range'] = {
                    'start': str(crnp_data['timestamp'].min().date()),
                    'end': str(crnp_data['timestamp'].max().date())
                }
                
                # 기간 필터링 (지정된 경우)
                if start_date and end_date:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    period_data = crnp_data[
                        (crnp_data['timestamp'] >= start_dt) & 
                        (crnp_data['timestamp'] <= end_dt)
                    ]
                    
                    if len(period_data) == 0:
                        status['issues'].append(f"No data in specified period ({start_date} ~ {end_date})")
                        return status
                        
                    status['total_records'] = len(period_data)
                    status['date_range'] = {
                        'start': start_date,
                        'end': end_date
                    }
                    
                # 중성자 카운트 확인
                neutron_col = None
                for col in ['N_counts', 'total_raw_counts']:
                    if col in crnp_data.columns:
                        neutron_col = col
                        break
                        
                if neutron_col is None:
                    status['issues'].append("No neutron counts column found")
                    return status
                    
                neutron_valid = crnp_data[neutron_col].notna().sum()
                if neutron_valid < len(crnp_data) * 0.5:
                    status['issues'].append("Too many missing neutron count values")
                    return status
                    
                # 기상 데이터 확인
                weather_cols = ['Ta', 'RH', 'Pa']
                missing_weather = [col for col in weather_cols if col not in crnp_data.columns]
                if missing_weather:
                    status['issues'].append(f"Missing weather data: {missing_weather}")
                    
                print(f"   📊 Data quality check:")
                print(f"      Total records: {status['total_records']}")
                print(f"      Valid neutron data: {neutron_valid} ({neutron_valid/len(crnp_data)*100:.1f}%)")
                
                weather_completeness = []
                for col in weather_cols:
                    if col in crnp_data.columns:
                        completeness = crnp_data[col].notna().sum() / len(crnp_data) * 100
                        weather_completeness.append(f"{col}: {completeness:.1f}%")
                        
                if weather_completeness:
                    print(f"      Weather data: {', '.join(weather_completeness)}")
                    
                # 충분한 데이터가 있으면 통과
                if neutron_valid >= 30:  # 최소 30개 데이터
                    status['sufficient'] = True
                else:
                    status['issues'].append("Insufficient valid data points")
                    
            else:
                status['issues'].append("No valid timestamp column")
                
        except Exception as e:
            status['issues'].append(f"Data check failed: {e}")
            
        return status
        
    def _show_calculation_status(self, sm_manager: SoilMoistureManager) -> bool:
        """토양수분 계산 상태 표시"""
        
        print("📊 Soil Moisture Calculation Status")
        print("-" * 50)
        
        status = sm_manager.get_calculation_status()
        
        print(f"Station: {status['station_id']}")
        
        # 계산 결과 가용성
        if status['calculation_available']:
            print("✅ Calculation result available")
            
            if status.get('data_records'):
                print(f"   Records: {status['data_records']} days")
                
            if status.get('calculation_date'):
                print(f"   Generated: {status['calculation_date']}")
                
            # 데이터 요약
            data_summary = status.get('data_summary')
            if data_summary:
                vwc_stats = data_summary.get('vwc_statistics', {})
                print(f"\n📈 VWC Statistics:")
                print(f"   Mean: {vwc_stats.get('mean', 0):.3f}")
                print(f"   Std: {vwc_stats.get('std', 0):.3f}")
                print(f"   Range: {vwc_stats.get('min', 0):.3f} ~ {vwc_stats.get('max', 0):.3f}")
                
                sensing_depth = data_summary.get('sensing_depth')
                if sensing_depth:
                    print(f"\n🎯 Sensing Depth:")
                    print(f"   Mean: {sensing_depth.get('mean', 0):.1f} mm")
                    print(f"   Range: {sensing_depth.get('min', 0):.1f} ~ {sensing_depth.get('max', 0):.1f} mm")
                    
                storage = data_summary.get('storage')
                if storage:
                    print(f"\n💧 Storage:")
                    print(f"   Mean: {storage.get('mean', 0):.1f} mm")
                    print(f"   Std: {storage.get('std', 0):.1f} mm")
                    
        else:
            print("❌ No calculation result found")
            
        # 캘리브레이션 상태
        print(f"\n🔬 Calibration Status:")
        cal_status = status.get('calibration_status', {})
        
        if cal_status.get('calibration_available'):
            print("   ✅ Available")
            if cal_status.get('N0_rdt'):
                print(f"   N0: {cal_status['N0_rdt']:.2f}")
        else:
            print("   ❌ Not available")
            
        # 데이터 파일
        print(f"\n📁 Data Files:")
        data_availability = status.get('data_availability', {})
        
        for filename, file_info in data_availability.items():
            status_icon = "✅" if file_info['exists'] else "❌"
            size_info = f"({file_info['size_mb']} MB)" if file_info['exists'] else ""
            print(f"   {filename}: {status_icon} {size_info}")
            
        # 다음 단계 안내
        if not status['calculation_available']:
            print(f"\n💡 Next Steps:")
            
            if not cal_status.get('calibration_available'):
                print(f"   1. Run calibration: python scripts/run_calibration.py --station {status['station_id']}")
                print(f"   2. Run calculation: python scripts/run_soil_moisture.py --station {status['station_id']}")
            else:
                print(f"   1. Run calculation: python scripts/run_soil_moisture.py --station {status['station_id']}")
                
        return status['calculation_available']
        
    def _analyze_and_display_results(self, result: Dict[str, Any], station_id: str, data_status: Dict) -> None:
        """결과 분석 및 표시"""
        
        print(f"\n📊 Calculation Results:")
        
        # 데이터 요약
        data_summary = result.get('data_summary', {})
        
        if data_summary:
            print(f"   Total days processed: {data_summary.get('total_days', 0)}")
            print(f"   Valid VWC days: {data_summary.get('valid_vwc_days', 0)}")
            
            date_range = data_summary.get('date_range', {})
            print(f"   Date range: {date_range.get('start')} ~ {date_range.get('end')}")
            
            # VWC 통계
            vwc_stats = data_summary.get('vwc_statistics', {})
            if vwc_stats:
                print(f"\n💧 VWC Statistics:")
                print(f"   Mean: {vwc_stats.get('mean', 0):.4f} m³/m³")
                print(f"   Std: {vwc_stats.get('std', 0):.4f} m³/m³")
                print(f"   Range: {vwc_stats.get('min', 0):.4f} ~ {vwc_stats.get('max', 0):.4f} m³/m³")
                print(f"   Q25/Q75: {vwc_stats.get('q25', 0):.4f} / {vwc_stats.get('q75', 0):.4f} m³/m³")
                
            # 유효깊이
            sensing_depth = data_summary.get('sensing_depth')
            if sensing_depth:
                print(f"\n🎯 Sensing Depth:")
                print(f"   Mean: {sensing_depth.get('mean', 0):.1f} mm")
                print(f"   Range: {sensing_depth.get('min', 0):.1f} ~ {sensing_depth.get('max', 0):.1f} mm")
                
            # 저장량
            storage = data_summary.get('storage')
            if storage:
                print(f"\n🏔️  Storage:")
                print(f"   Mean: {storage.get('mean', 0):.1f} mm")
                print(f"   Std: {storage.get('std', 0):.1f} mm")
                
        # 품질 평가
        print(f"\n📈 Quality Assessment:")
        
        valid_ratio = data_summary.get('valid_vwc_days', 0) / max(1, data_summary.get('total_days', 1))
        
        if valid_ratio >= 0.9:
            print("   🟢 Excellent data completeness (≥90%)")
        elif valid_ratio >= 0.8:
            print("   🟡 Good data completeness (≥80%)")
        elif valid_ratio >= 0.7:
            print("   🟠 Fair data completeness (≥70%)")
        else:
            print("   🔴 Poor data completeness (<70%)")
            
        # VWC 값 범위 평가
        if vwc_stats:
            vwc_mean = vwc_stats.get('mean', 0)
            vwc_std = vwc_stats.get('std', 0)
            
            if 0.1 <= vwc_mean <= 0.6:
                print("   🟢 VWC values in reasonable range")
            else:
                print("   🟠 VWC values may be outside typical range")
                
            if vwc_std >= 0.02:
                print("   🟢 Good temporal variability")
            elif vwc_std >= 0.01:
                print("   🟡 Moderate temporal variability")
            else:
                print("   🟠 Low temporal variability")
                
        # 생성된 파일들
        print(f"\n📁 Generated Files:")
        print(f"   📊 Main result: {station_id}_soil_moisture.xlsx")
        print(f"   📋 Metadata: {station_id}_calculation_metadata.json")
        print(f"   📄 Report: {station_id}_calculation_report.txt")
        
        # 다음 단계
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review generated plots:")
        print(f"      python scripts/run_visualization.py --station {station_id}")
        print(f"   2. Validate with field data (if available):")
        print(f"      python scripts/run_validation.py --station {station_id}")
        print(f"   3. Full pipeline with visualization:")
        print(f"      python scripts/run_crnp_pipeline.py --station {station_id} --steps soil_moisture visualization")
        
    def _run_optional_validation(self, station_id: str) -> None:
        """선택적 검증 실행"""
        
        try:
            print(f"\n🔍 4단계: 검증 (선택사항)")
            
            # FDR 데이터 확인
            fdr_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_FDR_input.xlsx"
            
            if fdr_file.exists():
                user_input = input("Run validation against field sensors? (y/N): ")
                if user_input.lower() == 'y':
                    from src.validation.validation_manager import ValidationManager
                    
                    validation_manager = ValidationManager(station_id)
                    validation_result = validation_manager.run_validation()
                    
                    print("   ✅ Validation completed")
                    
                    overall_metrics = validation_result.get('overall_metrics', {})
                    if overall_metrics:
                        print(f"   📊 R² = {overall_metrics.get('R2', 0):.3f}")
                        print(f"   📊 RMSE = {overall_metrics.get('RMSE', 0):.3f}")
                else:
                    print("   ⏭️  Validation skipped")
            else:
                print("   ℹ️  No field sensor data available for validation")
                
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP Soil Moisture Calculation")
    
    parser.add_argument("--station", "-s", required=True, help="Station ID (HC, PC, etc.)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force recalculation (ignore existing results)")
    parser.add_argument("--status", action="store_true", 
                       help="Check calculation status only")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip validation step")
    
    args = parser.parse_args()
    
    # 날짜 유효성 검사
    if (args.start and not args.end) or (args.end and not args.start):
        print("❌ Both start and end dates must be provided together")
        return 1
        
    if args.start and args.end:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(args.start, '%Y-%m-%d')
            end_dt = datetime.strptime(args.end, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                print("❌ Start date must be before end date")
                return 1
                
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return 1
            
    # 토양수분 계산 실행
    runner = SoilMoistureRunner()
    success = runner.run_soil_moisture_calculation(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        status_only=args.status,
        with_validation=not args.no_validation
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())