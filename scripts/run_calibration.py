# scripts/run_calibration.py

"""
CRNP 캘리브레이션 실행 스크립트

사용법:
    python scripts/run_calibration.py --station HC
    python scripts/run_calibration.py --station HC --start 2024-08-17 --end 2024-08-25
    python scripts/run_calibration.py --station HC --status
    python scripts/run_calibration.py --station HC --force
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calibration.calibration_manager import CalibrationManager
from src.core.logger import setup_logger


class CalibrationRunner:
    """캘리브레이션 실행을 관리하는 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("CalibrationRunner")
        
    def run_calibration(self, station_id: str, 
                       start_date: str = None, end_date: str = None,
                       force: bool = False, status_only: bool = False) -> bool:
        """캘리브레이션 실행"""
        
        print(f"🔬 CRNP 캘리브레이션 - {station_id} 관측소")
        print("=" * 60)
        
        try:
            # CalibrationManager 초기화
            calibration_manager = CalibrationManager(station_id)
            
            if status_only:
                # 상태 확인만
                return self._show_calibration_status(calibration_manager)
            else:
                # 캘리브레이션 실행
                return self._execute_calibration(
                    calibration_manager, start_date, end_date, force
                )
                
        except Exception as e:
            print(f"❌ 캘리브레이션 실패: {e}")
            self.logger.error(f"Calibration failed for {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _show_calibration_status(self, calibration_manager: CalibrationManager) -> bool:
        """캘리브레이션 상태 표시"""
        
        print("📊 캘리브레이션 상태 확인 중...")
        status = calibration_manager.get_calibration_status()
        
        print(f"\n🔍 {status['station_id']} 관측소 캘리브레이션 상태:")
        print("-" * 50)
        
        # 캘리브레이션 가용성
        if status['calibration_available']:
            print("✅ 캘리브레이션 결과 있음")
            
            # 캘리브레이션 정보
            if status.get('N0_rdt'):
                print(f"   N0 값: {status['N0_rdt']:.2f}")
                
            if status.get('calibration_period'):
                period = status['calibration_period']
                print(f"   캘리브레이션 기간: {period['start']} ~ {period['end']}")
                
            if status.get('calibration_date'):
                print(f"   생성 일시: {status['calibration_date']}")
                
            # 성능 지표
            metrics = status.get('performance_metrics', {})
            if metrics:
                print("\n📈 성능 지표:")
                print(f"   R² = {metrics.get('R2', 0):.3f}")
                print(f"   RMSE = {metrics.get('RMSE', 0):.3f}")
                print(f"   MAE = {metrics.get('MAE', 0):.3f}")
                print(f"   데이터 개수 = {metrics.get('n_samples', 0)}")
                
        else:
            print("❌ 캘리브레이션 결과 없음")
            
        # 데이터 가용성
        print(f"\n📁 데이터 파일 상태:")
        data_availability = status.get('data_availability', {})
        
        for filename, file_info in data_availability.items():
            status_icon = "✅" if file_info['exists'] else "❌"
            size_info = f"({file_info['size_mb']} MB)" if file_info['exists'] else ""
            print(f"   {filename}: {status_icon} {size_info}")
            
        # 다음 단계 안내
        if not status['calibration_available']:
            print(f"\n💡 다음 단계:")
            print(f"   1. 전처리 실행: python scripts/run_preprocessing.py --station {status['station_id']}")
            print(f"   2. 캘리브레이션 실행: python scripts/run_calibration.py --station {status['station_id']}")
            
        return status['calibration_available']
        
    def _execute_calibration(self, calibration_manager: CalibrationManager,
                           start_date: str, end_date: str, force: bool) -> bool:
        """캘리브레이션 실행"""
        
        # 기존 결과 확인
        if not force:
            existing_status = calibration_manager.get_calibration_status()
            if existing_status['calibration_available']:
                print("📋 기존 캘리브레이션 결과가 있습니다.")
                
                # 기간 확인
                if start_date and end_date:
                    existing_period = existing_status.get('calibration_period', {})
                    existing_start = existing_period.get('start', '').split('T')[0]
                    existing_end = existing_period.get('end', '').split('T')[0]
                    
                    if existing_start == start_date and existing_end == end_date:
                        print("✅ 동일한 기간의 캘리브레이션 결과를 사용합니다.")
                        self._print_calibration_summary(existing_status)
                        return True
                        
                user_input = input("기존 결과를 덮어쓰시겠습니까? (y/N): ")
                if user_input.lower() != 'y':
                    print("캘리브레이션을 취소했습니다.")
                    return False
                    
        # 캘리브레이션 기간 표시
        if start_date and end_date:
            print(f"📅 캘리브레이션 기간: {start_date} ~ {end_date}")
        else:
            print("📅 기본 캘리브레이션 기간 사용")
            
        # 필수 데이터 확인
        print("\n📊 필수 데이터 확인 중...")
        status = calibration_manager.get_calibration_status()
        data_availability = status.get('data_availability', {})
        
        missing_files = []
        for filename, file_info in data_availability.items():
            if not file_info['exists']:
                missing_files.append(filename)
                
        if missing_files:
            print(f"❌ 필수 데이터 파일이 누락되었습니다:")
            for filename in missing_files:
                print(f"   - {filename}")
            print(f"\n💡 전처리를 먼저 실행해주세요:")
            print(f"   python scripts/run_preprocessing.py --station {calibration_manager.station_id}")
            return False
            
        print("✅ 모든 필수 데이터 파일이 준비되었습니다.")
        
        # 캘리브레이션 실행
        print(f"\n🔄 캘리브레이션 시작...")
        print("   - 중성자 보정 적용")
        print("   - 지점 토양수분 가중평균 계산")
        print("   - N0 매개변수 최적화")
        print("   - 성능 지표 계산")
        print("   - 결과 저장")
        
        result = calibration_manager.run_calibration(
            calibration_start=start_date,
            calibration_end=end_date,
            force_recalibration=True
        )
        
        print("\n✅ 캘리브레이션 완료!")
        self._print_calibration_summary(result)
        
        return True
        
    def _print_calibration_summary(self, result: Dict[str, Any]) -> None:
        """캘리브레이션 결과 요약 출력"""
        
        print(f"\n📊 캘리브레이션 결과:")
        print(f"   N0 = {result.get('N0_rdt', 0):.2f}")
        
        metrics = result.get('performance_metrics', {})
        if metrics:
            print(f"   R² = {metrics.get('R2', 0):.3f}")
            print(f"   RMSE = {metrics.get('RMSE', 0):.3f}")
            print(f"   MAE = {metrics.get('MAE', 0):.3f}")
            print(f"   데이터 개수 = {metrics.get('n_samples', 0)}")
            
        # 품질 평가
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 1)
        
        print(f"\n📈 품질 평가:")
        if r2 >= 0.8:
            print("   🟢 우수: R² ≥ 0.8")
        elif r2 >= 0.6:
            print("   🟡 양호: 0.6 ≤ R² < 0.8")
        else:
            print("   🔴 개선 필요: R² < 0.6")
            
        if rmse <= 0.05:
            print("   🟢 우수: RMSE ≤ 0.05")
        elif rmse <= 0.1:
            print("   🟡 양호: 0.05 < RMSE ≤ 0.1")
        else:
            print("   🔴 개선 필요: RMSE > 0.1")
            
        # 다음 단계 안내
        station_id = result.get('station_id', 'UNKNOWN')
        print(f"\n🎯 다음 단계:")
        print(f"   1. 토양수분 계산: python scripts/run_soil_moisture.py --station {station_id}")
        print(f"   2. 시각화: python scripts/run_visualization.py --station {station_id}")
        print(f"   3. 검증: python scripts/run_validation.py --station {station_id}")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP 캘리브레이션 실행")
    
    parser.add_argument("--station", "-s", required=True,
                       help="관측소 ID (예: HC, PC)")
    parser.add_argument("--start", help="캘리브레이션 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="캘리브레이션 종료일 (YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="기존 캘리브레이션 결과 무시하고 재실행")
    parser.add_argument("--status", action="store_true",
                       help="캘리브레이션 상태만 확인")
    
    args = parser.parse_args()
    
    # 시작/종료일 유효성 검사
    if (args.start and not args.end) or (args.end and not args.start):
        print("❌ 시작일과 종료일을 모두 지정해야 합니다.")
        return 1
        
    if args.start and args.end:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(args.start, '%Y-%m-%d')
            end_dt = datetime.strptime(args.end, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                print("❌ 시작일이 종료일보다 늦습니다.")
                return 1
                
            if (end_dt - start_dt).days < 3:
                print("⚠️  캘리브레이션 기간이 3일 미만입니다. 정확도가 떨어질 수 있습니다.")
                user_input = input("계속 진행하시겠습니까? (y/N): ")
                if user_input.lower() != 'y':
                    return 1
                    
        except ValueError:
            print("❌ 날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
            return 1
            
    # 캘리브레이션 실행
    runner = CalibrationRunner()
    success = runner.run_calibration(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        status_only=args.status
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    main()