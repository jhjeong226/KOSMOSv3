# scripts/run_calibration_enhanced.py

"""
향상된 CRNP 캘리브레이션 실행 스크립트
PC 사이트 특별 처리 포함

사용법:
    python scripts/run_calibration_enhanced.py --station PC
    python scripts/run_calibration_enhanced.py --station PC --bulk-density 1.4
    python scripts/run_calibration_enhanced.py --station PC --extend-period
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calibration.calibration_manager import CalibrationManager
from src.core.logger import setup_logger


class EnhancedCalibrationRunner:
    """향상된 캘리브레이션 실행 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("EnhancedCalibrationRunner")
        
    def run_enhanced_calibration(self, station_id: str, 
                                start_date: str = None, end_date: str = None,
                                bulk_density: float = None,
                                extend_period: bool = False,
                                force: bool = False) -> bool:
        """향상된 캘리브레이션 실행"""
        
        print(f"🔬 Enhanced CRNP Calibration - {station_id} Station")
        print("=" * 60)
        
        try:
            # 1. PC 사이트 특별 처리
            if station_id == "PC":
                start_date, end_date, bulk_density = self._optimize_pc_settings(
                    start_date, end_date, bulk_density, extend_period
                )
                
            # 2. CalibrationManager 초기화
            calibration_manager = CalibrationManager(station_id)
            
            # 3. 벌크밀도 수정 (필요시)
            if bulk_density:
                print(f"🔧 Updating bulk density to {bulk_density}")
                calibration_manager.update_calibration_config({
                    'station_config': {
                        'soil_properties': {'bulk_density': bulk_density}
                    }
                })
                
            # 4. 캘리브레이션 실행
            print(f"\n🔄 Running calibration...")
            print(f"   Period: {start_date} to {end_date}")
            print(f"   Bulk density: {bulk_density or 'default'}")
            
            result = calibration_manager.run_calibration(
                calibration_start=start_date,
                calibration_end=end_date,
                force_recalibration=True
            )
            
            # 5. 결과 분석
            self._analyze_calibration_results(result, station_id)
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced calibration failed: {e}")
            self.logger.error(f"Enhanced calibration failed for {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _optimize_pc_settings(self, start_date: str, end_date: str, 
                             bulk_density: float, extend_period: bool) -> tuple:
        """PC 사이트 최적 설정"""
        
        print("🔧 Optimizing PC site settings...")
        
        # 1. 기간 확장 (더 많은 변동성 확보)
        if extend_period or not (start_date and end_date):
            print("   📅 Extending calibration period for better variability")
            start_date = "2024-08-15"  # 2일 더 일찍
            end_date = "2024-08-27"    # 2일 더 늦게
            print(f"   New period: {start_date} to {end_date}")
            
        # 2. 벌크밀도 조정
        if not bulk_density:
            # PC 사이트는 산지 토양이므로 더 높은 벌크밀도 사용
            bulk_density = 1.35  # 기본 1.2보다 높게
            print(f"   🏔️  Adjusted bulk density for mountain soil: {bulk_density}")
            
        # 3. 데이터 확인
        self._check_pc_data_availability(start_date, end_date)
        
        return start_date, end_date, bulk_density
        
    def _check_pc_data_availability(self, start_date: str, end_date: str) -> None:
        """PC 데이터 가용성 확인"""
        
        try:
            # FDR 데이터 확인
            fdr_file = self.project_root / "data/output/PC/preprocessed/PC_FDR_input.xlsx"
            crnp_file = self.project_root / "data/output/PC/preprocessed/PC_CRNP_input.xlsx"
            
            if fdr_file.exists() and crnp_file.exists():
                fdr_data = pd.read_excel(fdr_file)
                crnp_data = pd.read_excel(crnp_file)
                
                # 기간 내 데이터 확인
                fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
                crnp_data['timestamp'] = pd.to_datetime(crnp_data['timestamp'])
                
                fdr_period = fdr_data[
                    (fdr_data['Date'] >= start_date) & 
                    (fdr_data['Date'] <= end_date)
                ]
                crnp_period = crnp_data[
                    (crnp_data['timestamp'] >= start_date) & 
                    (crnp_data['timestamp'] <= end_date)
                ]
                
                print(f"   📊 Data availability check:")
                print(f"      FDR records in period: {len(fdr_period)}")
                print(f"      CRNP records in period: {len(crnp_period)}")
                
                # FDR 변동성 확인
                if len(fdr_period) > 0 and 'theta_v' in fdr_period.columns:
                    theta_std = fdr_period['theta_v'].std()
                    theta_range = fdr_period['theta_v'].max() - fdr_period['theta_v'].min()
                    print(f"      FDR variability: std={theta_std:.4f}, range={theta_range:.4f}")
                    
                    if theta_std < 0.01:
                        print("      ⚠️  Low FDR variability detected!")
                        
        except Exception as e:
            print(f"   ⚠️  Could not check data availability: {e}")
            
    def _analyze_calibration_results(self, result: Dict[str, Any], station_id: str) -> None:
        """캘리브레이션 결과 분석"""
        
        print(f"\n📊 Calibration Results Analysis:")
        
        # 기본 결과
        N0 = result.get('N0_rdt', 0)
        print(f"   N0 = {N0:.2f}")
        
        # 성능 지표
        metrics = result.get('performance_metrics', {})
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        correlation = metrics.get('Correlation', 0)
        obs_std = metrics.get('obs_std', 0)
        pred_std = metrics.get('pred_std', 0)
        
        print(f"   R² = {r2:.4f}")
        print(f"   RMSE = {rmse:.4f}")
        print(f"   Correlation = {correlation:.4f}")
        print(f"   FDR std = {obs_std:.4f}")
        print(f"   CRNP std = {pred_std:.4f}")
        
        # 품질 평가
        print(f"\n📈 Quality Assessment:")
        
        # R² 평가
        if r2 >= 0.7:
            print("   🟢 R² - Excellent (≥ 0.7)")
        elif r2 >= 0.5:
            print("   🟡 R² - Good (≥ 0.5)")
        elif r2 >= 0.3:
            print("   🟠 R² - Fair (≥ 0.3)")
        else:
            print("   🔴 R² - Poor (< 0.3)")
            
        # RMSE 평가
        if rmse <= 0.03:
            print("   🟢 RMSE - Excellent (≤ 0.03)")
        elif rmse <= 0.05:
            print("   🟡 RMSE - Good (≤ 0.05)")
        elif rmse <= 0.08:
            print("   🟠 RMSE - Fair (≤ 0.08)")
        else:
            print("   🔴 RMSE - Poor (> 0.08)")
            
        # 상관계수 평가
        if correlation >= 0.8:
            print("   🟢 Correlation - Strong (≥ 0.8)")
        elif correlation >= 0.6:
            print("   🟡 Correlation - Moderate (≥ 0.6)")
        elif correlation >= 0.4:
            print("   🟠 Correlation - Weak (≥ 0.4)")
        else:
            print("   🔴 Correlation - Very weak (< 0.4)")
            
        # 개선 제안
        print(f"\n💡 Improvement Suggestions:")
        
        if obs_std < 0.01:
            print("   - Consider extending calibration period for more variability")
            print("   - Check if soil moisture sensors are working properly")
            
        if correlation < 0.5:
            print("   - Review bulk density and soil parameters")
            print("   - Check neutron detector stability")
            
        if rmse > 0.08:
            print("   - Consider spatial weighting adjustment")
            print("   - Review FDR sensor calibration")
            
        # 생성된 파일 안내
        output_dir = self.project_root / "data/output" / station_id / "calibration"
        print(f"\n📁 Generated Files:")
        print(f"   📊 Diagnostics: {station_id}_calibration_diagnostics.png")
        print(f"   📈 Comparison: {station_id}_calibration_comparison.png")
        print(f"   📋 Data: {station_id}_calibration_debug_data.xlsx")
        print(f"   ⚙️  Results: {station_id}_calibration_result.json")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="Enhanced CRNP Calibration")
    
    parser.add_argument("--station", "-s", required=True, help="Station ID (PC, HC)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bulk-density", "-bd", type=float, help="Bulk density (kg/m³)")
    parser.add_argument("--extend-period", "-ep", action="store_true", help="Extend calibration period")
    parser.add_argument("--force", "-f", action="store_true", help="Force recalibration")
    
    args = parser.parse_args()
    
    # Enhanced calibration runner
    runner = EnhancedCalibrationRunner()
    success = runner.run_enhanced_calibration(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        bulk_density=args.bulk_density,
        extend_period=args.extend_period,
        force=args.force
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())