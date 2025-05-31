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
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calibration.calibration_manager import CalibrationManager
from src.core.logger import setup_logger


class UniversalCalibrationRunner:
    """범용 캘리브레이션 실행 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("UniversalCalibrationRunner")
        
    def run_calibration(self, station_id: str, 
                       start_date: str = None, end_date: str = None,
                       auto_optimize: bool = False,
                       force: bool = False, status_only: bool = False) -> bool:
        """범용 캘리브레이션 실행"""
        
        print(f"🔬 CRNP Calibration - {station_id} Station")
        print("=" * 60)
        
        try:
            # CalibrationManager 초기화
            calibration_manager = CalibrationManager(station_id)
            
            if status_only:
                # 상태 확인만
                return self._show_calibration_status(calibration_manager)
            
            # 자동 최적화 옵션
            if auto_optimize:
                start_date, end_date = self._auto_optimize_period(station_id, start_date, end_date)
                
            # 기존 결과 확인
            if not force:
                existing_status = calibration_manager.get_calibration_status()
                if existing_status['calibration_available']:
                    print("📋 Existing calibration result found.")
                    
                    # 기간 확인
                    if start_date and end_date:
                        existing_period = existing_status.get('calibration_period', {})
                        existing_start = existing_period.get('start', '').split('T')[0]
                        existing_end = existing_period.get('end', '').split('T')[0]
                        
                        if existing_start == start_date and existing_end == end_date:
                            print("✅ Using existing calibration result (same period).")
                            self._print_calibration_summary(existing_status)
                            return True
                            
                    user_input = input("Overwrite existing result? (y/N): ")
                    if user_input.lower() != 'y':
                        print("Calibration cancelled.")
                        return False
                        
            # 데이터 품질 사전 확인
            data_quality = self._check_data_quality(station_id, start_date, end_date)
            if not data_quality['sufficient']:
                print("❌ Insufficient data quality for calibration")
                print("💡 Suggestions:")
                for suggestion in data_quality['suggestions']:
                    print(f"   - {suggestion}")
                return False
                
            # 캘리브레이션 실행
            print(f"\n🔄 Running calibration...")
            if start_date and end_date:
                print(f"   Period: {start_date} to {end_date}")
            else:
                print("   Period: Default calibration period")
                
            result = calibration_manager.run_calibration(
                calibration_start=start_date,
                calibration_end=end_date,
                force_recalibration=True
            )
            
            # 결과 분석 및 출력
            self._analyze_and_display_results(result, station_id, data_quality)
            
            return True
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            self.logger.error(f"Calibration failed for {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _auto_optimize_period(self, station_id: str, start_date: str, end_date: str) -> Tuple[str, str]:
        """자동 기간 최적화"""
        
        print("🔧 Auto-optimizing calibration period...")
        
        try:
            # 데이터 파일 경로
            fdr_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_FDR_input.xlsx"
            
            if not fdr_file.exists():
                print("   ⚠️  FDR data not found, using default period")
                return start_date or "2024-08-17", end_date or "2024-08-25"
                
            # FDR 데이터 로드
            fdr_data = pd.read_excel(fdr_file)
            fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
            
            # 일별 변동성 분석
            daily_variability = fdr_data.groupby(fdr_data['Date'].dt.date)['theta_v'].agg(['mean', 'std']).reset_index()
            daily_variability = daily_variability.dropna()
            
            # 변동성이 높은 기간 찾기
            high_var_threshold = daily_variability['std'].quantile(0.7)  # 상위 30%
            high_var_days = daily_variability[daily_variability['std'] >= high_var_threshold]
            
            if len(high_var_days) >= 7:  # 최소 7일
                # 연속된 기간 찾기
                dates = pd.to_datetime(high_var_days['Date'])
                dates_sorted = dates.sort_values()
                
                # 가장 긴 연속 기간 찾기
                best_start, best_end = self._find_longest_continuous_period(dates_sorted, min_days=7)
                
                if best_start and best_end:
                    opt_start = best_start.strftime('%Y-%m-%d')
                    opt_end = best_end.strftime('%Y-%m-%d')
                    
                    variability_info = daily_variability[
                        (daily_variability['Date'] >= best_start.date()) & 
                        (daily_variability['Date'] <= best_end.date())
                    ]
                    
                    print(f"   📊 Optimized period: {opt_start} to {opt_end}")
                    print(f"   📈 Average variability: {variability_info['std'].mean():.4f}")
                    
                    return opt_start, opt_end
                    
        except Exception as e:
            print(f"   ⚠️  Auto-optimization failed: {e}")
            
        # 기본값 반환
        default_start = start_date or "2024-08-17"
        default_end = end_date or "2024-08-25"
        print(f"   📅 Using default period: {default_start} to {default_end}")
        
        return default_start, default_end
        
    def _find_longest_continuous_period(self, dates: pd.Series, min_days: int = 7) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """가장 긴 연속 기간 찾기"""
        
        if len(dates) == 0:
            return None, None
            
        dates_sorted = dates.sort_values().reset_index(drop=True)
        
        best_start = None
        best_end = None
        best_length = 0
        
        current_start = dates_sorted[0]
        current_end = dates_sorted[0]
        
        for i in range(1, len(dates_sorted)):
            if (dates_sorted[i] - dates_sorted[i-1]).days <= 2:  # 최대 2일 간격
                current_end = dates_sorted[i]
            else:
                # 연속 기간 종료
                current_length = (current_end - current_start).days + 1
                if current_length >= min_days and current_length > best_length:
                    best_start = current_start
                    best_end = current_end
                    best_length = current_length
                    
                current_start = dates_sorted[i]
                current_end = dates_sorted[i]
                
        # 마지막 기간 확인
        current_length = (current_end - current_start).days + 1
        if current_length >= min_days and current_length > best_length:
            best_start = current_start
            best_end = current_end
            
        return best_start, best_end
        
    def _check_data_quality(self, station_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """데이터 품질 사전 확인"""
        
        quality_check = {
            'sufficient': False,
            'issues': [],
            'suggestions': [],
            'variability_score': 0.0
        }
        
        try:
            # 데이터 파일 확인
            fdr_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_FDR_input.xlsx"
            crnp_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_CRNP_input.xlsx"
            
            if not fdr_file.exists():
                quality_check['issues'].append("FDR data file not found")
                quality_check['suggestions'].append("Run preprocessing first")
                return quality_check
                
            if not crnp_file.exists():
                quality_check['issues'].append("CRNP data file not found")
                quality_check['suggestions'].append("Run preprocessing first")
                return quality_check
                
            # 데이터 로드
            fdr_data = pd.read_excel(fdr_file)
            crnp_data = pd.read_excel(crnp_file)
            
            # 기간 필터링
            if start_date and end_date:
                fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
                crnp_data['timestamp'] = pd.to_datetime(crnp_data['timestamp'])
                
                fdr_period = fdr_data[(fdr_data['Date'] >= start_date) & (fdr_data['Date'] <= end_date)]
                crnp_period = crnp_data[(crnp_data['timestamp'] >= start_date) & (crnp_data['timestamp'] <= end_date)]
            else:
                fdr_period = fdr_data
                crnp_period = crnp_data
                
            # 기본 데이터 확인
            if len(fdr_period) == 0:
                quality_check['issues'].append("No FDR data in specified period")
                quality_check['suggestions'].append("Check date range or extend period")
                return quality_check
                
            if len(crnp_period) == 0:
                quality_check['issues'].append("No CRNP data in specified period")
                quality_check['suggestions'].append("Check date range or extend period")
                return quality_check
                
            # FDR 변동성 확인
            if 'theta_v' in fdr_period.columns:
                theta_std = fdr_period['theta_v'].std()
                theta_range = fdr_period['theta_v'].max() - fdr_period['theta_v'].min()
                
                quality_check['variability_score'] = theta_std
                
                print(f"📊 Data quality assessment:")
                print(f"   FDR records: {len(fdr_period)}")
                print(f"   CRNP records: {len(crnp_period)}")
                print(f"   FDR variability: std={theta_std:.4f}, range={theta_range:.4f}")
                
                # 변동성 평가
                if theta_std < 0.005:
                    quality_check['issues'].append("Very low FDR variability")
                    quality_check['suggestions'].append("Extend calibration period")
                    quality_check['suggestions'].append("Check sensor functioning")
                elif theta_std < 0.010:
                    quality_check['issues'].append("Low FDR variability")
                    quality_check['suggestions'].append("Consider extending period")
                else:
                    print("   ✅ Good FDR variability for calibration")
                    
            # 중성자 카운트 확인
            if 'N_counts' in crnp_period.columns:
                neutron_std = crnp_period['N_counts'].std()
                neutron_valid = crnp_period['N_counts'].notna().sum()
                
                print(f"   Neutron variability: std={neutron_std:.1f}")
                print(f"   Valid neutron counts: {neutron_valid}/{len(crnp_period)}")
                
                if neutron_valid < len(crnp_period) * 0.8:
                    quality_check['issues'].append("High missing neutron data")
                    quality_check['suggestions'].append("Check CRNP data quality")
                    
            # 일별 매칭 가능성 확인
            fdr_dates = set(pd.to_datetime(fdr_period['Date']).dt.date)
            crnp_dates = set(pd.to_datetime(crnp_period['timestamp']).dt.date)
            common_dates = fdr_dates.intersection(crnp_dates)
            
            print(f"   Matchable days: {len(common_dates)}")
            
            if len(common_dates) < 5:
                quality_check['issues'].append("Insufficient matchable days")
                quality_check['suggestions'].append("Extend period or check data gaps")
            elif len(common_dates) < 7:
                quality_check['issues'].append("Limited matchable days")
                quality_check['suggestions'].append("Consider extending period")
            else:
                print("   ✅ Sufficient matchable days")
                
            # 최종 평가
            critical_issues = len([issue for issue in quality_check['issues'] if 'No' in issue or 'Insufficient' in issue])
            
            if critical_issues == 0 and len(common_dates) >= 5:
                quality_check['sufficient'] = True
                print("   ✅ Data quality sufficient for calibration")
            else:
                print("   ⚠️  Data quality issues detected")
                
        except Exception as e:
            quality_check['issues'].append(f"Data quality check failed: {e}")
            quality_check['suggestions'].append("Check data file integrity")
            
        return quality_check
        
    def _show_calibration_status(self, calibration_manager: CalibrationManager) -> bool:
        """캘리브레이션 상태 표시"""
        
        print("📊 Calibration Status Check")
        print("-" * 40)
        
        status = calibration_manager.get_calibration_status()
        
        print(f"Station: {status['station_id']}")
        
        # 캘리브레이션 가용성
        if status['calibration_available']:
            print("✅ Calibration result available")
            
            # 캘리브레이션 정보
            if status.get('N0_rdt'):
                print(f"   N0 value: {status['N0_rdt']:.2f}")
                
            if status.get('calibration_period'):
                period = status['calibration_period']
                print(f"   Period: {period['start']} to {period['end']}")
                
            if status.get('calibration_date'):
                print(f"   Generated: {status['calibration_date']}")
                
            # 성능 지표
            metrics = status.get('performance_metrics', {})
            if metrics:
                print(f"\n📈 Performance Metrics:")
                print(f"   R² = {metrics.get('R2', 0):.3f}")
                print(f"   RMSE = {metrics.get('RMSE', 0):.3f}")
                print(f"   Correlation = {metrics.get('Correlation', 0):.3f}")
                print(f"   Sample size = {metrics.get('n_samples', 0)}")
                
        else:
            print("❌ No calibration result found")
            
        # 데이터 가용성
        print(f"\n📁 Data Files:")
        data_availability = status.get('data_availability', {})
        
        for filename, file_info in data_availability.items():
            status_icon = "✅" if file_info['exists'] else "❌"
            size_info = f"({file_info['size_mb']} MB)" if file_info['exists'] else ""
            print(f"   {filename}: {status_icon} {size_info}")
            
        # 다음 단계 안내
        if not status['calibration_available']:
            print(f"\n💡 Next Steps:")
            print(f"   1. Check preprocessing: python scripts/run_preprocessing.py --station {status['station_id']} --check-only")
            print(f"   2. Run calibration: python scripts/run_calibration_universal.py --station {status['station_id']}")
            print(f"   3. Auto-optimize: python scripts/run_calibration_universal.py --station {status['station_id']} --auto-optimize")
            
        return status['calibration_available']
        
    def _analyze_and_display_results(self, result: Dict[str, Any], station_id: str, data_quality: Dict) -> None:
        """결과 분석 및 표시"""
        
        print(f"\n📊 Calibration Results:")
        
        # 기본 결과
        N0 = result.get('N0_rdt', 0)
        print(f"   N0 = {N0:.2f}")
        
        # 성능 지표
        metrics = result.get('performance_metrics', {})
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        correlation = metrics.get('Correlation', 0)
        method_used = metrics.get('method_used', 'unknown')
        
        print(f"   R² = {r2:.4f} (method: {method_used})")
        print(f"   RMSE = {rmse:.4f}")
        print(f"   Correlation = {correlation:.4f}")
        print(f"   Sample size = {metrics.get('n_samples', 0)}")
        
        # 품질 평가
        print(f"\n📈 Quality Assessment:")
        
        # R² 평가 (방법에 따라 다른 기준 적용)
        if method_used == "correlation_squared" or method_used == "relative_error_based":
            # 변동성이 낮은 경우 더 관대한 기준
            if r2 >= 0.4:
                print("   🟢 R² - Good for low variability data (≥ 0.4)")
            elif r2 >= 0.2:
                print("   🟡 R² - Fair for low variability data (≥ 0.2)")
            else:
                print("   🔴 R² - Poor (< 0.2)")
        else:
            # 전통적인 기준
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
        print(f"\n💡 Recommendations:")
        
        variability_score = data_quality.get('variability_score', 0)
        if variability_score < 0.01:
            print("   - Low soil moisture variability detected")
            print("   - Consider using correlation-based performance evaluation")
            print("   - Extend calibration period if possible")
            
        if correlation >= 0.5 and rmse <= 0.05:
            print("   - Calibration quality is acceptable for operational use")
            
        if r2 < 0.3 and correlation >= 0.5:
            print("   - Strong correlation despite low R² suggests systematic bias")
            print("   - Consider reviewing soil parameters or sensor calibration")
            
        # 생성된 파일 안내
        print(f"\n📁 Generated Files:")
        print(f"   📊 Diagnostics: {station_id}_calibration_diagnostics.png")
        print(f"   📈 Comparison: {station_id}_calibration_comparison.png")
        print(f"   📋 Data: {station_id}_calibration_debug_data.xlsx")
        print(f"   ⚙️  Results: {station_id}_calibration_result.json")
        
        # 다음 단계
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review generated plots and diagnostics")
        print(f"   2. Run soil moisture calculation:")
        print(f"      python scripts/run_soil_moisture.py --station {station_id}")
        print(f"   3. Validate results with independent data if available")
        
    def _print_calibration_summary(self, result: Dict[str, Any]) -> None:
        """기존 캘리브레이션 결과 요약 출력"""
        
        print(f"\n📊 Existing Calibration Summary:")
        print(f"   N0 = {result.get('N0_rdt', 0):.2f}")
        
        metrics = result.get('performance_metrics', {})
        if metrics:
            print(f"   R² = {metrics.get('R2', 0):.3f}")
            print(f"   RMSE = {metrics.get('RMSE', 0):.3f}")
            print(f"   Correlation = {metrics.get('Correlation', 0):.3f}")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="Universal CRNP Calibration")
    
    parser.add_argument("--station", "-s", required=True, help="Station ID (HC, PC, etc.)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--auto-optimize", "-auto", action="store_true", 
                       help="Automatically optimize calibration period")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force recalibration (ignore existing results)")
    parser.add_argument("--status", action="store_true", 
                       help="Check calibration status only")
    
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
                
            if (end_dt - start_dt).days < 3:
                print("⚠️  Calibration period is very short (< 3 days)")
                print("   Consider using --auto-optimize for better period selection")
                user_input = input("Continue anyway? (y/N): ")
                if user_input.lower() != 'y':
                    return 1
                    
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return 1
            
    # 캘리브레이션 실행
    runner = UniversalCalibrationRunner()
    success = runner.run_calibration(
        station_id=args.station,
        start_date=args.start,
        end_date=args.end,
        auto_optimize=args.auto_optimize,
        force=args.force,
        status_only=args.status
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())