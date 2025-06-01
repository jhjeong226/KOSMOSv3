# scripts/run_visualization.py

"""
CRNP 시각화 생성 실행 스크립트

사용법:
    python scripts/run_visualization.py --station HC
    python scripts/run_visualization.py --station HC --open
    python scripts/run_visualization.py --station HC --no-validation
    python scripts/run_visualization.py --station HC --status
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.visualization_manager import VisualizationManager
from src.core.logger import setup_logger


class VisualizationRunner:
    """시각화 실행 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("VisualizationRunner")
        
    def run_visualization(self, station_id: str, 
                         include_validation: bool = True,
                         open_report: bool = False,
                         status_only: bool = False) -> bool:
        """시각화 실행"""
        
        print(f"🎨 CRNP Visualization Generation - {station_id} Station")
        print("=" * 70)
        
        try:
            # VisualizationManager 초기화
            viz_manager = VisualizationManager(station_id)
            
            if status_only:
                # 상태 확인만
                return self._show_visualization_status(viz_manager)
            
            # 1. 사전 데이터 확인
            print("🔍 1단계: 데이터 가용성 확인")
            data_status = self._check_data_availability(station_id)
            
            if not data_status['sufficient']:
                print("❌ Insufficient data for visualization")
                for issue in data_status['issues']:
                    print(f"   ⚠️  {issue}")
                return False
                
            print("   ✅ Required data available")
            
            available_categories = data_status['available_categories']
            print(f"   📊 Available categories: {', '.join(available_categories)}")
            
            # validation 데이터가 없으면 자동으로 제외
            if 'validation' not in available_categories:
                include_validation = False
                print("   ℹ️  Validation data not available - skipping validation plots")
                
            # 2. 시각화 생성
            print(f"\n🔄 2단계: 시각화 생성")
            if include_validation:
                print("   포함: 중성자 분석, 토양수분, 검증")
            else:
                print("   포함: 중성자 분석, 토양수분")
                
            result = viz_manager.generate_all_plots(include_validation=include_validation)
            
            # 3. 결과 분석 및 출력
            self._analyze_and_display_results(result, station_id)
            
            # 4. HTML 리포트 열기 (옵션)
            if open_report and result.get('html_report'):
                print(f"\n🌐 3단계: HTML 리포트 열기")
                success = viz_manager.open_html_report()
                if success:
                    print("   ✅ HTML 리포트가 브라우저에서 열렸습니다")
                else:
                    print("   ⚠️  브라우저에서 열기 실패")
                    print(f"   📁 수동으로 열기: {result['html_report']}")
            elif result.get('html_report'):
                print(f"\n📄 HTML 리포트 위치: {os.path.relpath(result['html_report'], self.project_root)}")
                print("   브라우저에서 열려면 --open 옵션을 사용하세요")
                
            return True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            self.logger.error(f"Visualization failed for {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _check_data_availability(self, station_id: str) -> Dict[str, Any]:
        """데이터 가용성 확인"""
        
        status = {
            'sufficient': False,
            'issues': [],
            'available_categories': []
        }
        
        base_dir = self.project_root / f"data/output/{station_id}"
        
        # 1. 전처리 데이터 확인
        preprocessed_dir = base_dir / "preprocessed"
        crnp_file = preprocessed_dir / f"{station_id}_CRNP_input.xlsx"
        fdr_file = preprocessed_dir / f"{station_id}_FDR_input.xlsx"
        
        if not crnp_file.exists():
            status['issues'].append("CRNP preprocessed data not found")
        else:
            print(f"   ✅ CRNP 전처리 데이터: {os.path.relpath(crnp_file, self.project_root)}")
            
        if not fdr_file.exists():
            status['issues'].append("FDR preprocessed data not found")
        else:
            print(f"   ✅ FDR 전처리 데이터: {os.path.relpath(fdr_file, self.project_root)}")
            
        # 2. 캘리브레이션 결과 확인 (중성자 플롯용)
        calibration_dir = base_dir / "calibration"
        cal_result_file = calibration_dir / f"{station_id}_calibration_result.json"
        cal_debug_file = list(calibration_dir.glob("*debug_data.xlsx"))
        
        neutron_data_available = crnp_file.exists() or (cal_result_file.exists() and cal_debug_file)
        
        if neutron_data_available:
            status['available_categories'].append('neutron')
            if cal_result_file.exists():
                print(f"   ✅ 캘리브레이션 결과: {os.path.relpath(cal_result_file, self.project_root)}")
            else:
                print(f"   ⚠️  캘리브레이션 결과 없음 - 기본 중성자 플롯만 가능")
        else:
            status['issues'].append("No neutron data available")
            
        # 3. 토양수분 계산 결과 확인
        sm_dir = base_dir / "soil_moisture"
        sm_file = sm_dir / f"{station_id}_soil_moisture.xlsx"
        
        if sm_file.exists():
            status['available_categories'].append('soil_moisture')
            print(f"   ✅ 토양수분 계산 결과: {os.path.relpath(sm_file, self.project_root)}")
        else:
            status['issues'].append("Soil moisture calculation results not found")
            
        # 4. 검증 데이터 확인 (토양수분 + FDR)
        if sm_file.exists() and fdr_file.exists():
            status['available_categories'].append('validation')
            print(f"   ✅ 검증 데이터 준비됨")
        else:
            print(f"   ℹ️  검증 데이터 불완전 - 검증 플롯 제외")
            
        # 5. 최소 요구사항 확인
        if len(status['available_categories']) >= 1:
            status['sufficient'] = True
        else:
            status['issues'].append("No visualization data categories available")
            
        return status
        
    def _show_visualization_status(self, viz_manager: VisualizationManager) -> bool:
        """시각화 상태 표시"""
        
        print("📊 Visualization Status")
        print("-" * 40)
        
        status = viz_manager.get_visualization_status()
        
        print(f"Station: {status['station_id']}")
        
        # 시각화 가용성
        if status['plots_available']:
            print("✅ Visualization plots available")
            
            if status.get('plots_count'):
                print(f"   Total plots: {status['plots_count']}")
                
            if status.get('generation_date'):
                print(f"   Generated: {status['generation_date']}")
                
            # 카테고리별 플롯 수
            plots_by_category = status.get('plots_by_category', {})
            if plots_by_category:
                print(f"\n📈 Plots by Category:")
                for category, plots in plots_by_category.items():
                    plot_count = len(plots) if isinstance(plots, dict) else 0
                    print(f"   {category.title()}: {plot_count} plots")
                    
            # HTML 리포트 상태
            if status.get('html_report_available'):
                print(f"\n🌐 HTML Report: ✅ Available")
                html_path = status.get('html_report_path', '')
                if html_path:
                    rel_path = os.path.relpath(html_path, self.project_root)
                    print(f"   Path: {rel_path}")
            else:
                print(f"\n🌐 HTML Report: ❌ Not available")
                
        else:
            print("❌ No visualization plots found")
            
        # 다음 단계 안내
        if not status['plots_available']:
            print(f"\n💡 Next Steps:")
            print(f"   1. Ensure data is processed:")
            print(f"      python scripts/run_crnp_pipeline.py --station {status['station_id']} --steps preprocessing calibration soil_moisture")
            print(f"   2. Generate visualizations:")
            print(f"      python scripts/run_visualization.py --station {status['station_id']}")
        else:
            print(f"\n🎯 Available Actions:")
            print(f"   1. Regenerate plots:")
            print(f"      python scripts/run_visualization.py --station {status['station_id']}")
            if status.get('html_report_available'):
                print(f"   2. Open HTML report:")
                print(f"      python scripts/run_visualization.py --station {status['station_id']} --open")
                
        return status['plots_available']
        
    def _analyze_and_display_results(self, result: Dict[str, Any], station_id: str) -> None:
        """결과 분석 및 표시"""
        
        print(f"\n📊 Visualization Results:")
        
        total_plots = result.get('total_plots', 0)
        print(f"   Total plots generated: {total_plots}")
        
        # 카테고리별 결과
        plots_generated = result.get('plots_generated', {})
        
        if plots_generated:
            print(f"\n📈 Generated Plots by Category:")
            
            for category, plots in plots_generated.items():
                if plots:
                    print(f"   {category.title()}: {len(plots)} plots")
                    
                    # 주요 플롯들 나열
                    plot_names = list(plots.keys())
                    if len(plot_names) <= 3:
                        for plot_name in plot_names:
                            print(f"      • {plot_name.replace('_', ' ').title()}")
                    else:
                        for plot_name in plot_names[:2]:
                            print(f"      • {plot_name.replace('_', ' ').title()}")
                        print(f"      • ... and {len(plot_names)-2} more")
                else:
                    print(f"   {category.title()}: ⚠️ No plots generated")
                    
        # HTML 리포트
        html_report = result.get('html_report')
        if html_report:
            rel_path = os.path.relpath(html_report, self.project_root)
            print(f"\n🌐 HTML Report:")
            print(f"   Location: {rel_path}")
            print(f"   Size: {self._get_file_size(html_report)}")
            
        # 생성된 파일들 위치
        viz_dir = self.project_root / f"data/output/{station_id}/visualization"
        if viz_dir.exists():
            png_files = list(viz_dir.glob("*.png"))
            print(f"\n📁 Output Directory:")
            print(f"   Location: {os.path.relpath(viz_dir, self.project_root)}")
            print(f"   PNG files: {len(png_files)}")
            
        # 품질 평가
        print(f"\n📈 Quality Assessment:")
        
        if total_plots >= 10:
            print("   🟢 Comprehensive visualization coverage")
        elif total_plots >= 5:
            print("   🟡 Good visualization coverage")
        elif total_plots >= 1:
            print("   🟠 Basic visualization coverage")
        else:
            print("   🔴 Insufficient visualization")
            
        # 데이터 카테고리별 평가
        required_categories = ['neutron', 'soil_moisture']
        available_categories = [cat for cat in required_categories if plots_generated.get(cat)]
        
        coverage = len(available_categories) / len(required_categories) * 100
        print(f"   Data coverage: {coverage:.0f}% ({len(available_categories)}/{len(required_categories)} categories)")
        
        if 'validation' in plots_generated and plots_generated['validation']:
            print("   ✅ Validation plots included")
        else:
            print("   ℹ️  Validation plots not available")
            
        # 다음 단계 안내
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review HTML report: python scripts/run_visualization.py --station {station_id} --open")
        print(f"   2. Share results with stakeholders")
        
        if 'validation' not in plots_generated or not plots_generated['validation']:
            print(f"   3. Generate validation plots (if field data available):")
            print(f"      python scripts/run_validation.py --station {station_id}")
            
    def _get_file_size(self, file_path: str) -> str:
        """파일 크기 반환"""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except:
            return "Unknown"


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP Visualization Generation")
    
    parser.add_argument("--station", "-s", required=True, help="Station ID (HC, PC, etc.)")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Skip validation plots")
    parser.add_argument("--open", "-o", action="store_true",
                       help="Open HTML report in browser after generation")
    parser.add_argument("--status", action="store_true", 
                       help="Check visualization status only")
    
    args = parser.parse_args()
    
    # 시각화 실행
    runner = VisualizationRunner()
    success = runner.run_visualization(
        station_id=args.station,
        include_validation=not args.no_validation,
        open_report=args.open,
        status_only=args.status
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())