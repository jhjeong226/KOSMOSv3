# scripts/run_visualization.py

"""
CRNP 간단한 시각화 실행 스크립트

사용법:
    python scripts/run_visualization.py --station HC
    python scripts/run_visualization.py --station HC --check-only
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.simple_plotter import create_simple_visualization


class SimpleVisualizationRunner:
    """간단한 시각화 실행 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        
    def run_visualization(self, station_id: str, check_only: bool = False) -> bool:
        """시각화 실행"""
        
        print(f"🎨 CRNP Simple Visualization - {station_id} Station")
        print("=" * 60)
        
        try:
            # 1. 데이터 가용성 확인
            print("🔍 1단계: 데이터 가용성 확인")
            data_status = self._check_data_availability(station_id)
            
            if not data_status['sufficient']:
                print("❌ Insufficient data for visualization")
                for issue in data_status['issues']:
                    print(f"   ⚠️  {issue}")
                return False
            
            print("   ✅ Required data available")
            print(f"   📊 Available data: {', '.join(data_status['available_data'])}")
            
            if check_only:
                return True
            
            # 2. 시각화 생성
            print("\n🔄 2단계: 시각화 생성")
            output_dir = f"data/output/{station_id}/visualization"
            
            plot_files = create_simple_visualization(station_id, output_dir)
            
            # 3. 결과 요약
            self._print_results_summary(plot_files, output_dir)
            
            return len(plot_files) > 0
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_data_availability(self, station_id: str) -> dict:
        """데이터 가용성 확인"""
        
        status = {
            'sufficient': False,
            'issues': [],
            'available_data': []
        }
        
        # 필수 데이터 파일들
        data_files = {
            'soil_moisture': f"data/output/{station_id}/soil_moisture/{station_id}_soil_moisture.xlsx",
            'validation': f"data/output/{station_id}/validation/{station_id}_validation_data.xlsx"
        }
        
        available_count = 0
        
        for data_type, file_path in data_files.items():
            if Path(file_path).exists():
                status['available_data'].append(data_type)
                available_count += 1
                print(f"   ✅ {data_type.title()} data: {file_path}")
            else:
                print(f"   ❌ {data_type.title()} data: {file_path}")
                status['issues'].append(f"{data_type.title()} data not found")
        
        # 토양수분 데이터는 필수
        if 'soil_moisture' in status['available_data']:
            status['sufficient'] = True
        else:
            status['issues'].append("Soil moisture data is required for visualization")
        
        return status
    
    def _print_results_summary(self, plot_files: dict, output_dir: str):
        """결과 요약 출력"""
        
        print(f"\n📊 Visualization Results:")
        print(f"   Total plots generated: {len(plot_files)}")
        
        if plot_files:
            print(f"\n📈 Generated Plots:")
            plot_descriptions = {
                'neutron_comparison': 'Raw vs Corrected Neutron Counts',
                'correction_factors': 'Neutron Correction Factors',
                'vwc_timeseries': 'Volumetric Water Content',
                'sm_timeseries': 'Soil Moisture Comparison (CRNP vs Field)',
                'sm_scatter': 'Soil Moisture Scatter Plot'
            }
            
            for plot_type, file_path in plot_files.items():
                description = plot_descriptions.get(plot_type, plot_type.replace('_', ' ').title())
                print(f"   ✅ {description}")
        
        print(f"\n📁 Output Directory:")
        print(f"   Location: {output_dir}")
        
        # PNG 파일 개수 확인
        output_path = Path(output_dir)
        if output_path.exists():
            png_files = list(output_path.glob("*.png"))
            print(f"   PNG files: {len(png_files)}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review generated plots in: {output_dir}")
        print(f"   2. Share results with stakeholders")
        if 'sm_scatter' not in plot_files:
            print(f"   3. For validation plots, ensure field data is available")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP Simple Visualization")
    
    parser.add_argument("--station", "-s", required=True, help="Station ID (HC, PC, etc.)")
    parser.add_argument("--check-only", "-c", action="store_true", 
                       help="Check data availability only")
    
    args = parser.parse_args()
    
    try:
        runner = SimpleVisualizationRunner()
        success = runner.run_visualization(
            station_id=args.station,
            check_only=args.check_only
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Visualization script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())