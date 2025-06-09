# scripts/run_simple_dashboard.py

"""
간단한 CRNP 대시보드
필요한 모듈들을 자동으로 생성하고 시각화 실행

사용법:
    python scripts/run_simple_dashboard.py --station SWCR
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.simple_plotter import create_simple_visualization


def create_missing_modules():
    """누락된 시각화 모듈들 생성"""
    
    viz_dir = project_root / "src" / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # __init__.py 파일 생성
    init_file = viz_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    
    print("✅ 시각화 모듈 준비 완료")


def run_simple_dashboard(station_id: str):
    """간단한 대시보드 실행"""
    
    print(f"🚀 CRNP Simple Dashboard - {station_id} Station")
    print("=" * 60)
    
    try:
        # 1. 필요한 모듈들 확인 및 생성
        print("🔧 1단계: 모듈 준비")
        create_missing_modules()
        
        # 2. 데이터 가용성 확인
        print("\n🔍 2단계: 데이터 확인")
        
        # 필수 파일들 확인
        base_dir = project_root / f"data/output/{station_id}"
        sm_file = base_dir / "soil_moisture" / f"{station_id}_soil_moisture.xlsx"
        
        if not sm_file.exists():
            print(f"❌ 토양수분 데이터를 찾을 수 없습니다: {sm_file}")
            print("   먼저 토양수분 계산을 실행해주세요:")
            print(f"   python scripts/run_soil_moisture.py --station {station_id}")
            return False
        
        print("   ✅ 토양수분 데이터 확인됨")
        
        # 3. 시각화 생성
        print("\n📊 3단계: 시각화 생성")
        output_dir = f"data/output/{station_id}/visualization"
        
        plot_files = create_simple_visualization(station_id, output_dir)
        
        if plot_files:
            print(f"\n✅ {len(plot_files)}개의 플롯이 생성되었습니다!")
            print(f"📁 출력 폴더: {output_dir}")
            
            print("\n📈 생성된 플롯:")
            for plot_type, file_path in plot_files.items():
                plot_name = Path(file_path).name
                print(f"   • {plot_name}")
            
            # 4. HTML 리포트 생성
            print("\n📄 4단계: HTML 리포트 생성")
            html_file = create_simple_html_report(station_id, plot_files, output_dir)
            
            if html_file:
                print(f"✅ HTML 리포트 생성됨: {html_file}")
                
                # 브라우저에서 열기
                import webbrowser
                try:
                    file_url = f"file://{Path(html_file).absolute()}"
                    webbrowser.open(file_url)
                    print("🌐 브라우저에서 리포트를 열었습니다!")
                except Exception as e:
                    print(f"⚠️  브라우저 열기 실패: {e}")
                    print(f"   수동으로 열어주세요: {html_file}")
            
            return True
            
        else:
            print("❌ 플롯 생성에 실패했습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_simple_html_report(station_id: str, plot_files: dict, output_dir: str) -> str:
    """간단한 HTML 리포트 생성"""
    
    try:
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{station_id} 관측소 - CRNP 분석 결과</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .plot-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .plot-card h3 {{
            background: #f8f9fa;
            margin: 0;
            padding: 15px 20px;
            font-size: 1.1em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{station_id} 관측소</h1>
            <p>CRNP 토양수분 분석 결과</p>
        </div>
        
        <div class="content">
            <h2>📊 분석 결과 플롯</h2>
            <div class="plot-grid">
        """
        
        # 플롯 정보
        plot_descriptions = {
            'neutron_comparison': '중성자 카운트 비교 (원시 vs 보정)',
            'correction_factors': '중성자 보정계수',
            'vwc_timeseries': '체적수분함량 시계열',
            'sm_timeseries': '토양수분 비교 (CRNP vs 현장센서)',
            'sm_scatter': '토양수분 산점도'
        }
        
        # 각 플롯 추가
        for plot_type, file_path in plot_files.items():
            plot_name = Path(file_path).name
            description = plot_descriptions.get(plot_type, plot_type.replace('_', ' ').title())
            
            html_content += f"""
                <div class="plot-card">
                    <h3>{description}</h3>
                    <img src="{plot_name}" alt="{description}">
                </div>
            """
        
        html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>CRNP (Cosmic Ray Neutron Probe) 토양수분 모니터링 시스템</p>
        </div>
    </div>
</body>
</html>
        """
        
        # HTML 파일 저장
        html_file = Path(output_dir) / f"{station_id}_simple_report.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
        
    except Exception as e:
        print(f"HTML 리포트 생성 실패: {e}")
        return ""


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP Simple Dashboard")
    parser.add_argument("--station", "-s", required=True, help="Station ID (SWCR, HC, PC, etc.)")
    
    args = parser.parse_args()
    
    try:
        success = run_simple_dashboard(args.station)
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ 대시보드 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())