# scripts/run_preprocessing.py

"""
CRNP 데이터 전처리 실행 스크립트

사용법:
    python scripts/run_preprocessing.py --station HC
    python scripts/run_preprocessing.py --station HC --check-only
    python scripts/run_preprocessing.py --setup-station HC
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger


class PreprocessingRunner:
    """전처리 실행을 관리하는 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = setup_logger("PreprocessingRunner")
        
    def setup_directories(self):
        """필요한 디렉토리 구조 생성"""
        directories = [
            "config/stations",
            "data/input/HC/fdr",
            "data/input/HC/crnp", 
            "data/output/HC/preprocessed",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger.info("Project directory structure created")
        
    def setup_station_config(self, station_id: str):
        """관측소 설정 파일 생성"""
        config_manager = ConfigManager()
        
        # 관측소별 기본 설정
        station_configs = {
            'HC': {
                'name': 'Hongcheon Station',
                'lat': 37.7049111,
                'lon': 128.0316412,
                'bulk_density': 1.44,
                'clay_content': 0.35
            },
            'PC': {
                'name': 'Pyeongchang Station',
                'lat': 37.7049111,  # 실제 좌표로 수정 필요
                'lon': 128.0316412,  # 실제 좌표로 수정 필요
                'bulk_density': 1.44,
                'clay_content': 0.35
            }
        }
        
        if station_id not in station_configs:
            self.logger.error(f"Unknown station ID: {station_id}")
            return False
            
        config = station_configs[station_id]
        
        try:
            config_file = config_manager.create_station_template(
                station_id=station_id,
                station_name=config['name'],
                lat=config['lat'],
                lon=config['lon'],
                soil_bulk_density=config['bulk_density'],
                clay_content=config['clay_content']
            )
            
            self.logger.info(f"Created station configuration: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create station config: {e}")
            return False
            
    def check_data_availability(self, station_id: str) -> Dict[str, bool]:
        """데이터 파일 존재 확인"""
        base_path = self.project_root / "data" / "input" / station_id
        
        checks = {
            'fdr_folder': (base_path / "fdr").exists(),
            'crnp_folder': (base_path / "crnp").exists()
        }
        
        # FDR 폴더에 CSV 파일이 있는지 확인
        if checks['fdr_folder']:
            fdr_files = list((base_path / "fdr").glob("*.csv"))
            checks['fdr_files'] = len(fdr_files) > 0
            checks['fdr_file_count'] = len(fdr_files)
        else:
            checks['fdr_files'] = False
            checks['fdr_file_count'] = 0
            
        # CRNP 폴더에 데이터 파일이 있는지 확인
        if checks['crnp_folder']:
            crnp_files = list((base_path / "crnp").glob("*.xlsx")) + list((base_path / "crnp").glob("*.csv"))
            checks['crnp_files'] = len(crnp_files) > 0
            checks['crnp_file_count'] = len(crnp_files)
        else:
            checks['crnp_files'] = False
            checks['crnp_file_count'] = 0
            
        return checks
        
    def print_data_setup_guide(self, station_id: str):
        """데이터 배치 가이드 출력"""
        base_path = self.project_root / "data" / "input" / station_id
        
        print(f"\n📁 {station_id} 관측소 데이터 배치 가이드")
        print("=" * 50)
        print("다음 위치에 데이터 파일들을 복사해주세요:\n")
        
        print(f"1. FDR 센서 데이터 (CSV 파일들):")
        print(f"   📂 {base_path / 'fdr' / ''}")
        if station_id == "HC":
            print(f"   예시: HC-E1(z6-19850)(z6-19850)-Configuration 2-....csv")
        else:
            print(f"   예시: z6-25663(S25)(z6-25663)-Configuration 2-....csv")
        print()
        
        print(f"2. CRNP 데이터 (Excel 또는 CSV 파일):")
        print(f"   📂 {base_path / 'crnp' / ''}")
        print(f"   예시: hourly_CRNP.xlsx")
        print()
        
        print("📝 지리정보는 이제 YAML 설정 파일에서 관리됩니다:")
        print(f"   📄 config/stations/{station_id}.yaml")
        print()
        
        print("💡 파일 배치 후 다시 실행해주세요:")
        print(f"   python scripts/run_preprocessing.py --station {station_id}")
        
    def run_preprocessing(self, station_id: str, check_only: bool = False):
        """전처리 실행"""
        
        print(f"🚀 CRNP 데이터 전처리 시작 - {station_id} 관측소")
        print("=" * 60)
        
        # 1. 디렉토리 구조 확인/생성
        self.setup_directories()
        
        # 2. 데이터 파일 존재 확인
        print(f"📊 {station_id} 관측소 데이터 파일 확인 중...")
        data_checks = self.check_data_availability(station_id)
        
        # 3. 확인 결과 출력
        print("\n데이터 파일 상태:")
        status_items = [
            ("FDR 폴더", "✅" if data_checks['fdr_folder'] else "❌"),
            ("FDR 파일", f"✅ ({data_checks['fdr_file_count']}개)" if data_checks['fdr_files'] else "❌"),
            ("CRNP 폴더", "✅" if data_checks['crnp_folder'] else "❌"),
            ("CRNP 파일", f"✅ ({data_checks['crnp_file_count']}개)" if data_checks['crnp_files'] else "❌")
        ]
        
        for item, status in status_items:
            print(f"  {item:15}: {status}")
            
        # 4. 필수 파일 누락 확인
        missing_required = []
        if not data_checks['fdr_files']:
            missing_required.append("FDR 데이터 파일")
        if not data_checks['crnp_files']:
            missing_required.append("CRNP 데이터 파일")
            
        if missing_required:
            print(f"\n❌ 필수 파일이 누락되었습니다: {', '.join(missing_required)}")
            self.print_data_setup_guide(station_id)
            return False
            
        if check_only:
            print(f"\n✅ {station_id} 관측소 데이터 파일이 모두 준비되었습니다!")
            return True
            
        # 5. 설정 파일 확인/생성
        config_file = self.project_root / "config" / "stations" / f"{station_id}.yaml"
        if not config_file.exists():
            print(f"\n⚙️  {station_id} 관측소 설정 파일 생성 중...")
            if not self.setup_station_config(station_id):
                return False
                
        # 6. 전처리 파이프라인 실행
        print(f"\n🔄 {station_id} 관측소 데이터 전처리 시작...")
        
        try:
            pipeline = PreprocessingPipeline()
            results = pipeline.run_station_preprocessing(station_id)
            
            # 7. 결과 출력
            self.print_results(results)
            
            # 8. 요약 보고서 생성
            print("\n📋 요약 보고서 생성 중...")
            summary_report = pipeline.generate_summary_report()
            
            print("✅ 전처리 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 전처리 실패: {e}")
            self.logger.error(f"Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def print_results(self, results: Dict):
        """결과 출력"""
        print(f"\n📊 처리 결과 요약:")
        print(f"  전체 상태: {results['overall_status']}")
        print(f"  FDR 처리: {results['fdr']['status']}")
        print(f"  CRNP 처리: {results['crnp']['status']}")
        
        # FDR 상세 정보
        fdr_summary = results['fdr'].get('summary', {})
        if fdr_summary:
            print(f"\n🌱 FDR 데이터:")
            print(f"  센서 수: {fdr_summary.get('sensors', 0)}개")
            print(f"  총 레코드: {fdr_summary.get('total_records', 0)}개")
            if fdr_summary.get('date_range'):
                print(f"  기간: {fdr_summary['date_range']['start']} ~ {fdr_summary['date_range']['end']}")
                
        # CRNP 상세 정보
        crnp_summary = results['crnp'].get('summary', {})
        if crnp_summary:
            print(f"\n🛰️  CRNP 데이터:")
            print(f"  총 레코드: {crnp_summary.get('total_records', 0)}개")
            if crnp_summary.get('date_range'):
                print(f"  기간: {crnp_summary['date_range']['start']} ~ {crnp_summary['date_range']['end']}")
                
        # 생성된 파일들
        print(f"\n📁 생성된 파일들:")
        all_files = {}
        all_files.update(results['fdr'].get('output_files', {}))
        all_files.update(results['crnp'].get('output_files', {}))
        
        for file_type, file_path in all_files.items():
            relative_path = os.path.relpath(file_path, self.project_root)
            print(f"  {file_type}: {relative_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="CRNP 데이터 전처리 실행")
    
    parser.add_argument("--station", "-s", required=True,
                       help="관측소 ID (예: HC, PC)")
    parser.add_argument("--check-only", "-c", action="store_true",
                       help="데이터 파일 존재만 확인 (전처리 실행 안함)")
    parser.add_argument("--setup-station", action="store_true",
                       help="관측소 설정 파일만 생성")
    
    args = parser.parse_args()
    
    runner = PreprocessingRunner()
    
    if args.setup_station:
        # 설정 파일만 생성
        runner.setup_directories()
        success = runner.setup_station_config(args.station)
        if success:
            print(f"✅ {args.station} 관측소 설정 파일이 생성되었습니다.")
        else:
            print(f"❌ {args.station} 관측소 설정 파일 생성에 실패했습니다.")
    else:
        # 전처리 실행 또는 확인
        success = runner.run_preprocessing(args.station, args.check_only)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()