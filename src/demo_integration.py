# demo_integration.py - 기반 시스템 통합 데모

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.logger import CRNPLogger, ProcessTimer
from src.utils.file_handler import FileHandler
from src.preprocessing.data_validator import DataValidator


def demo_integrated_system():
    """기반 시스템 통합 데모"""
    
    print("🚀 CRNP 처리 시스템 기반 모듈 데모 시작")
    print("=" * 60)
    
    # 1. 로거 초기화
    main_logger = CRNPLogger("CRNP_Demo", level="INFO")
    main_logger.info("CRNP 처리 시스템 데모 시작")
    
    # 2. 설정 관리자 초기화
    with ProcessTimer(main_logger, "Configuration Setup"):
        config_manager = ConfigManager()
        
        # 홍천 관측소 설정 템플릿 생성
        hc_config_file = config_manager.create_station_template(
            station_id="HC",
            station_name="Hongcheon Station",
            lat=37.7049111,
            lon=128.0316412,
            soil_bulk_density=1.44,
            clay_content=0.35
        )
        main_logger.info(f"Created station config: {hc_config_file}")
        
        # 처리 옵션 설정 로드
        processing_config = config_manager.load_processing_config()
        main_logger.info("Loaded processing configuration")
    
    # 3. 관측소별 로거 생성
    hc_logger = main_logger.create_station_logger("HC")
    hc_logger.info("Hongcheon station logger initialized")
    
    # 4. 파일 핸들러 테스트
    with ProcessTimer(hc_logger, "File Discovery"):
        file_handler = FileHandler(hc_logger)
        
        # 테스트용 파일명 리스트 (실제 홍천 파일명)
        test_files = [
            "HC-E1(z6-19850)(z6-19850)-Configuration 2-1726742190.6906087.csv",
            "HC-E2(z6-19846)(z6-19846)-Configuration 2-1726742157.351288.csv",
            "HC-E3(z6-19853)(z6-19853)-Configuration 2-1726742116.9166503.csv",
            "HC-W1(z6-19852)(z6-19852)-Configuration-1726742241.8616657.csv",
            "random_file.txt"  # 매칭되지 않을 파일
        ]
        
        # 파일명에서 loc_key 추출 테스트
        for filename in test_files:
            loc_key = file_handler.extract_loc_key_from_filename(filename, "HC")
            if loc_key:
                hc_logger.info(f"Extracted loc_key {loc_key} from {filename}")
            else:
                hc_logger.warning(f"Could not extract loc_key from {filename}")
    
    # 5. 지리정보 파일 처리 (실제 geo_locations.xlsx 사용)
    with ProcessTimer(hc_logger, "Geo Info Processing"):
        try:
            geo_info = config_manager.load_geo_info("geo_locations.xlsx")
            hc_logger.log_data_summary(
                "GeoInfo", 
                len(geo_info['all']),
                sensors=len(geo_info['sensors']),
                crnp_station="Yes" if geo_info['crnp'] is not None else "No"
            )
            
            # 파일과 센서 매칭
            matching_result = config_manager.match_files_to_sensors(
                test_files, 
                geo_info['sensors'], 
                "HC"
            )
            
            hc_logger.info(f"File matching: {len(matching_result['matched'])} matched, {len(matching_result['unmatched'])} unmatched")
            
            # 매칭 결과 상세 출력
            for sensor_id, filename in matching_result['matched'].items():
                hc_logger.info(f"Matched: {sensor_id} -> {filename}")
                
        except FileNotFoundError:
            hc_logger.warning("geo_locations.xlsx not found, skipping geo info demo")
    
    # 6. 데이터 검증 테스트
    with ProcessTimer(hc_logger, "Data Validation"):
        validator = DataValidator(hc_logger)
        
        # 테스트용 FDR 데이터 생성
        import pandas as pd
        import numpy as np
        
        # 정상적인 토양수분 데이터
        test_fdr_data = pd.DataFrame({
            'Timestamps': pd.date_range('2024-08-17', periods=100, freq='H'),
            ' m3/m3 Water Content': np.random.normal(0.25, 0.03, 100),
            ' m3/m3 Water Content.1': np.random.normal(0.30, 0.03, 100),
            ' m3/m3 Water Content.2': np.random.normal(0.35, 0.03, 100)
        })
        
        # 몇 개의 이상값 추가
        test_fdr_data.iloc[10, 1] = 0.95  # 범위 초과
        test_fdr_data.iloc[20, 2] = -0.1  # 음수값
        test_fdr_data.iloc[30, 3] = np.nan  # 결측값
        
        # 검증 실행
        validation_result = validator.validate_fdr_data(test_fdr_data, "E1")
        
        # 검증 보고서 생성 및 출력
        report = validator.generate_validation_report(validation_result)
        print("\n" + report)
        
        # 로그에 검증 결과 기록
        severity_counts = validation_result['severity_counts']
        hc_logger.info(f"Validation complete: {severity_counts['critical']} critical, {severity_counts['warning']} warnings")
    
    # 7. 설정 파일 구조 확인
    with ProcessTimer(hc_logger, "Configuration Review"):
        try:
            # 생성된 HC 설정 로드 (존재하는 경우)
            if os.path.exists("config/stations/HC.yaml"):
                hc_config = config_manager.load_station_config("HC")
                hc_logger.info("Successfully loaded HC station configuration")
                
                # 중성자 모니터 설정 확인
                neutron_monitors = config_manager.load_neutron_monitors()
                available_monitors = list(neutron_monitors['monitors'].keys())
                hc_logger.info(f"Available neutron monitors: {available_monitors}")
                
        except Exception as e:
            hc_logger.warning(f"Configuration review failed: {e}")
    
    # 8. 성능 및 메모리 사용량 로깅
    import psutil
    import time
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    main_logger.info(f"Demo completed successfully")
    main_logger.info(f"Memory usage: {memory_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✅ 기반 시스템 통합 데모 완료!")
    print("\n📊 구현된 모듈:")
    print("  ✓ ConfigManager - 설정 파일 관리")
    print("  ✓ Logger - 다중 레벨 로깅 시스템")
    print("  ✓ FileHandler - 파일 패턴 처리")
    print("  ✓ DataValidator - 데이터 품질 검증")
    
    print("\n🎯 다음 단계: 데이터 전처리 모듈 구현")
    print("  • FDR 데이터 전처리")
    print("  • CRNP 데이터 전처리")
    print("  • 데이터 통합 및 저장")


def create_directory_structure():
    """프로젝트 디렉토리 구조 생성"""
    directories = [
        "config/stations",
        "src/core",
        "src/preprocessing", 
        "src/utils",
        "data/input/HC/fdr",
        "data/input/HC/crnp",
        "data/output/HC",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("📁 프로젝트 디렉토리 구조 생성 완료")


if __name__ == "__main__":
    # 디렉토리 구조 생성
    create_directory_structure()
    
    # 통합 데모 실행
    try:
        demo_integrated_system()
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()