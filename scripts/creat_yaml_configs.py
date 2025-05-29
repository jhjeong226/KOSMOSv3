# scripts/create_yaml_configs.py

"""
CRNP 관측소 YAML 설정 파일 생성 스크립트
HC.yaml과 PC.yaml 파일을 config/stations/ 폴더에 생성합니다.
"""

import os
import yaml
from pathlib import Path

def create_hc_config():
    """홍천(HC) 관측소 설정 파일 생성"""
    
    hc_config = {
        'station_info': {
            'id': 'HC',
            'name': 'Hongcheon Station',
            'description': '홍천 CRNP 관측소',
            'operator': 'KIHS',
            'established_date': '2024'
        },
        'coordinates': {
            'latitude': 37.7049111,
            'longitude': 128.0316412,
            'altitude': 444.3027,
            'utm_zone': '52S'
        },
        'soil_properties': {
            'bulk_density': 1.44,
            'clay_content': 0.35,
            'lattice_water': None,
            'soil_type': 'forest_soil'
        },
        'sensor_configuration': {
            'depths': [10, 30, 60],
            'fdr_sensor_count': 3,
            'total_sensors': 12
        },
        'data_paths': {
            'fdr_folder': 'data/input/HC/fdr/',
            'crnp_folder': 'data/input/HC/crnp/',
            'output_folder': 'data/output/HC/'
        },
        'file_patterns': {
            'crnp_pattern': '*.xlsx',
            'fdr_pattern': 'HC-*.csv'
        },
        'calibration': {
            'neutron_monitor': 'MXCO',
            'utc_offset': 9,
            'reference_depths': [10, 30, 60]
        },
        'sensors': {
            'E1': {
                'loc_key': '19850',
                'latitude': 37.705095,
                'longitude': 128.031827,
                'distance_from_station': 26,
                'bulk_density': 1.44,
                'direction': '동쪽',
                'description': '동쪽 1번 센서'
            },
            'E2': {
                'loc_key': '19846',
                'latitude': 37.705243,
                'longitude': 128.032208,
                'distance_from_station': 62,
                'bulk_density': 1.44,
                'direction': '동쪽',
                'description': '동쪽 2번 센서'
            },
            'E3': {
                'loc_key': '19853',
                'latitude': 37.705527,
                'longitude': 128.032452,
                'distance_from_station': 99,
                'bulk_density': 1.44,
                'direction': '동쪽',
                'description': '동쪽 3번 센서'
            },
            'E4': {
                'loc_key': '19843',
                'latitude': 37.705797,
                'longitude': 128.032682,
                'distance_from_station': 135,
                'bulk_density': 1.44,
                'direction': '동쪽',
                'description': '동쪽 4번 센서'
            },
            'N1': {
                'loc_key': '05589',
                'latitude': 37.705723,
                'longitude': 128.031175,
                'distance_from_station': 99,
                'bulk_density': 1.44,
                'direction': '북쪽',
                'description': '북쪽 1번 센서'
            },
            'N2': {
                'loc_key': '19848',
                'latitude': 37.705849,
                'longitude': 128.030982,
                'distance_from_station': 119,
                'bulk_density': 1.44,
                'direction': '북쪽',
                'description': '북쪽 2번 센서'
            },
            'W1': {
                'loc_key': '19852',
                'latitude': 37.704954,
                'longitude': 128.031102,
                'distance_from_station': 48,
                'bulk_density': 1.44,
                'direction': '서쪽',
                'description': '서쪽 1번 센서'
            },
            'W2': {
                'loc_key': '19851',
                'latitude': 37.704848,
                'longitude': 128.030916,
                'distance_from_station': 65,
                'bulk_density': 1.44,
                'direction': '서쪽',
                'description': '서쪽 2번 센서'
            },
            'S1': {
                'loc_key': '19854',
                'latitude': 37.704633,
                'longitude': 128.031398,
                'distance_from_station': 37,
                'bulk_density': 1.44,
                'direction': '남쪽',
                'description': '남쪽 1번 센서'
            },
            'S2': {
                'loc_key': '19847',
                'latitude': 37.704295,
                'longitude': 128.031615,
                'distance_from_station': 68,
                'bulk_density': 1.44,
                'direction': '남쪽',
                'description': '남쪽 2번 센서'
            },
            'S3': {
                'loc_key': '19903',
                'latitude': 37.704037,
                'longitude': 128.031937,
                'distance_from_station': 100,
                'bulk_density': 1.44,
                'direction': '남쪽',
                'description': '남쪽 3번 센서'
            },
            'S4': {
                'loc_key': '19845',
                'latitude': 37.70389,
                'longitude': 128.03169,
                'distance_from_station': 113,
                'bulk_density': 1.44,
                'direction': '남쪽',
                'description': '남쪽 4번 센서'
            }
        },
        'quality_control': {
            'theta_v_range': {
                'min': 0.0,
                'max': 0.8
            },
            'outlier_detection': {
                'method': 'mad',
                'threshold': 3.0
            },
            'missing_data_threshold': 0.1
        },
        'metadata': {
            'created_date': '2024-05-29',
            'last_updated': '2024-05-29',
            'version': '1.0',
            'contact': 'KIHS CRNP Team',
            'notes': '홍천 관측소 FDR 센서 12개소'
        }
    }
    
    return hc_config

def create_pc_config():
    """평창(PC) 관측소 설정 파일 생성"""
    
    pc_config = {
        'station_info': {
            'id': 'PC',
            'name': 'Pyeongchang Station',
            'description': '평창 CRNP 관측소',
            'operator': 'KIHS',
            'established_date': '2024'
        },
        'coordinates': {
            'latitude': 37.53126519,  # 실제 평창 관측소 좌표
            'longitude': 128.4461,    # 실제 평창 관측소 좌표
            'altitude': 700.0,
            'utm_zone': '52S'
        },
        'soil_properties': {
            'bulk_density': 1.2,      # 실제 측정값
            'clay_content': 0.35,
            'lattice_water': None,
            'soil_type': 'mountain_soil'
        },
        'sensor_configuration': {
            'depths': [10, 30, 60],
            'fdr_sensor_count': 3,
            'total_sensors': 12
        },
        'data_paths': {
            'fdr_folder': 'data/input/PC/fdr/',
            'crnp_folder': 'data/input/PC/crnp/',
            'output_folder': 'data/output/PC/'
        },
        'file_patterns': {
            'crnp_pattern': '*.xlsx',
            'fdr_pattern': 'z6-*.csv'
        },
        'calibration': {
            'neutron_monitor': 'MXCO',
            'utc_offset': 9,
            'reference_depths': [10, 30, 60]
        },
        'sensors': {
            'S25': {
                'loc_key': '25663',
                'latitude': 37.53108603,
                'longitude': 128.4462,
                'distance_from_station': 21.66148,
                'bulk_density': 1.2,
                'direction': '남쪽',
                'description': '남쪽 25m 센서'
            },
            'N50': {
                'loc_key': '25665',
                'latitude': 37.53170031,
                'longitude': 128.446,
                'distance_from_station': 50.11846,
                'bulk_density': 1.2,
                'direction': '북쪽',
                'description': '북쪽 50m 센서'
            },
            'N100': {
                'loc_key': '25666',
                'latitude': 37.532126,
                'longitude': 128.4458,
                'distance_from_station': 99.30223,
                'bulk_density': 1.2,
                'direction': '북쪽',
                'description': '북쪽 100m 센서'
            },
            'SW35': {
                'loc_key': '25668',
                'latitude': 37.53100367,
                'longitude': 128.4459,
                'distance_from_station': 35.65449,
                'bulk_density': 1.2,
                'direction': '남서쪽',
                'description': '남서쪽 35m 센서'
            },
            'NW100': {
                'loc_key': '27722',
                'latitude': 37.53196992,
                'longitude': 128.4455,
                'distance_from_station': 97.0183,
                'bulk_density': 1.2,
                'direction': '북서쪽',
                'description': '북서쪽 100m 센서'
            },
            'E50': {
                'loc_key': '27897',
                'latitude': 37.53139256,
                'longitude': 128.4467,
                'distance_from_station': 48.76589,
                'bulk_density': 1.2,
                'direction': '동쪽',
                'description': '동쪽 50m 센서'
            },
            'NE50': {
                'loc_key': '28109',
                'latitude': 37.53163275,
                'longitude': 128.4465,
                'distance_from_station': 50.12719,
                'bulk_density': 1.2,
                'direction': '북동쪽',
                'description': '북동쪽 50m 센서'
            },
            'NW25': {
                'loc_key': '28110',
                'latitude': 37.53143569,
                'longitude': 128.446,
                'distance_from_station': 24.76993,
                'bulk_density': 1.2,
                'direction': '북서쪽',
                'description': '북서쪽 25m 센서'
            },
            'SE75': {
                'loc_key': '28156',
                'latitude': 37.53078683,
                'longitude': 128.4467,
                'distance_from_station': 74.13626,
                'bulk_density': 1.2,
                'direction': '남동쪽',
                'description': '남동쪽 75m 센서'
            },
            'NE75': {
                'loc_key': '28157',
                'latitude': 37.53181806,
                'longitude': 128.4466,
                'distance_from_station': 73.51689,
                'bulk_density': 1.2,
                'direction': '북동쪽',
                'description': '북동쪽 75m 센서'
            },
            'S75': {
                'loc_key': '28158',
                'latitude': 37.53063353,
                'longitude': 128.4464,
                'distance_from_station': 73.29251,
                'bulk_density': 1.2,
                'direction': '남쪽',
                'description': '남쪽 75m 센서'
            },
            'SE100': {
                'loc_key': '28159',
                'latitude': 37.53063417,
                'longitude': 128.4469,
                'distance_from_station': 98.36039,
                'bulk_density': 1.2,
                'direction': '남동쪽',
                'description': '남동쪽 100m 센서'
            }
        },
        'quality_control': {
            'theta_v_range': {
                'min': 0.0,
                'max': 0.8
            },
            'outlier_detection': {
                'method': 'mad',
                'threshold': 3.0
            },
            'missing_data_threshold': 0.1
        },
        'metadata': {
            'created_date': '2024-05-29',
            'last_updated': '2024-05-29',
            'version': '1.0',
            'contact': 'KIHS CRNP Team',
            'notes': '평창 관측소 FDR 센서 12개소 - 실제 측정 좌표 적용 완료'
        }
    }
    
    return pc_config

def main():
    """메인 실행 함수"""
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config" / "stations"
    
    # config/stations 디렉토리 생성
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 CRNP 관측소 YAML 설정 파일 생성")
    print("=" * 50)
    
    # HC.yaml 생성
    print("📝 홍천(HC) 관측소 설정 파일 생성 중...")
    hc_config = create_hc_config()
    hc_file = config_dir / "HC.yaml"
    
    with open(hc_file, 'w', encoding='utf-8') as f:
        yaml.dump(hc_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"   ✅ 생성 완료: {hc_file}")
    
    # PC.yaml 생성
    print("📝 평창(PC) 관측소 설정 파일 생성 중...")
    pc_config = create_pc_config()
    pc_file = config_dir / "PC.yaml"
    
    with open(pc_file, 'w', encoding='utf-8') as f:
        yaml.dump(pc_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"   ✅ 생성 완료: {pc_file}")
    
    print("\n📊 생성된 설정 파일 정보:")
    print(f"   HC 관측소: {len(hc_config['sensors'])}개 센서")
    print(f"   PC 관측소: {len(pc_config['sensors'])}개 센서")
    
    print("\n⚠️  중요 안내:")
    print("   - HC.yaml: 홍천 관측소 실제 데이터로 생성됨")
    print("   - PC.yaml: 평창 관측소 실제 데이터로 생성됨")
    
    print("\n🎯 다음 단계:")
    print("   1. 데이터 파일 배치 (data/input/{station_id}/)")
    print("   2. 전처리 실행: python scripts/run_preprocessing.py --station HC")
    print("   3. 전처리 실행: python scripts/run_preprocessing.py --station PC")
    
    print("\n✅ YAML 설정 파일 생성 완료!")

if __name__ == "__main__":
    main()