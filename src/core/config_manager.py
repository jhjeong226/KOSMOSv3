# src/core/config_manager.py

import yaml
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class ConfigManager:
    """CRNP 시스템의 설정 관리를 담당하는 클래스"""
    
    def __init__(self, config_root: str = "config"):
        self.config_root = Path(config_root)
        self.logger = logging.getLogger(__name__)
        
        # 파일 패턴 정의
        self.file_patterns = {
            'PC': r'z6-(\d+)\([^)]+\)\(z6-\1\)-Configuration.*\.csv$',
            'HC': r'HC-[^(]+\(z6-(\d+)\)\(z6-\1\)-Configuration.*\.csv$',
            'unified': r'(?:HC-[^(]+)?\(z6-(\d+)\)\(z6-\1\)-Configuration.*\.csv$'
        }
        
    def load_station_config(self, station_id: str) -> Dict:
        """관측소별 설정 파일 로드"""
        config_file = self.config_root / "stations" / f"{station_id}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Station config file not found: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 경로 검증
        self._validate_station_paths(config)
        
        self.logger.info(f"Loaded station config for {station_id}")
        return config
        
    def load_processing_config(self) -> Dict:
        """처리 옵션 설정 파일 로드"""
        config_file = self.config_root / "processing_options.yaml"
        
        if not config_file.exists():
            # 기본 설정 생성
            return self._create_default_processing_config()
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.logger.info("Loaded processing options config")
        return config
        
    def load_neutron_monitors(self) -> Dict:
        """중성자 모니터 설정 파일 로드"""
        config_file = self.config_root / "neutron_monitors.yaml"
        
        if not config_file.exists():
            return self._create_default_neutron_monitors()
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
        
    def load_geo_info_from_yaml(self, station_config: Dict) -> Dict:
        """YAML 설정에서 지리정보 로드 및 전처리"""
        try:
            # CRNP 관측소 정보
            crnp_info = {
                'id': 'CRNP',
                'lat': station_config['coordinates']['latitude'],
                'lon': station_config['coordinates']['longitude'],
                'dist': 0,
                'sbd': station_config['soil_properties']['bulk_density']
            }
            
            # 센서 정보를 DataFrame으로 변환
            sensors_data = []
            for sensor_id, sensor_info in station_config['sensors'].items():
                sensor_data = {
                    'id': sensor_id,
                    'lat': sensor_info['latitude'],
                    'lon': sensor_info['longitude'],
                    'dist': sensor_info['distance_from_station'],
                    'sbd': sensor_info['bulk_density'],
                    'loc_key': str(sensor_info['loc_key'])
                }
                sensors_data.append(sensor_data)
                
            sensor_df = pd.DataFrame(sensors_data)
            
            # 전체 데이터 (CRNP + 센서들)
            all_data = sensors_data + [crnp_info]
            all_df = pd.DataFrame(all_data)
            
            self.logger.info(f"Loaded geo info from YAML: {len(sensor_df)} sensors, CRNP: Yes")
            
            return {
                'sensors': sensor_df,
                'crnp': pd.Series(crnp_info),
                'all': all_df
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load geo info from YAML: {str(e)}")
            raise
            
    def match_files_to_sensors(self, file_list: List[str], geo_df: pd.DataFrame, 
                              station_type: str = 'unified') -> Dict[str, str]:
        """파일명과 센서 정보 매칭"""
        pattern = self.file_patterns.get(station_type, self.file_patterns['unified'])
        matched_files = {}
        unmatched_files = []
        
        # geo_df에서 loc_key 매핑 생성
        loc_key_mapping = {
            str(row['loc_key']): row['id'] 
            for _, row in geo_df.iterrows()
        }
        
        for file_name in file_list:
            match = re.search(pattern, file_name)
            if match:
                loc_key = match.group(1)
                if loc_key in loc_key_mapping:
                    sensor_id = loc_key_mapping[loc_key]
                    matched_files[sensor_id] = file_name
                    self.logger.debug(f"Matched: {file_name} -> {sensor_id} (loc_key: {loc_key})")
                else:
                    self.logger.warning(f"No sensor found for loc_key {loc_key} in file {file_name}")
                    unmatched_files.append(file_name)
            else:
                self.logger.warning(f"File pattern not matched: {file_name}")
                unmatched_files.append(file_name)
                
        self.logger.info(f"File matching complete: {len(matched_files)} matched, {len(unmatched_files)} unmatched")
        
        return {
            'matched': matched_files,
            'unmatched': unmatched_files
        }
        
    def create_station_template(self, station_id: str, station_name: str, 
                               lat: float, lon: float, 
                               soil_bulk_density: float = 1.44,
                               clay_content: float = 0.35) -> str:
        """새 관측소 설정 템플릿 생성"""
        template = {
            'station_info': {
                'id': station_id,
                'name': station_name,
                'description': f"{station_name} 관측소"
            },
            'coordinates': {
                'latitude': lat,
                'longitude': lon,
                'altitude': None  # 사용자가 입력
            },
            'soil_properties': {
                'bulk_density': soil_bulk_density,
                'clay_content': clay_content,
                'lattice_water': None  # auto-calculate
            },
            'sensor_configuration': {
                'depths': [10, 30, 60],
                'fdr_sensor_count': 3
            },
            'data_paths': {
                'crnp_folder': f"data/input/{station_id}/crnp/",
                'fdr_folder': f"data/input/{station_id}/fdr/",
                'geo_info_file': f"data/input/{station_id}/geo_info.xlsx"
            },
            'file_patterns': {
                'crnp_pattern': "*.xlsx",
                'fdr_pattern': "*z6*.csv"
            },
            'calibration': {
                'neutron_monitor': "MXCO",
                'utc_offset': 9,
                'reference_depths': [10, 30, 60]
            }
        }
        
        # 파일 저장
        station_dir = self.config_root / "stations"
        station_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = station_dir / f"{station_id}.yaml"
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
            
        self.logger.info(f"Created station template: {template_file}")
        return str(template_file)
        
    def _validate_station_paths(self, config: Dict) -> None:
        """관측소 설정의 경로 유효성 검사"""
        paths = config.get('data_paths', {})
        
        for path_key, path_value in paths.items():
            if path_key.endswith('_folder'):
                if not os.path.exists(path_value):
                    self.logger.warning(f"Folder not found: {path_value}")
            elif path_key.endswith('_file'):
                if not os.path.exists(path_value):
                    self.logger.warning(f"File not found: {path_value}")
                    
    def _create_default_processing_config(self) -> Dict:
        """기본 처리 설정 생성"""
        return {
            'corrections': {
                'incoming_flux': True,
                'pressure': True,
                'humidity': True,
                'biomass': False
            },
            'calibration': {
                'optimization_method': 'Nelder-Mead',
                'initial_N0': 1000,
                'weighting_method': 'Schron_2017'
            },
            'calculation': {
                'exclude_periods': {
                    'winter_months': [12, 1, 2],
                    'custom_dates': []
                },
                'smoothing': {
                    'enabled': False,
                    'method': 'savitzky_golay',
                    'window': 11,
                    'order': 3
                }
            },
            'validation': {
                'metrics': ['R2', 'RMSE', 'MAE', 'NSE']
            },
            'visualization': {
                'style': 'modern',
                'dpi': 300,
                'figure_size': [15, 7],
                'color_palette': 'Set2',
                'plots': {
                    'raw_neutron_timeseries': True,
                    'corrected_neutron_timeseries': True,
                    'correction_factors': True,
                    'soil_moisture_timeseries': True,
                    'sensing_depth_timeseries': True,
                    'validation_scatter': True,
                    'validation_timeseries': True,
                    'correlation_matrix': True
                }
            },
            'parallel_processing': {
                'enabled': True,
                'max_workers': 4
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True,
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
    def _create_default_neutron_monitors(self) -> Dict:
        """기본 중성자 모니터 설정 생성"""
        return {
            'monitors': {
                'MXCO': {
                    'name': 'Mexico City',
                    'country': 'Mexico',
                    'cutoff_rigidity': 8.2
                },
                'ATHN': {
                    'name': 'Athens',
                    'country': 'Greece', 
                    'cutoff_rigidity': 8.5
                },
                'ROME': {
                    'name': 'Rome',
                    'country': 'Italy',
                    'cutoff_rigidity': 6.3
                },
                'OULU': {
                    'name': 'Oulu',
                    'country': 'Finland',
                    'cutoff_rigidity': 0.8
                }
            }
        }


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # ConfigManager 인스턴스 생성
    config_manager = ConfigManager()
    
    # 홍천 관측소 설정 템플릿 생성
    config_manager.create_station_template(
        station_id="HC",
        station_name="Hongcheon Station", 
        lat=37.7049111,
        lon=128.0316412,
        soil_bulk_density=1.44,
        clay_content=0.35
    )
    
    print("ConfigManager 구현 완료!")