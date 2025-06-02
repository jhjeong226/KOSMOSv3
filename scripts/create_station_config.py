# scripts/create_station_config.py

"""
대화형 관측소 설정 파일 생성 스크립트

새로운 관측소의 YAML 설정 파일을 단계별로 생성합니다.

사용법:
    python scripts/create_station_config.py
    python scripts/create_station_config.py --station-id NEW_STATION
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import re

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StationConfigCreator:
    """대화형 관측소 설정 생성 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config" / "stations"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본값들
        self.defaults = {
            'soil_bulk_density': 1.44,
            'clay_content': 0.35,
            'depths': [10, 30, 60],
            'fdr_sensor_count': 3,
            'neutron_monitor': 'ATHN',
            'utc_offset': 9
        }
        
        # 검증 패턴들
        self.patterns = {
            'station_id': r'^[A-Z]{2,5}$',  # 2-5자 대문자
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'altitude': (-500, 9000),  # 해수면 기준 -500m ~ 9000m
            'bulk_density': (0.5, 2.5),  # g/cm³
            'clay_content': (0.0, 1.0),   # 0-100%를 0.0-1.0으로
            'depth': (5, 200),  # cm
            'sensor_count': (1, 10)
        }
        
    def create_config_interactive(self, station_id: Optional[str] = None) -> str:
        """대화형으로 설정 파일 생성"""
        
        print("🚀 CRNP 관측소 설정 파일 생성 도구")
        print("=" * 60)
        print("새로운 관측소의 설정 파일을 생성합니다.")
        print("각 단계에서 정보를 입력해주세요. (Enter = 기본값 사용)")
        print()
        
        try:
            # 1. 기본 정보
            station_info = self._get_station_info(station_id)
            
            # 2. 좌표 정보
            coordinates = self._get_coordinates()
            
            # 3. 토양 특성
            soil_properties = self._get_soil_properties()
            
            # 4. 센서 설정
            sensor_config = self._get_sensor_configuration()
            
            # 5. 데이터 경로
            data_paths = self._get_data_paths(station_info['id'])
            
            # 6. 캘리브레이션 설정
            calibration = self._get_calibration_settings()
            
            # 7. 센서 상세 정보 (선택사항)
            sensors = self._get_sensors_info(station_info['id'], sensor_config)
            
            # 8. 설정 파일 생성
            config = self._build_config(
                station_info, coordinates, soil_properties, 
                sensor_config, data_paths, calibration, sensors
            )
            
            # 9. 파일 저장
            config_file = self._save_config(config, station_info['id'])
            
            # 10. 완료 메시지
            self._print_completion_message(config, config_file)
            
            return config_file
            
        except KeyboardInterrupt:
            print("\n\n⏹️  사용자에 의해 중단되었습니다.")
            return ""
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            return ""
            
    def _get_station_info(self, station_id: Optional[str] = None) -> Dict[str, str]:
        """관측소 기본 정보 입력"""
        
        print("📍 1단계: 관측소 기본 정보")
        print("-" * 30)
        
        # 관측소 ID
        if station_id:
            station_id = station_id.upper()
            print(f"관측소 ID: {station_id} (매개변수로 지정됨)")
        else:
            while True:
                station_id = input("관측소 ID (2-5자 대문자, 예: HC, PC, KWL): ").strip().upper()
                
                if not station_id:
                    print("❌ 관측소 ID는 필수입니다.")
                    continue
                    
                if not re.match(self.patterns['station_id'], station_id):
                    print("❌ 관측소 ID는 2-5자 대문자여야 합니다. (예: HC, PC, SEOUL)")
                    continue
                    
                # 기존 파일 확인
                existing_file = self.config_dir / f"{station_id}.yaml"
                if existing_file.exists():
                    overwrite = input(f"⚠️  {station_id}.yaml이 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ").lower()
                    if overwrite != 'y':
                        continue
                        
                break
        
        # 관측소 이름
        default_name = f"{station_id} Station"
        station_name = input(f"관측소 이름 [{default_name}]: ").strip()
        if not station_name:
            station_name = default_name
            
        # 설명
        default_desc = f"{station_name} 관측소"
        description = input(f"설명 [{default_desc}]: ").strip()
        if not description:
            description = default_desc
            
        print(f"✅ 관측소 정보: {station_id} - {station_name}")
        print()
        
        return {
            'id': station_id,
            'name': station_name,
            'description': description
        }
        
    def _get_coordinates(self) -> Dict[str, float]:
        """좌표 정보 입력"""
        
        print("🌍 2단계: 좌표 정보")
        print("-" * 30)
        
        # 위도
        latitude = self._get_numeric_input(
            "위도 (도, -90~90)", 
            self.patterns['latitude'],
            example="37.7049"
        )
        
        # 경도
        longitude = self._get_numeric_input(
            "경도 (도, -180~180)", 
            self.patterns['longitude'],
            example="128.0316"
        )
        
        # 고도 (선택사항)
        altitude = self._get_numeric_input(
            "고도 (m, 해수면 기준, 선택사항)", 
            self.patterns['altitude'],
            optional=True,
            example="100"
        )
        
        print(f"✅ 좌표: {latitude:.4f}, {longitude:.4f}" + 
              (f", {altitude}m" if altitude is not None else ""))
        print()
        
        return {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        }
        
    def _get_soil_properties(self) -> Dict[str, float]:
        """토양 특성 입력"""
        
        print("🌱 3단계: 토양 특성")
        print("-" * 30)
        
        # 벌크밀도
        bulk_density = self._get_numeric_input(
            f"토양 벌크밀도 (g/cm³) [{self.defaults['soil_bulk_density']}]",
            self.patterns['bulk_density'],
            default=self.defaults['soil_bulk_density'],
            example="1.44"
        )
        
        # 점토함량
        print("점토함량을 입력하세요 (0-100% 또는 0.0-1.0):")
        clay_input = input(f"점토함량 [{self.defaults['clay_content']*100}%]: ").strip()
        
        if not clay_input:
            clay_content = self.defaults['clay_content']
        else:
            try:
                clay_value = float(clay_input.replace('%', ''))
                
                # 0-100 범위면 0-1로 변환
                if clay_value > 1.0:
                    clay_content = clay_value / 100.0
                else:
                    clay_content = clay_value
                    
                if not (0.0 <= clay_content <= 1.0):
                    raise ValueError("점토함량은 0-100% 범위여야 합니다.")
                    
            except ValueError as e:
                print(f"❌ {e}, 기본값 사용: {self.defaults['clay_content']*100}%")
                clay_content = self.defaults['clay_content']
        
        print(f"✅ 토양 특성: 벌크밀도 {bulk_density} g/cm³, 점토함량 {clay_content*100:.1f}%")
        print()
        
        return {
            'bulk_density': bulk_density,
            'clay_content': clay_content,
            'lattice_water': None  # 자동 계산
        }
        
    def _get_sensor_configuration(self) -> Dict[str, Any]:
        """센서 설정 입력"""
        
        print("🔬 4단계: 센서 설정")
        print("-" * 30)
        
        # FDR 센서 개수
        fdr_count = self._get_numeric_input(
            f"FDR 센서 개수 [{self.defaults['fdr_sensor_count']}]",
            self.patterns['sensor_count'],
            default=self.defaults['fdr_sensor_count'],
            integer=True
        )
        
        # 측정 깊이들
        print(f"측정 깊이들을 입력하세요 (cm, 쉼표로 구분)")
        default_depths_str = ", ".join(map(str, self.defaults['depths']))
        depths_input = input(f"측정 깊이 [{default_depths_str}]: ").strip()
        
        if not depths_input:
            depths = self.defaults['depths']
        else:
            try:
                depths = []
                for depth_str in depths_input.split(','):
                    depth = int(depth_str.strip())
                    if not (self.patterns['depth'][0] <= depth <= self.patterns['depth'][1]):
                        raise ValueError(f"깊이 {depth}cm는 {self.patterns['depth'][0]}-{self.patterns['depth'][1]}cm 범위를 벗어납니다.")
                    depths.append(depth)
                    
                depths = sorted(list(set(depths)))  # 중복 제거 및 정렬
                
            except ValueError as e:
                print(f"❌ {e}, 기본값 사용: {default_depths_str}")
                depths = self.defaults['depths']
        
        print(f"✅ 센서 설정: FDR {fdr_count}개, 깊이 {depths}cm")
        print()
        
        return {
            'depths': depths,
            'fdr_sensor_count': fdr_count
        }
        
    def _get_data_paths(self, station_id: str) -> Dict[str, str]:
        """데이터 경로 설정"""
        
        print("📁 5단계: 데이터 경로")
        print("-" * 30)
        
        # 기본 경로들
        default_paths = {
            'crnp_folder': f"data/input/{station_id}/crnp/",
            'fdr_folder': f"data/input/{station_id}/fdr/",
            'geo_info_file': f"data/input/{station_id}/geo_info.xlsx"
        }
        
        print("기본 데이터 경로 설정:")
        for key, path in default_paths.items():
            print(f"  {key}: {path}")
        
        use_default = input("\n기본 경로를 사용하시겠습니까? (Y/n): ").lower()
        
        if use_default == 'n':
            paths = {}
            for key, default_path in default_paths.items():
                custom_path = input(f"{key} [{default_path}]: ").strip()
                paths[key] = custom_path if custom_path else default_path
        else:
            paths = default_paths
            
        # 디렉토리 생성
        for key, path in paths.items():
            if key.endswith('_folder'):
                Path(path).mkdir(parents=True, exist_ok=True)
                print(f"📂 생성됨: {path}")
        
        print("✅ 데이터 경로 설정 완료")
        print()
        
        return paths
        
    def _get_calibration_settings(self) -> Dict[str, Any]:
        """캘리브레이션 설정 입력"""
        
        print("⚙️  6단계: 캘리브레이션 설정")
        print("-" * 30)
        
        # 중성자 모니터
        available_monitors = ['ATHN', 'MXCO', 'ROME', 'OULU']
        print(f"사용 가능한 중성자 모니터: {', '.join(available_monitors)}")
        
        monitor_input = input(f"중성자 모니터 [{self.defaults['neutron_monitor']}]: ").strip().upper()
        neutron_monitor = monitor_input if monitor_input in available_monitors else self.defaults['neutron_monitor']
        
        # UTC 오프셋
        utc_offset = self._get_numeric_input(
            f"UTC 오프셋 (시간) [{self.defaults['utc_offset']}]",
            (-12, 14),
            default=self.defaults['utc_offset'],
            integer=True
        )
        
        print(f"✅ 캘리브레이션: {neutron_monitor} 모니터, UTC{utc_offset:+d}")
        print()
        
        return {
            'neutron_monitor': neutron_monitor,
            'utc_offset': utc_offset,
            'reference_depths': None  # sensor_configuration에서 가져옴
        }
        
    def _get_sensors_info(self, station_id: str, sensor_config: Dict) -> Optional[Dict]:
        """센서 상세 정보 입력 (선택사항)"""
        
        print("📊 7단계: 센서 상세 정보 (선택사항)")
        print("-" * 30)
        print("개별 센서의 위치와 특성을 입력할 수 있습니다.")
        print("이 정보는 가중평균 계산에 사용됩니다.")
        
        add_sensors = input("센서 상세 정보를 입력하시겠습니까? (y/N): ").lower()
        
        if add_sensors != 'y':
            print("⏭️  센서 상세 정보를 건너뜁니다.")
            print()
            return None
            
        sensors = {}
        sensor_count = sensor_config['fdr_sensor_count']
        
        print(f"\n{sensor_count}개 센서의 정보를 입력해주세요:")
        
        for i in range(sensor_count):
            sensor_id = input(f"\n센서 {i+1} ID (예: E1, W2): ").strip()
            if not sensor_id:
                sensor_id = f"S{i+1}"
                
            print(f"센서 {sensor_id} 정보:")
            
            # 위도
            latitude = self._get_numeric_input(
                "위도 (도)", 
                self.patterns['latitude'],
                example="37.7050"
            )
            
            # 경도
            longitude = self._get_numeric_input(
                "경도 (도)", 
                self.patterns['longitude'],
                example="128.0320"
            )
            
            # 관측소로부터 거리
            distance = self._get_numeric_input(
                "관측소로부터 거리 (m)",
                (0, 1000),
                example="50"
            )
            
            # loc_key (파일명 매칭용)
            loc_key = input("loc_key (파일명의 숫자 부분, 예: 19850): ").strip()
            if not loc_key:
                loc_key = f"000{i+1}"
                
            # 토양 벌크밀도 (센서별)
            bulk_density = self._get_numeric_input(
                f"토양 벌크밀도 (g/cm³) [{self.defaults['soil_bulk_density']}]",
                self.patterns['bulk_density'],
                default=self.defaults['soil_bulk_density']
            )
            
            sensors[sensor_id] = {
                'latitude': latitude,
                'longitude': longitude,
                'distance_from_station': distance,
                'loc_key': loc_key,
                'bulk_density': bulk_density
            }
            
            print(f"✅ 센서 {sensor_id} 정보 저장됨")
        
        print(f"✅ {len(sensors)}개 센서 정보 입력 완료")
        print()
        
        return sensors
        
    def _get_numeric_input(self, prompt: str, range_or_tuple: Tuple[float, float], 
                          default: Optional[float] = None, optional: bool = False,
                          integer: bool = False, example: str = "") -> Optional[float]:
        """숫자 입력 헬퍼"""
        
        min_val, max_val = range_or_tuple
        range_str = f"{min_val}~{max_val}"
        
        if example:
            prompt += f" (예: {example})"
        if optional:
            prompt += " (선택사항)"
        prompt += ": "
        
        while True:
            user_input = input(prompt).strip()
            
            # 빈 입력 처리
            if not user_input:
                if default is not None:
                    return default
                elif optional:
                    return None
                else:
                    print("❌ 값을 입력해주세요.")
                    continue
            
            try:
                if integer:
                    value = int(user_input)
                else:
                    value = float(user_input)
                    
                if min_val <= value <= max_val:
                    return float(value) if not integer else int(value)
                else:
                    print(f"❌ {range_str} 범위의 값을 입력해주세요.")
                    
            except ValueError:
                print("❌ 유효한 숫자를 입력해주세요.")
                
    def _build_config(self, station_info: Dict, coordinates: Dict, 
                     soil_properties: Dict, sensor_config: Dict,
                     data_paths: Dict, calibration: Dict,
                     sensors: Optional[Dict]) -> Dict:
        """설정 딕셔너리 구성"""
        
        config = {
            'station_info': station_info,
            'coordinates': coordinates,
            'soil_properties': soil_properties,
            'sensor_configuration': sensor_config,
            'data_paths': data_paths,
            'file_patterns': {
                'crnp_pattern': "*.xlsx",
                'fdr_pattern': "*z6*.csv"
            },
            'calibration': {
                **calibration,
                'reference_depths': sensor_config['depths']
            }
        }
        
        # 센서 정보 추가 (있는 경우)
        if sensors:
            config['sensors'] = sensors
            
        return config
        
    def _save_config(self, config: Dict, station_id: str) -> str:
        """설정 파일 저장"""
        
        config_file = self.config_dir / f"{station_id}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
                         
            return str(config_file)
            
        except Exception as e:
            raise Exception(f"설정 파일 저장 실패: {e}")
            
    def _print_completion_message(self, config: Dict, config_file: str):
        """완료 메시지 출력"""
        
        station_id = config['station_info']['id']
        
        print("=" * 60)
        print("🎉 관측소 설정 파일 생성 완료!")
        print("=" * 60)
        print(f"파일 위치: {config_file}")
        print(f"관측소 ID: {station_id}")
        print(f"관측소 이름: {config['station_info']['name']}")
        
        coords = config['coordinates']
        print(f"좌표: {coords['latitude']:.4f}, {coords['longitude']:.4f}")
        
        soil = config['soil_properties']
        print(f"토양: 벌크밀도 {soil['bulk_density']} g/cm³, 점토함량 {soil['clay_content']*100:.1f}%")
        
        sensor = config['sensor_configuration']
        print(f"센서: FDR {sensor['fdr_sensor_count']}개, 깊이 {sensor['depths']}cm")
        
        if 'sensors' in config:
            print(f"센서 상세 정보: {len(config['sensors'])}개 센서")
            
        print("\n🎯 다음 단계:")
        print("   1. 데이터 파일 배치:")
        for key, path in config['data_paths'].items():
            if key.endswith('_folder'):
                print(f"      📂 {path}")
                
        print(f"   2. 데이터 확인: python scripts/run_preprocessing.py --station {station_id} --check-only")
        print(f"   3. 전처리 실행: python scripts/run_preprocessing.py --station {station_id}")
        print(f"   4. 전체 파이프라인: python scripts/run_crnp_pipeline.py --station {station_id} --all")
        
        print("\n💡 설정 파일 수정:")
        print(f"   nano {config_file}")
        print(f"   code {config_file}")


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="대화형 관측소 설정 생성")
    parser.add_argument("--station-id", "-s", help="관측소 ID (미리 지정)")
    
    args = parser.parse_args()
    
    try:
        creator = StationConfigCreator()
        config_file = creator.create_config_interactive(args.station_id)
        
        if config_file:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())