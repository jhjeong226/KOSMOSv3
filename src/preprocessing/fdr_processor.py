# src/preprocessing/fdr_processor.py

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler
from ..preprocessing.data_validator import DataValidator


class FDRProcessor:
    """FDR (토양수분센서) 데이터 전처리 클래스"""
    
    def __init__(self, station_config: Dict, processing_config: Dict, 
                 logger: Optional[CRNPLogger] = None):
        self.station_config = station_config
        self.processing_config = processing_config
        self.logger = logger or CRNPLogger("FDRProcessor")
        
        # 종속 모듈 초기화
        self.file_handler = FileHandler(self.logger)
        self.validator = DataValidator(self.logger)
        
        # 센서 깊이 설정
        self.depths = self.station_config['sensor_configuration']['depths']
        
        # 기존 코드와 동일한 컬럼 매핑 (정확한 컬럼명 사용)
        self.required_columns = ['Timestamps', ' m3/m3 Water Content', ' m3/m3 Water Content.1', ' m3/m3 Water Content.2']
        self.target_columns = ['Date', 'theta_v_d1', 'theta_v_d2', 'theta_v_d3']
        
    def process_all_fdr_data(self, output_dir: str) -> Dict[str, str]:
        """모든 FDR 데이터 처리 (3가지 출력 형식)"""
        
        with ProcessTimer(self.logger, "FDR Data Processing", 
                         station=self.station_config['station_info']['id']):
            
            # 1. 지리정보 로드
            geo_info = self._load_geo_info()
            
            # 2. FDR 파일들 자동 탐지 및 매칭
            matched_files = self._discover_and_match_files(geo_info['sensors'])
            
            # 3. 모든 데이터 처리
            all_processed_data = self._process_matched_files(matched_files, geo_info)
            
            # 4. 3가지 형식으로 출력 생성
            output_files = self._generate_outputs(all_processed_data, output_dir)
            
            self.logger.info(f"FDR processing complete. Generated {len(output_files)} output files")
            return output_files
            
    def _load_geo_info(self) -> Dict:
        """YAML 설정에서 지리정보 로드"""
        try:
            # ConfigManager를 사용하여 YAML에서 지리정보 로드
            from ..core.config_manager import ConfigManager
            config_manager = ConfigManager()
            geo_info = config_manager.load_geo_info_from_yaml(self.station_config)
            
            self.logger.log_data_summary(
                "GeoInfo", len(geo_info['all']),
                sensors=len(geo_info['sensors']),
                crnp_station="Yes" if geo_info['crnp'] is not None else "No"
            )
            
            return geo_info
            
        except Exception as e:
            self.logger.error(f"Failed to load geo info from YAML: {str(e)}")
            raise
            
    def _discover_and_match_files(self, geo_df: pd.DataFrame) -> Dict:
        """FDR 파일 자동 탐지 및 센서와 매칭"""
        fdr_folder = self.station_config['data_paths']['fdr_folder']
        
        if not os.path.exists(fdr_folder):
            raise FileNotFoundError(f"FDR folder not found: {fdr_folder}")
            
        # CSV 파일들 탐지
        csv_files = self.file_handler.discover_files(fdr_folder, "*.csv")
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {fdr_folder}")
            
        self.logger.info(f"Discovered {len(csv_files)} CSV files")
        
        # 파일과 센서 매칭
        station_id = self.station_config['station_info']['id']
        file_matching = self.file_handler.match_fdr_files(fdr_folder, station_id)
        
        # 지리정보와 연결
        loc_key_to_sensor = {
            str(row['loc_key']): {
                'id': row['id'],
                'lat': row['lat'],
                'lon': row['lon'],
                'dist': row['dist'],
                'sbd': row['sbd']
            }
            for _, row in geo_df.iterrows()
        }
        
        matched_files_with_info = {}
        
        for file_path in file_matching['matched']:
            loc_key = self.file_handler.extract_loc_key_from_filename(
                os.path.basename(file_path), station_id
            )
            
            if loc_key and loc_key in loc_key_to_sensor:
                sensor_info = loc_key_to_sensor[loc_key]
                matched_files_with_info[sensor_info['id']] = {
                    'file_path': file_path,
                    'loc_key': loc_key,
                    **sensor_info
                }
                
        self.logger.info(f"Successfully matched {len(matched_files_with_info)} files with sensor info")
        
        if not matched_files_with_info:
            raise ValueError("No files could be matched with sensor information")
            
        return matched_files_with_info
        
    def _process_matched_files(self, matched_files: Dict, geo_info: Dict) -> List[pd.DataFrame]:
        """매칭된 파일들 처리"""
        all_processed_data = []
        
        for sensor_id, file_info in matched_files.items():
            
            with ProcessTimer(self.logger, f"Processing {sensor_id}", 
                            file=os.path.basename(file_info['file_path'])):
                
                try:
                    # 1. 파일 읽기
                    df = self.file_handler.read_fdr_file(file_info['file_path'])
                    
                    # 2. 데이터 검증 (경고만, 처리 계속)
                    validation_result = self.validator.validate_fdr_data(df, sensor_id)
                    
                    # Critical 이슈가 있어도 경고만 하고 계속 진행
                    critical_issues = [i for i in validation_result['issues'] 
                                     if i['severity'] == 'critical']
                    if critical_issues:
                        self.logger.warning(f"Validation issues in {sensor_id}: {len(critical_issues)} critical, continuing anyway")
                    else:
                        self.logger.info(f"Data validation passed for {sensor_id}")
                            
                    # 3. 데이터 전처리 (기존 코드 방식)
                    processed_df = self._preprocess_single_file(df, file_info)
                    
                    # 4. 데이터 품질 체크
                    if len(processed_df) > 0:
                        all_processed_data.append(processed_df)
                        self.logger.log_data_summary(
                            f"FDR_{sensor_id}", len(processed_df),
                            date_range=f"{processed_df['Date'].min()} to {processed_df['Date'].max()}"
                        )
                    else:
                        self.logger.warning(f"No valid data after preprocessing for {sensor_id}")
                        
                except Exception as e:
                    self.logger.log_error_with_context(e, f"Processing {sensor_id}")
                    continue
                    
        # 처리된 데이터가 하나도 없으면 에러, 일부라도 있으면 계속 진행
        if not all_processed_data:
            raise ValueError("No valid data could be processed from any files")
        else:
            self.logger.info(f"Successfully processed {len(all_processed_data)} out of {len(matched_files)} sensors")
            
        return all_processed_data
        
    def _preprocess_single_file(self, df: pd.DataFrame, file_info: Dict) -> pd.DataFrame:
        """단일 FDR 파일 전처리 (기존 코드와 동일한 방식)"""
        
        # 1. 필요한 열 선택 및 이름 지정 (기존 코드와 동일)
        required_columns = ['Timestamps', ' m3/m3 Water Content', ' m3/m3 Water Content.1', ' m3/m3 Water Content.2']
        
        # 필요한 컬럼이 모두 있는지 확인
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns in {file_info['id']}: {missing_columns}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df_selected = df[required_columns].copy()
        df_selected.columns = ['Date', 'theta_v_d1', 'theta_v_d2', 'theta_v_d3']
        
        # 2. Date 열을 datetime 형식으로 변환
        df_selected['Date'] = pd.to_datetime(df_selected['Date'], errors='coerce')
        
        # 유효하지 않은 날짜 제거
        initial_count = len(df_selected)
        df_selected = df_selected.dropna(subset=['Date'])
        removed_dates = initial_count - len(df_selected)
        
        if removed_dates > 0:
            self.logger.warning(f"Removed {removed_dates} records with invalid dates from {file_info['id']}")
        
        if len(df_selected) == 0:
            self.logger.warning(f"No valid data remaining after date processing for {file_info['id']}")
            return pd.DataFrame()
        
        # 3. 정각에만 해당하는 데이터 필터링 (분이 00인 데이터만 선택)
        df_selected = df_selected[df_selected['Date'].dt.minute == 0]
        
        if len(df_selected) == 0:
            self.logger.warning(f"No hourly data found for {file_info['id']}")
            return pd.DataFrame()
        
        # 4. theta_v_d1, theta_v_d2, theta_v_d3 데이터를 하나로 합치기 (long format)
        df_long = pd.melt(df_selected, id_vars=['Date'],
                          value_vars=['theta_v_d1', 'theta_v_d2', 'theta_v_d3'],
                          var_name='theta_v_source', value_name='theta_v')
        
        # 5. 같은 로거에서 읽어온 토양수분 자료들을 각각 depths 순서에 맞게 할당하기
        df_long['FDR_depth'] = df_long['theta_v_source'].map({
            'theta_v_d1': self.depths[0],
            'theta_v_d2': self.depths[1],
            'theta_v_d3': self.depths[2]
        })
        
        # 6. 지점별 정보 추가
        df_long['id'] = file_info['id']
        df_long['latitude'] = file_info['lat']
        df_long['longitude'] = file_info['lon']
        df_long['distance_from_station'] = file_info['dist']
        df_long['bulk_density'] = file_info['sbd']
        
        # 7. 수치 데이터 변환
        df_long['theta_v'] = pd.to_numeric(df_long['theta_v'], errors='coerce')
        
        # 8. Date 열에서 일자 정보만 추출
        df_long['Date'] = df_long['Date'].dt.date
        
        # 9. 기본적인 데이터 검증 (물리적으로 불가능한 값 제거)
        initial_count = len(df_long)
        df_long = df_long[(df_long['theta_v'] >= 0) & (df_long['theta_v'] <= 1)]
        removed_outliers = initial_count - len(df_long)
        
        if removed_outliers > 0:
            self.logger.info(f"Removed {removed_outliers} out-of-range theta_v values from {file_info['id']}")
        
        self.logger.info(f"Processed {file_info['id']}: {len(df_long)} records")
        
        return df_long
        
    def _generate_outputs(self, all_data: List[pd.DataFrame], output_dir: str) -> Dict[str, str]:
        """3가지 형식의 출력 파일 생성"""
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 모든 데이터 합치기
        combined_data = pd.concat(all_data, ignore_index=True)
        
        station_id = self.station_config['station_info']['id']
        output_files = {}
        
        # 1. 원본 형식 (기존 PC_FDR_input.xlsx와 동일)
        with ProcessTimer(self.logger, "Generating input format"):
            input_file = output_path / f"{station_id}_FDR_input.xlsx"
            self.file_handler.save_dataframe(combined_data, str(input_file), index=False)
            output_files['input_format'] = str(input_file)
            
        # 2. 일평균 와이드 형식 (기존 PC_FDR_daily_avg.xlsx와 동일)
        with ProcessTimer(self.logger, "Generating daily average format"):
            daily_avg_df = combined_data.groupby(['Date', 'FDR_depth'])['theta_v'].mean().unstack()
            
            # 컬럼명 변경
            if len(self.depths) >= 3:
                daily_avg_df.columns = [f'{depth}cm' for depth in self.depths]
                
            daily_avg_file = output_path / f"{station_id}_FDR_daily_avg.xlsx"
            self.file_handler.save_dataframe(daily_avg_df, str(daily_avg_file), index=True)
            output_files['daily_average'] = str(daily_avg_file)
            
        # 3. 조직화된 형식 (기존 PC_FDR_all_sites.xlsx 및 PC_FDR_daily_depths.xlsx와 동일)
        with ProcessTimer(self.logger, "Generating organized formats"):
            
            # 3-1. 모든 사이트 와이드 형식
            combined_data['column_name'] = combined_data['id'] + '_' + combined_data['FDR_depth'].astype(str) + 'cm'
            all_sites_df = combined_data.pivot_table(
                index='Date', 
                columns='column_name', 
                values='theta_v', 
                aggfunc='mean'
            )
            
            all_sites_file = output_path / f"{station_id}_FDR_all_sites.xlsx"
            self.file_handler.save_dataframe(all_sites_df, str(all_sites_file), index=True)
            output_files['all_sites'] = str(all_sites_file)
            
            # 3-2. 깊이별 시트 형식
            depth_data = {}
            for depth in self.depths:
                depth_df = combined_data[combined_data['FDR_depth'] == depth]
                if not depth_df.empty:
                    # 일평균 계산
                    daily_depth_df = depth_df.groupby(['Date', 'id'], as_index=False)['theta_v'].mean()
                    pivoted = daily_depth_df.pivot(index='Date', columns='id', values='theta_v')
                    depth_data[f"{depth}cm"] = pivoted
                    
            if depth_data:
                depths_file = output_path / f"{station_id}_FDR_daily_depths.xlsx"
                self.file_handler.save_multiple_sheets(depth_data, str(depths_file))
                output_files['daily_depths'] = str(depths_file)
                
        return output_files
        
    def get_processing_summary(self, all_data: List[pd.DataFrame]) -> Dict:
        """처리 결과 요약 정보 생성"""
        if not all_data:
            return {}
            
        combined_data = pd.concat(all_data, ignore_index=True)
        
        summary = {
            'total_records': len(combined_data),
            'sensors_processed': combined_data['id'].nunique(),
            'depths_processed': sorted(combined_data['FDR_depth'].unique()),
            'date_range': {
                'start': combined_data['Date'].min(),
                'end': combined_data['Date'].max()
            },
            'data_quality': {
                'missing_values': combined_data['theta_v'].isna().sum(),
                'valid_records': combined_data['theta_v'].notna().sum()
            },
            'sensor_list': sorted(combined_data['id'].unique())
        }
        
        return summary


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트용 설정
    test_station_config = {
        'station_info': {'id': 'HC'},
        'sensor_configuration': {'depths': [10, 30, 60]},
        'data_paths': {
            'fdr_folder': 'data/input/HC/fdr/',
            'geo_info_file': 'data/input/HC/geo_locations.xlsx'
        }
    }
    
    test_processing_config = {}
    
    # FDRProcessor 테스트
    from ..core.logger import setup_logger
    
    logger = setup_logger("FDRProcessor_Test")
    processor = FDRProcessor(test_station_config, test_processing_config, logger)
    
    try:
        output_files = processor.process_all_fdr_data("data/output/HC/preprocessed/")
        print("✅ FDR 처리 완료!")
        print("생성된 파일들:")
        for output_type, file_path in output_files.items():
            print(f"  {output_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ FDR 처리 실패: {e}")
        import traceback
        traceback.print_exc()