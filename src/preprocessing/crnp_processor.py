# src/preprocessing/crnp_processor.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler
from ..preprocessing.data_validator import DataValidator


class CRNPProcessor:
    """CRNP (우주선 중성자 탐지기) 데이터 전처리 클래스"""
    
    def __init__(self, station_config: Dict, processing_config: Dict, 
                 logger: Optional[CRNPLogger] = None):
        self.station_config = station_config
        self.processing_config = processing_config
        self.logger = logger or CRNPLogger("CRNPProcessor")
        
        # 종속 모듈 초기화
        self.file_handler = FileHandler(self.logger)
        self.validator = DataValidator(self.logger)
        
        # CRNP 데이터 표준 컬럼명
        self.standard_columns = [
            'Timestamp', 'RN', 'Ta', 'RH', 'Pa', 
            'WS', 'WS_max', 'WD_VCT', 'N_counts'
        ]
        
        # TOA5 컬럼 매핑 (더 포괄적인 매핑)
        self.toa5_column_mapping = {
            # 표준 TOA5 컬럼명
            'TIMESTAMP': 'Timestamp',
            'RECORD': 'RN',
            'Air_Temp_Avg': 'Ta',
            'RH_Avg': 'RH', 
            'Air_Press_Avg': 'Pa',
            'WS_avg_Avg': 'WS',
            'WS_max_Max': 'WS_max',
            'WD_VCT': 'WD_VCT',
            'HI_NeutronCts_Tot': 'N_counts',
            
            # 대안적 TOA5 컬럼명들
            'AirTC_Avg': 'Ta',
            'RH': 'RH',
            'BP_hPa_Avg': 'Pa',
            'WS_ms_Avg': 'WS',
            'WindDir_D1_WVT': 'WD_VCT',
            'Wind_Direction': 'WD_VCT',
            'NeutronCounts_Tot': 'N_counts',
            'CRD_Tot': 'N_counts',
            'CRNP_Tot': 'N_counts',
            
            # 축약형들 (row 2에서 나타날 수 있는)
            'TS': 'Timestamp',  # Timestamp 축약형
            'RN': 'RN',         # Record Number 축약형
            'DegC': 'Ta',       # 온도 단위
            '%': 'RH',          # 습도 단위
            'hPa': 'Pa',        # 기압 단위
            'm/s': 'WS',        # 풍속 단위
            'Deg': 'WD_VCT',    # 풍향 단위
        }
        
        # 컬럼별 설명
        self.column_descriptions = {
            'Timestamp': '시간',
            'RN': '레코드 번호',
            'Ta': '기온 (°C)',
            'RH': '상대습도 (%)',
            'Pa': '기압 (hPa)',
            'WS': '풍속 (m/s)',
            'WS_max': '최대풍속 (m/s)',
            'WD_VCT': '풍향 (도)',
            'N_counts': '중성자 카운트'
        }
        
    def process_crnp_data(self, output_dir: str) -> Dict[str, str]:
        """CRNP 데이터 전체 처리"""
        
        with ProcessTimer(self.logger, "CRNP Data Processing",
                         station=self.station_config['station_info']['id']):
            
            # 1. CRNP 파일 자동 탐지
            crnp_file = self._discover_crnp_file()
            
            # 2. 데이터 읽기 (TOA5 형식 전용)
            raw_df = self._read_crnp_file(crnp_file)
            
            # 3. 기본 데이터 품질 확인
            if len(raw_df) == 0:
                raise ValueError("No data found in CRNP file")
                
            if 'Timestamp' not in raw_df.columns:
                raise ValueError("Timestamp column not found after processing")
                
            # 중성자 카운트가 없어도 경고만 하고 계속 진행
            if raw_df['N_counts'].isna().all():
                self.logger.warning("⚠️ No neutron counts data found - CRNP functionality will be limited")
                self.logger.warning("This may indicate:")
                self.logger.warning("  1. Missing neutron detector column in TOA5 file")
                self.logger.warning("  2. Column naming mismatch")
                self.logger.warning("  3. Data collection issue")
                self.logger.warning("Continuing with available meteorological data only...")
            else:
                neutron_count = raw_df['N_counts'].notna().sum()
                self.logger.info(f"✅ Found {neutron_count} neutron count records")
            
            # 4. 데이터 전처리
            processed_df = self._preprocess_crnp_data(raw_df)
            
            # 5. 출력 파일 생성
            output_files = self._generate_outputs(processed_df, output_dir)
            
            # 6. 처리 요약
            self._log_processing_summary(processed_df)
            
            return output_files
            
    def _discover_crnp_file(self) -> str:
        """CRNP 데이터 파일 자동 탐지"""
        crnp_folder = self.station_config['data_paths']['crnp_folder']
        
        if not os.path.exists(crnp_folder):
            raise FileNotFoundError(f"CRNP folder not found: {crnp_folder}")
            
        # 지원하는 파일 형식들 탐지
        excel_files = self.file_handler.discover_files(crnp_folder, "*.xlsx")
        csv_files = self.file_handler.discover_files(crnp_folder, "*.csv")
        
        all_files = excel_files + csv_files
        
        if not all_files:
            raise FileNotFoundError(f"No CRNP data files found in {crnp_folder}")
            
        # CRNP 관련 파일 찾기
        crnp_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path).lower()
            if any(keyword in filename for keyword in ['crnp', 'neutron', 'hourly']):
                crnp_files.append(file_path)
                
        if not crnp_files:
            # CRNP 키워드가 없으면 첫 번째 파일 사용
            self.logger.warning("No files with CRNP keywords found, using first available file")
            crnp_files = [all_files[0]]
            
        if len(crnp_files) > 1:
            self.logger.warning(f"Multiple CRNP files found, using: {crnp_files[0]}")
            
        selected_file = crnp_files[0]
        self.logger.info(f"Selected CRNP file: {os.path.basename(selected_file)}")
        
        return selected_file
        
    def _read_crnp_file(self, file_path: str) -> pd.DataFrame:
        """CRNP 파일 읽기 (TOA5 형식 전용)"""
        
        try:
            # 파일 정보 로깅
            file_info = self.file_handler.get_file_info(file_path)
            self.logger.log_file_operation("read", file_path, "attempting", 
                                         size_mb=file_info['size_mb'])
            
            # 파일 형식 감지 및 읽기
            file_path_obj = Path(file_path)
            
            if file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                df = self._read_excel_toa5(file_path)
            elif file_path_obj.suffix.lower() in ['.csv', '.txt']:
                df = self._read_csv_toa5(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
                
            # 기본 데이터 정보 로깅
            if len(df) > 0:
                self.logger.info(f"Successfully read CRNP file: {len(df)} records, {len(df.columns)} columns")
                self.logger.info(f"Final columns: {list(df.columns)}")
            else:
                self.logger.warning("Empty dataframe after reading CRNP file")
            
            self.logger.log_data_summary(
                "CRNP_Raw", len(df),
                columns=len(df.columns),
                file_size_mb=file_info['size_mb']
            )
            
            return df
            
        except Exception as e:
            self.logger.log_error_with_context(e, f"Reading CRNP file {file_path}")
            raise
            
    def _read_excel_toa5(self, file_path: str) -> pd.DataFrame:
        """Excel TOA5 파일 읽기 - 정확한 구조 반영"""
        
        # 1. 전체 헤더 구조 분석 (처음 6행)
        header_df = pd.read_excel(file_path, header=None, nrows=6)
        
        self.logger.info("Analyzing TOA5 header structure:")
        for i in range(min(6, len(header_df))):
            row_data = header_df.iloc[i, :5].tolist()  # 처음 5개 컬럼만
            self.logger.info(f"  Row {i}: {row_data}")
        
        # 2. TOA5 형식 확인
        if len(header_df) < 4 or 'TOA5' not in str(header_df.iloc[0, 0]):
            raise ValueError("Not a valid TOA5 format file")
            
        self.logger.info("Processing Excel TOA5 format")
        
        # 3. 실제 컬럼명 추출 (행 1, 0-based) - 대문자 컬럼명들
        actual_columns = []
        if len(header_df) > 1:
            row1_data = header_df.iloc[1, :].tolist()
            # NaN이 아닌 값들만 컬럼명으로 사용
            actual_columns = [str(col) for col in row1_data if pd.notna(col)]
            self.logger.info(f"TOA5 columns from row 1: {actual_columns}")
        
        # 4. 행 2의 축약형도 확인 (TS, RN, DegC 등)
        abbreviations = []
        if len(header_df) > 2:
            row2_data = header_df.iloc[2, :].tolist()
            abbreviations = [str(abbr) for abbr in row2_data if pd.notna(abbr)]
            self.logger.info(f"TOA5 abbreviations from row 2: {abbreviations}")
        
        # 5. 데이터 부분 읽기 (4행 스킵: 0=TOA5메타, 1=컬럼명, 2=축약형, 3=공란)
        data_df = pd.read_excel(file_path, skiprows=4)
        
        self.logger.info(f"Data shape after reading: {data_df.shape}")
        self.logger.info(f"Original data columns: {list(data_df.columns)}")
        
        # 6. 컬럼명 설정
        if actual_columns and len(actual_columns) <= len(data_df.columns):
            # 실제 컬럼명이 있고 데이터 컬럼 수보다 적거나 같으면
            final_columns = actual_columns + [f"Col_{i}" for i in range(len(actual_columns), len(data_df.columns))]
            data_df.columns = final_columns[:len(data_df.columns)]
            self.logger.info(f"Applied TOA5 column names: {list(data_df.columns)}")
        else:
            # 컬럼명 추출에 실패했으면 기존 컬럼명 유지
            self.logger.warning("Failed to extract proper column names, keeping original")
            
        # 7. 데이터 샘플 로깅 (디버깅용)
        if len(data_df) > 0:
            self.logger.info("Sample data (first 3 rows):")
            for i in range(min(3, len(data_df))):
                sample_row = data_df.iloc[i, :min(5, len(data_df.columns))].tolist()
                self.logger.info(f"  Data row {i}: {sample_row}")
        
        # 8. 표준 CRNP 컬럼으로 매핑
        mapped_df = self._map_toa5_to_standard(data_df)
        
        return mapped_df
        
    def _read_csv_toa5(self, file_path: str) -> pd.DataFrame:
        """CSV TOA5 파일 읽기 - 정확한 구조 반영"""
        
        encoding = self.file_handler.detect_encoding(str(file_path))
        
        # 1. 전체 헤더 구조 분석 (처음 6행)
        header_df = pd.read_csv(file_path, encoding=encoding, header=None, nrows=6)
        
        self.logger.info("Analyzing CSV TOA5 header structure:")
        for i in range(min(6, len(header_df))):
            row_data = header_df.iloc[i, :5].tolist()  # 처음 5개 컬럼만
            self.logger.info(f"  Row {i}: {row_data}")
        
        # 2. TOA5 형식 확인
        if len(header_df) < 4 or 'TOA5' not in str(header_df.iloc[0, 0]):
            raise ValueError("Not a valid TOA5 format file")
            
        self.logger.info("Processing CSV TOA5 format")
        
        # 3. 실제 컬럼명 추출 (행 1, 0-based) - 대문자 컬럼명들
        actual_columns = []
        if len(header_df) > 1:
            row1_data = header_df.iloc[1, :].tolist()
            # NaN이 아닌 값들만 컬럼명으로 사용
            actual_columns = [str(col) for col in row1_data if pd.notna(col)]
            self.logger.info(f"TOA5 columns from row 1: {actual_columns}")
        
        # 4. 행 2의 축약형도 확인 (TS, RN, DegC 등)
        abbreviations = []
        if len(header_df) > 2:
            row2_data = header_df.iloc[2, :].tolist()
            abbreviations = [str(abbr) for abbr in row2_data if pd.notna(abbr)]
            self.logger.info(f"TOA5 abbreviations from row 2: {abbreviations}")
        
        # 5. 데이터 부분 읽기 (4행 스킵: 0=TOA5메타, 1=컬럼명, 2=축약형, 3=공란)
        data_df = pd.read_csv(file_path, encoding=encoding, skiprows=4)
        
        self.logger.info(f"Data shape after reading: {data_df.shape}")
        self.logger.info(f"Original data columns: {list(data_df.columns)}")
        
        # 6. 컬럼명 설정
        if actual_columns and len(actual_columns) <= len(data_df.columns):
            # 실제 컬럼명이 있고 데이터 컬럼 수보다 적거나 같으면
            final_columns = actual_columns + [f"Col_{i}" for i in range(len(actual_columns), len(data_df.columns))]
            data_df.columns = final_columns[:len(data_df.columns)]
            self.logger.info(f"Applied TOA5 column names: {list(data_df.columns)}")
        else:
            # 컬럼명 추출에 실패했으면 기존 컬럼명 유지
            self.logger.warning("Failed to extract proper column names, keeping original")
            
        # 7. 데이터 샘플 로깅 (디버깅용)
        if len(data_df) > 0:
            self.logger.info("Sample data (first 3 rows):")
            for i in range(min(3, len(data_df))):
                sample_row = data_df.iloc[i, :min(5, len(data_df.columns))].tolist()
                self.logger.info(f"  Data row {i}: {sample_row}")
        
        # 8. 표준 CRNP 컬럼으로 매핑
        mapped_df = self._map_toa5_to_standard(data_df)
        
        return mapped_df
        
    def _map_toa5_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """TOA5 컬럼을 표준 CRNP 컬럼으로 매핑 - 개선된 버전"""
        
        self.logger.info("Mapping TOA5 columns to standard CRNP format")
        self.logger.info(f"Available columns: {list(df.columns)}")
        
        # 새 데이터프레임 생성 (표준 컬럼명으로)
        mapped_df = pd.DataFrame()
        
        # 각 표준 컬럼에 대해 매핑 찾기
        for standard_col in self.standard_columns:
            mapped_value = None
            source_col = None
            
            # 1. 직접 매핑 확인 (정확한 이름)
            for toa5_col, std_col in self.toa5_column_mapping.items():
                if std_col == standard_col and toa5_col in df.columns:
                    mapped_value = df[toa5_col]
                    source_col = toa5_col
                    break
                    
            # 2. 부분 매칭 (대소문자 무시하고 부분 문자열 포함)
            if mapped_value is None:
                for col in df.columns:
                    if self._is_matching_column(col, standard_col):
                        mapped_value = df[col]
                        source_col = col
                        break
                        
            # 3. 특별한 경우 처리 (중성자 카운트)
            if mapped_value is None and standard_col == 'N_counts':
                # 중성자 관련 키워드로 찾기
                neutron_keywords = ['neutron', 'counts', 'hi_neutron', 'crnp', 'cosmic', 'count']
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in neutron_keywords):
                        # 숫자 데이터인지 확인
                        try:
                            numeric_data = pd.to_numeric(df[col], errors='coerce')
                            if numeric_data.notna().sum() > 0:
                                mapped_value = df[col]
                                source_col = col
                                self.logger.info(f"Found potential neutron column: {col}")
                                break
                        except:
                            continue
                            
            # 4. 위치 기반 매핑 (마지막 수단)
            if mapped_value is None and standard_col == 'N_counts':
                # 마지막 컬럼이 중성자 카운트일 가능성이 높음
                if len(df.columns) > 0:
                    last_col = df.columns[-1]
                    try:
                        numeric_data = pd.to_numeric(df[last_col], errors='coerce')
                        if numeric_data.notna().sum() > 0 and numeric_data.mean() > 10:  # 중성자 카운트는 보통 큰 값
                            mapped_value = df[last_col]
                            source_col = f"{last_col} (position-based)"
                            self.logger.info(f"Using last column as neutron counts: {last_col}")
                    except:
                        pass
                        
            # 매핑 결과 적용
            if mapped_value is not None:
                mapped_df[standard_col] = mapped_value
                self.logger.info(f"  ✅ {standard_col} ← {source_col}")
            else:
                mapped_df[standard_col] = np.nan
                self.logger.warning(f"  ❌ {standard_col} ← (missing)")
                
        # 매핑 결과 요약 및 특별 처리
        mapped_count = sum(1 for col in self.standard_columns if mapped_df[col].notna().any())
        self.logger.info(f"Mapping complete: {mapped_count}/{len(self.standard_columns)} columns mapped")
        
        # 중성자 카운트가 없는 경우 특별 로깅
        if mapped_df['N_counts'].isna().all():
            self.logger.error("⚠️ CRITICAL: No neutron counts column found!")
            self.logger.error("Available columns for analysis:")
            for i, col in enumerate(df.columns):
                sample_values = df[col].head(3).tolist()
                self.logger.error(f"  [{i}] '{col}': {sample_values}")
                
            # 수동 매핑 시도 (모든 숫자 컬럼 확인)
            self.logger.error("Attempting manual neutron detection...")
            for col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if numeric_data.notna().sum() > 0:
                        mean_val = numeric_data.mean()
                        std_val = numeric_data.std()
                        min_val = numeric_data.min()
                        max_val = numeric_data.max()
                        self.logger.error(f"  Numeric column '{col}': mean={mean_val:.1f}, std={std_val:.1f}, range=[{min_val:.1f}, {max_val:.1f}]")
                        
                        # 중성자 카운트 특성 확인 (보통 100-10000 범위, 높은 변동성)
                        if 50 < mean_val < 50000 and std_val > mean_val * 0.1:
                            self.logger.error(f"  👆 '{col}' might be neutron counts (will use as fallback)")
                            mapped_df['N_counts'] = df[col]
                            break
                except:
                    continue
        else:
            # 중성자 카운트 데이터 요약
            neutron_data = mapped_df['N_counts'].dropna()
            if len(neutron_data) > 0:
                self.logger.info(f"✅ Neutron counts found: mean={neutron_data.mean():.1f}, range=[{neutron_data.min():.1f}, {neutron_data.max():.1f}]")
        
        return mapped_df
        
    def _is_matching_column(self, col_name: str, standard_col: str) -> bool:
        """컬럼명 매칭 확인 - 개선된 버전"""
        
        col_lower = str(col_name).lower().replace('_', '').replace('-', '')
        
        # 표준 컬럼별 키워드 매칭 (우선순위 순)
        matching_patterns = {
            'Timestamp': [
                ['timestamp', 'time'],
                ['date'],
                ['ts']  # TOA5에서 자주 사용
            ],
            'RN': [
                ['record'],
                ['rn'],
                ['rec']
            ],
            'Ta': [
                ['airtemp', 'air_temp'],
                ['temp', 'temperature'],
                ['ta'],
                ['degc']  # 단위명도 확인
            ],
            'RH': [
                ['rh', 'relativehumidity'],
                ['humidity', 'humid'],
                ['%']  # 단위로도 확인
            ],
            'Pa': [
                ['airpress', 'air_press', 'pressure'],
                ['pa', 'press'],
                ['hpa']  # 단위명
            ],
            'WS': [
                ['windspeed', 'wsavg'],
                ['ws', 'wind'],
                ['m/s']  # 단위명
            ],
            'WS_max': [
                ['windmax', 'wsmax'],
                ['maxwind', 'windgust']
            ],
            'WD_VCT': [
                ['winddir', 'wd'],
                ['direction', 'dir'],
                ['deg']  # 단위명
            ],
            'N_counts': [
                ['neutron', 'cosmic'],
                ['counts', 'count', 'cnt'],
                ['hi', 'crnp'],
                ['tot', 'total']  # TOA5에서 _Tot 자주 사용
            ]
        }
        
        if standard_col in matching_patterns:
            # 우선순위별로 매칭 확인
            for pattern_group in matching_patterns[standard_col]:
                if all(pattern in col_lower for pattern in pattern_group):
                    return True
                    
            # 단일 키워드 매칭
            for pattern_group in matching_patterns[standard_col]:
                for pattern in pattern_group:
                    if pattern in col_lower:
                        return True
                        
        return False
            
    def _preprocess_crnp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRNP 데이터 전처리"""
        
        processed_df = df.copy()
        
        self.logger.info("CRNP data preprocessing started")
        self.logger.info(f"Original data shape: {processed_df.shape}")
        
        # 1. 타임스탬프 처리
        with ProcessTimer(self.logger, "Timestamp Processing"):
            
            if 'Timestamp' in processed_df.columns:
                # 이미 datetime 객체인지 확인
                sample_timestamp = processed_df['Timestamp'].iloc[0] if len(processed_df) > 0 else None
                
                if isinstance(sample_timestamp, (pd.Timestamp, datetime)):
                    self.logger.info("Timestamp already in datetime format")
                    processed_df['timestamp'] = processed_df['Timestamp']
                else:
                    self.logger.info("Converting timestamp from string/numeric format")
                    processed_df['timestamp'] = pd.to_datetime(processed_df['Timestamp'], errors='coerce')
                    
                # 유효하지 않은 타임스탬프 확인
                invalid_timestamps = processed_df['timestamp'].isna().sum()
                if invalid_timestamps > 0:
                    self.logger.warning(f"Found {invalid_timestamps} invalid timestamps")
                    
                # 유효한 타임스탬프만 유지
                initial_count = len(processed_df)
                processed_df = processed_df.dropna(subset=['timestamp'])
                removed_count = initial_count - len(processed_df)
                
                if removed_count > 0:
                    self.logger.warning(f"Removed {removed_count} records with invalid timestamps")
                    
                # 최종 타임스탬프 검증
                if len(processed_df) > 0:
                    final_date_range = f"{processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}"
                    self.logger.info(f"Final timestamp range: {final_date_range}")
            else:
                raise ValueError("Timestamp column not found")
                
        # 2. 수치 데이터 변환
        numeric_columns = ['RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                original_values = processed_df[col].notna().sum()
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                converted_values = processed_df[col].notna().sum()
                
                if original_values != converted_values:
                    self.logger.warning(f"Lost {original_values - converted_values} values during numeric conversion in {col}")
                    
        # 3. 중성자 카운트 특별 처리
        if 'N_counts' in processed_df.columns:
            # 0 이하 값을 NaN으로 변경
            invalid_neutrons = (processed_df['N_counts'] <= 0).sum()
            if invalid_neutrons > 0:
                self.logger.warning(f"Found {invalid_neutrons} non-positive neutron counts, setting to NaN")
                processed_df.loc[processed_df['N_counts'] <= 0, 'N_counts'] = np.nan
                
        # 4. 기상 데이터 범위 확인 및 수정
        processed_df = self._apply_range_limits(processed_df)
        
        # 5. 시간 순서 정렬
        processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(processed_df) > 0:
            self.logger.log_data_summary(
                "CRNP_Processed", len(processed_df),
                date_range=f"{processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}"
            )
        else:
            self.logger.error("No data remaining after preprocessing")
        
        return processed_df
        
    def _apply_range_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """물리적 범위 제한 적용"""
        
        range_limits = {
            'Ta': (-40, 50),      # 기온
            'RH': (0, 100),       # 상대습도
            'Pa': (800, 1100),    # 기압
            'WS': (0, 50),        # 풍속
            'WS_max': (0, 50),    # 최대풍속
            'WD_VCT': (0, 360),   # 풍향
        }
        
        processed_df = df.copy()
        
        for column, (min_val, max_val) in range_limits.items():
            if column in processed_df.columns:
                out_of_range = ((processed_df[column] < min_val) | 
                               (processed_df[column] > max_val))
                out_of_range_count = out_of_range.sum()
                
                if out_of_range_count > 0:
                    self.logger.warning(f"Found {out_of_range_count} out-of-range values in {column}, setting to NaN")
                    processed_df.loc[out_of_range, column] = np.nan
                    
        return processed_df
        
    def _generate_outputs(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """출력 파일 생성"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        station_id = self.station_config['station_info']['id']
        output_files = {}
        
        # 1. 전처리된 CRNP 데이터 (입력 형식)
        with ProcessTimer(self.logger, "Generating CRNP input file"):
            input_file = output_path / f"{station_id}_CRNP_input.xlsx"
            self.file_handler.save_dataframe(df, str(input_file), index=False)
            output_files['input_format'] = str(input_file)
            
        # 2. 품질 보고서
        with ProcessTimer(self.logger, "Generating quality report"):
            quality_report = self._generate_quality_report(df)
            report_file = output_path / f"{station_id}_CRNP_quality_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(quality_report)
                
            output_files['quality_report'] = str(report_file)
            
        return output_files
        
    def _generate_quality_report(self, df: pd.DataFrame) -> str:
        """품질 보고서 생성"""
        
        lines = []
        lines.append("=" * 60)
        lines.append("CRNP DATA QUALITY REPORT")
        lines.append("=" * 60)
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Station: {self.station_config['station_info']['name']}")
        lines.append(f"Total Records: {len(df)}")
        
        if 'timestamp' in df.columns and len(df) > 0:
            lines.append(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
        lines.append("")
        lines.append("DATA COMPLETENESS:")
        lines.append("-" * 30)
        
        total_records = len(df)
        for col in self.standard_columns[1:]:  # Timestamp 제외
            if col in df.columns:
                valid_count = df[col].notna().sum()
                completeness = (valid_count / total_records) * 100 if total_records > 0 else 0
                desc = self.column_descriptions.get(col, col)
                lines.append(f"{desc:15} ({col:8}): {completeness:6.2f}% ({valid_count:5}/{total_records})")
                
        lines.append("")
        lines.append("DATA SUMMARY STATISTICS:")
        lines.append("-" * 30)
        
        # 주요 변수들의 통계
        numeric_columns = ['Ta', 'RH', 'Pa', 'N_counts']
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                try:
                    desc = self.column_descriptions.get(col, col)
                    stats = df[col].describe()
                    lines.append(f"{desc} ({col}):")
                    lines.append(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                    lines.append(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                    lines.append("")
                except Exception as e:
                    lines.append(f"{desc} ({col}): Statistics calculation failed - {e}")
                    lines.append("")
            elif col == 'N_counts':
                lines.append("중성자 카운트 (N_counts): ❌ 데이터 없음")
                lines.append("  경고: CRNP 기능이 제한됩니다.")
                lines.append("")
                
        lines.append("=" * 60)
        return "\n".join(lines)
        
    def _log_processing_summary(self, df: pd.DataFrame) -> None:
        """처리 요약 로깅"""
        
        summary = {
            'total_records': len(df),
            'date_range': "Unknown",
            'completeness': {}
        }
        
        if 'timestamp' in df.columns and len(df) > 0:
            summary['date_range'] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        
        # 데이터 완성도 계산
        for col in self.standard_columns[1:]:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
                summary['completeness'][col] = round(completeness, 1)
                
        self.logger.info(f"CRNP processing summary: {summary['total_records']} records, {summary['date_range']}")
        
        # 주요 변수들의 완성도 로깅
        key_variables = ['N_counts', 'Ta', 'Pa', 'RH']
        for var in key_variables:
            if var in summary['completeness']:
                if var == 'N_counts' and summary['completeness'][var] == 0:
                    self.logger.warning(f"  {var}: {summary['completeness'][var]}% complete ⚠️ MISSING - CRNP functionality limited")
                else:
                    self.logger.info(f"  {var}: {summary['completeness'][var]}% complete")
                    
        # 중성자 카운트 누락 시 추가 안내
        if summary['completeness'].get('N_counts', 0) == 0:
            self.logger.warning("⚠️ IMPORTANT: No neutron count data processed")
            self.logger.warning("  - Calibration will not be possible")
            self.logger.warning("  - Only meteorological data is available")
            self.logger.warning("  - Check original TOA5 file for neutron detector columns")


# 사용 예시
if __name__ == "__main__":
    # 테스트용 설정
    test_station_config = {
        'station_info': {'id': 'PC', 'name': 'Pyeongchang Station'},
        'data_paths': {
            'crnp_folder': 'data/input/PC/crnp/'
        }
    }
    
    test_processing_config = {}
    
    # CRNPProcessor 테스트
    from ..core.logger import setup_logger
    
    logger = setup_logger("CRNPProcessor_Test")
    processor = CRNPProcessor(test_station_config, test_processing_config, logger)
    
    try:
        output_files = processor.process_crnp_data("data/output/PC/preprocessed/")
        print("✅ CRNP 처리 완료!")
        print("생성된 파일들:")
        for output_type, file_path in output_files.items():
            print(f"  {output_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ CRNP 처리 실패: {e}")
        import traceback
        traceback.print_exc()