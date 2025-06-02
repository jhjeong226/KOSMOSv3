# src/utils/file_handler.py

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import glob
import chardet

from ..core.logger import CRNPLogger


class FileHandler:
    """파일 입출력 및 패턴 매칭을 담당하는 클래스"""
    
    def __init__(self, logger: Optional[CRNPLogger] = None):
        self.logger = logger or CRNPLogger("FileHandler")
        
        # 지원하는 파일 확장자
        self.supported_extensions = {
            'excel': ['.xlsx', '.xls'],
            'csv': ['.csv', '.txt'],
            'data': ['.dat']
        }
        
        # 파일 패턴 정의 (ConfigManager와 동일)
        self.file_patterns = {
            'PC': r'z6-(\d+)\([^)]+\)\(z6-\1\)-Configuration.*\.csv$',
            'HC': r'HC-[^(]+\(z6-(\d+)\)\(z6-\1\)-Configuration.*\.csv$',
            'SWCR': r'z6-\d+\s+[^(]+\(z6-(\d+)\)-Configuration.*\.csv$',
            'unified': r'z6-\d+.*?\(z6-(\d+)\).*?-Configuration.*\.csv$',
            'crnp': r'.*CRNP.*\.(xlsx|csv)$'
        }
        
    def discover_files(self, directory: str, pattern: str = "*", 
                      recursive: bool = False) -> List[str]:
        """디렉토리에서 패턴에 맞는 파일들을 자동 탐지"""
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []
            
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
            
        file_paths = [str(f) for f in files if f.is_file()]
        
        self.logger.info(f"Discovered {len(file_paths)} files in {directory}")
        return sorted(file_paths)
        
    def match_fdr_files(self, directory: str, station_type: str = 'unified') -> Dict[str, List[str]]:
        """FDR 센서 파일들을 패턴별로 매칭"""
        csv_files = self.discover_files(directory, "*.csv")
        
        pattern = self.file_patterns.get(station_type, self.file_patterns['unified'])
        
        matched_files = []
        unmatched_files = []
        loc_keys = {}
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            match = re.search(pattern, file_name)
            
            if match:
                loc_key = match.group(1)
                matched_files.append(file_path)
                loc_keys[file_path] = loc_key
                self.logger.debug(f"Matched FDR file: {file_name} (loc_key: {loc_key})")
            else:
                unmatched_files.append(file_path)
                self.logger.debug(f"Unmatched file: {file_name}")
                
        self.logger.info(f"FDR file matching: {len(matched_files)} matched, {len(unmatched_files)} unmatched")
        
        return {
            'matched': matched_files,
            'unmatched': unmatched_files,
            'loc_keys': loc_keys
        }
        
    def detect_encoding(self, file_path: str) -> str:
        """파일 인코딩 자동 감지"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 첫 10KB만 읽어서 인코딩 감지
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding or 'utf-8'
                
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
            
    def read_fdr_file(self, file_path: str, skip_rows: int = 2) -> pd.DataFrame:
        """FDR 센서 파일 읽기 (CSV/Excel 자동 감지)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # 파일 확장자에 따른 읽기
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, skiprows=skip_rows)
                self.logger.debug(f"Read Excel file: {file_path}")
                
            elif file_path.suffix.lower() in ['.csv', '.txt']:
                encoding = self.detect_encoding(str(file_path))
                df = pd.read_csv(file_path, skiprows=skip_rows, encoding=encoding)
                self.logger.debug(f"Read CSV file: {file_path}")
                
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            # 기본 데이터 검증
            self._validate_fdr_data(df, str(file_path))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise
            
    def read_crnp_file(self, file_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """CRNP 데이터 파일 읽기"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                encoding = self.detect_encoding(str(file_path))
                df = pd.read_csv(file_path, encoding=encoding)
                
            # 컬럼명 지정
            if columns:
                if len(columns) <= len(df.columns):
                    df.columns = columns + list(df.columns[len(columns):])
                else:
                    self.logger.warning(f"Column count mismatch: expected {len(columns)}, got {len(df.columns)}")
                    
            self._validate_crnp_data(df, str(file_path))
            
            self.logger.info(f"Read CRNP file: {file_path} ({len(df)} records)")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read CRNP file {file_path}: {str(e)}")
            raise
            
    def save_dataframe(self, df: pd.DataFrame, file_path: str, 
                      index: bool = True, **kwargs) -> None:
        """DataFrame을 파일로 저장"""
        file_path = Path(file_path)
        
        # 디렉토리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=index, **kwargs)
            elif file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=index, encoding='utf-8-sig', **kwargs)
            else:
                raise ValueError(f"Unsupported output format: {file_path.suffix}")
                
            self.logger.info(f"Saved file: {file_path} ({len(df)} records)")
            
        except Exception as e:
            self.logger.error(f"Failed to save file {file_path}: {str(e)}")
            raise
            
    def save_multiple_sheets(self, data_dict: Dict[str, pd.DataFrame], 
                           file_path: str) -> None:
        """여러 DataFrame을 Excel 시트로 저장"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
                    
            self.logger.info(f"Saved multi-sheet Excel: {file_path} ({len(data_dict)} sheets)")
            
        except Exception as e:
            self.logger.error(f"Failed to save multi-sheet Excel {file_path}: {str(e)}")
            raise
            
    def backup_file(self, file_path: str, backup_dir: str = "backup") -> str:
        """파일 백업"""
        file_path = Path(file_path)
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"File backed up: {file_path} -> {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to backup file {file_path}: {str(e)}")
            raise
            
    def get_file_info(self, file_path: str) -> Dict[str, Union[str, int, float]]:
        """파일 정보 추출"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {}
            
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path.absolute()),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': file_path.suffix.lower()
        }
        
    def _validate_fdr_data(self, df: pd.DataFrame, file_path: str) -> None:
        """FDR 데이터 기본 검증"""
        required_patterns = [
            'Timestamps',
            'Water Content'
        ]
        
        columns = df.columns.tolist()
        column_str = ' '.join(columns)
        
        missing_patterns = []
        for pattern in required_patterns:
            if not any(pattern in col for col in columns):
                missing_patterns.append(pattern)
                
        if missing_patterns:
            self.logger.warning(f"Missing expected columns in {file_path}: {missing_patterns}")
            
        # 행 수 확인
        if len(df) == 0:
            self.logger.warning(f"Empty dataframe in {file_path}")
        elif len(df) < 10:
            self.logger.warning(f"Very few records in {file_path}: {len(df)} rows")
            
        self.logger.debug(f"FDR data validation complete: {file_path} ({len(df)} rows, {len(columns)} columns)")
        
    def _validate_crnp_data(self, df: pd.DataFrame, file_path: str) -> None:
        """CRNP 데이터 기본 검증"""
        if len(df) == 0:
            self.logger.warning(f"Empty CRNP dataframe in {file_path}")
            return
            
        # 최소 컬럼 수 확인
        if len(df.columns) < 5:
            self.logger.warning(f"CRNP data has too few columns: {len(df.columns)}")
            
        self.logger.debug(f"CRNP data validation complete: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        
    def extract_loc_key_from_filename(self, file_path: str, 
                                    pattern_type: str = 'unified') -> Optional[str]:
        """파일명에서 loc_key 추출"""
        file_name = os.path.basename(file_path)
        pattern = self.file_patterns.get(pattern_type, self.file_patterns['unified'])
        
        match = re.search(pattern, file_name)
        if match:
            return match.group(1)
        return None
        
    def organize_files_by_date(self, file_list: List[str]) -> Dict[str, List[str]]:
        """파일들을 수정 날짜별로 그룹화"""
        date_groups = {}
        
        for file_path in file_list:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    mtime = datetime.fromtimestamp(file_path_obj.stat().st_mtime)
                    date_key = mtime.strftime("%Y-%m-%d")
                    
                    if date_key not in date_groups:
                        date_groups[date_key] = []
                    date_groups[date_key].append(file_path)
                    
            except Exception as e:
                self.logger.warning(f"Could not get date for {file_path}: {e}")
                
        return date_groups


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    
    # 로거 설정
    logger = setup_logger("FileHandler_Test")
    
    # FileHandler 인스턴스 생성
    file_handler = FileHandler(logger)
    
    # 파일 탐지 테스트
    test_dir = "data/input/HC/fdr"
    fdr_files = file_handler.match_fdr_files(test_dir, "HC")
    
    print(f"매칭된 FDR 파일: {len(fdr_files['matched'])}개")
    print(f"매칭되지 않은 파일: {len(fdr_files['unmatched'])}개")
    
    # loc_key 추출 테스트
    test_filename = "HC-E1(z6-19850)(z6-19850)-Configuration 2-1726742190.6906087.csv"
    loc_key = file_handler.extract_loc_key_from_filename(test_filename, "HC")
    print(f"추출된 loc_key: {loc_key}")
    
    print("FileHandler 구현 완료!")