# src/preprocessing/crnp_processor.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
        
        # 컬럼별 설명
        self.column_descriptions = {
            'Timestamp': '시간',
            'RN': '레코드 번호',  # Record Number
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
            
            # 2. 데이터 읽기
            raw_df = self._read_crnp_file(crnp_file)
            
            # 3. 데이터 검증
            self._validate_crnp_data(raw_df)
            
            # 4. 데이터 전처리
            processed_df = self._preprocess_crnp_data(raw_df)
            
            # 5. 품질 관리
            cleaned_df = self._apply_quality_control(processed_df)
            
            # 6. 출력 파일 생성
            output_files = self._generate_outputs(cleaned_df, output_dir)
            
            # 7. 처리 요약
            self._log_processing_summary(cleaned_df)
            
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
        """CRNP 파일 읽기"""
        
        try:
            # 파일 정보 로깅
            file_info = self.file_handler.get_file_info(file_path)
            self.logger.log_file_operation("read", file_path, "attempting", 
                                         size_mb=file_info['size_mb'])
            
            # 파일 읽기
            df = self.file_handler.read_crnp_file(file_path, self.standard_columns)
            
            self.logger.log_data_summary(
                "CRNP_Raw", len(df),
                columns=len(df.columns),
                file_size_mb=file_info['size_mb']
            )
            
            return df
            
        except Exception as e:
            self.logger.log_error_with_context(e, f"Reading CRNP file {file_path}")
            raise
            
    def _validate_crnp_data(self, df: pd.DataFrame) -> None:
        """CRNP 데이터 검증"""
        
        with ProcessTimer(self.logger, "CRNP Data Validation"):
            validation_result = self.validator.validate_crnp_data(df, self.standard_columns)
            
            # 심각한 문제가 있는지 확인
            critical_issues = [i for i in validation_result['issues'] 
                             if i['severity'] == 'critical']
            
            if critical_issues:
                self.logger.error(f"Critical validation issues found: {len(critical_issues)}")
                for issue in critical_issues:
                    self.logger.error(f"  - {issue['message']}")
                    
                # 심각한 문제가 있어도 처리 계속 (로깅만)
                self.logger.warning("Continuing processing despite critical issues")
                
            # 검증 결과 로깅
            severity_counts = validation_result['severity_counts']
            self.logger.info(f"Validation complete: {severity_counts['critical']} critical, "
                           f"{severity_counts['warning']} warnings")
                           
    def _preprocess_crnp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRNP 데이터 전처리"""
        
        processed_df = df.copy()
        
        # 1. 컬럼명 표준화 (이미 read_crnp_file에서 처리됨)
        if len(processed_df.columns) >= len(self.standard_columns):
            processed_df.columns = self.standard_columns + list(processed_df.columns[len(self.standard_columns):])
        
        # 2. 타임스탬프 처리
        with ProcessTimer(self.logger, "Timestamp Processing"):
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
                
        # 3. 수치 데이터 변환
        numeric_columns = ['RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                original_values = processed_df[col].notna().sum()
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                converted_values = processed_df[col].notna().sum()
                
                if original_values != converted_values:
                    self.logger.warning(f"Lost {original_values - converted_values} values during numeric conversion in {col}")
                    
        # 4. 중성자 카운트 특별 처리
        if 'N_counts' in processed_df.columns:
            # 0 이하 값을 NaN으로 변경
            invalid_neutrons = (processed_df['N_counts'] <= 0).sum()
            if invalid_neutrons > 0:
                self.logger.warning(f"Found {invalid_neutrons} non-positive neutron counts, setting to NaN")
                processed_df.loc[processed_df['N_counts'] <= 0, 'N_counts'] = np.nan
                
        # 5. 기상 데이터 범위 확인 및 수정
        processed_df = self._apply_range_limits(processed_df)
        
        # 6. 시간 순서 정렬
        processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.log_data_summary(
            "CRNP_Processed", len(processed_df),
            date_range=f"{processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}"
        )
        
        return processed_df
        
    def _apply_range_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """물리적 범위 제한 적용 (RN은 Record Number이므로 제외)"""
        
        # 범위 제한값 (RN 제외)
        range_limits = {
            'Ta': (-40, 50),      # 기온
            'RH': (0, 100),       # 상대습도
            'Pa': (800, 1100),    # 기압
            'WS': (0, 50),        # 풍속
            'WS_max': (0, 50),    # 최대풍속
            'WD_VCT': (0, 360),   # 풍향
            # RN은 Record Number이므로 범위 제한 없음
        }
        
        processed_df = df.copy()
        
        for column, (min_val, max_val) in range_limits.items():
            if column in processed_df.columns:
                # 범위 밖 값 개수 확인
                out_of_range = ((processed_df[column] < min_val) | 
                               (processed_df[column] > max_val))
                out_of_range_count = out_of_range.sum()
                
                if out_of_range_count > 0:
                    self.logger.warning(f"Found {out_of_range_count} out-of-range values in {column}, setting to NaN")
                    processed_df.loc[out_of_range, column] = np.nan
                    
        return processed_df
        
    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """품질 관리 적용"""
        
        cleaned_df = df.copy()
        
        with ProcessTimer(self.logger, "Quality Control"):
            
            # 1. 이상값 탐지 및 제거 (중성자 카운트)
            if 'N_counts' in cleaned_df.columns:
                cleaned_df = self._remove_neutron_outliers(cleaned_df)
                
            # 2. 데이터 연속성 확인
            self._check_data_continuity(cleaned_df)
            
            # 3. 기상 데이터 상관관계 확인
            self._check_weather_consistency(cleaned_df)
            
            # 4. 최종 품질 통계
            self._calculate_quality_metrics(cleaned_df)
            
        return cleaned_df
        
    def _remove_neutron_outliers(self, df: pd.DataFrame, method: str = 'mad', threshold: float = 3.0) -> pd.DataFrame:
        """중성자 카운트 이상값 제거"""
        
        if 'N_counts' not in df.columns:
            return df
            
        neutron_data = df['N_counts'].dropna()
        
        if len(neutron_data) == 0:
            return df
            
        # MAD (Median Absolute Deviation) 방법 사용
        median = np.median(neutron_data)
        mad = np.median(np.abs(neutron_data - median))
        
        if mad > 0:
            mad_scores = np.abs(neutron_data - median) / mad
            outlier_mask = mad_scores > threshold
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                self.logger.warning(f"Removing {outlier_count} neutron outliers using MAD method (threshold={threshold})")
                
                # 이상값을 NaN으로 설정
                cleaned_df = df.copy()
                outlier_indices = neutron_data[outlier_mask].index
                cleaned_df.loc[outlier_indices, 'N_counts'] = np.nan
                
                return cleaned_df
                
        return df
        
    def _check_data_continuity(self, df: pd.DataFrame) -> None:
        """데이터 연속성 확인"""
        
        if 'timestamp' not in df.columns or len(df) < 2:
            return
            
        # 시간 간격 분석
        time_diffs = df['timestamp'].diff()[1:]
        
        # 가장 일반적인 간격 찾기
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) > 0:
            expected_interval = mode_diff.iloc[0]
            
            # 긴 간격 찾기
            long_gaps = time_diffs[time_diffs > expected_interval * 2]
            
            if len(long_gaps) > 0:
                self.logger.warning(f"Found {len(long_gaps)} data gaps longer than expected interval")
                self.logger.info(f"Expected interval: {expected_interval}, Max gap: {long_gaps.max()}")
                
    def _check_weather_consistency(self, df: pd.DataFrame) -> None:
        """기상 데이터 일관성 확인"""
        
        # 온도와 습도 관계 확인
        if 'Ta' in df.columns and 'RH' in df.columns:
            valid_mask = df['Ta'].notna() & df['RH'].notna()
            
            if valid_mask.sum() > 10:
                correlation = df.loc[valid_mask, 'Ta'].corr(df.loc[valid_mask, 'RH'])
                self.logger.info(f"Temperature-Humidity correlation: {correlation:.3f}")
                
                if correlation > 0.3:
                    self.logger.warning("Unexpected positive correlation between temperature and humidity")
                    
        # 풍속과 최대풍속 관계 확인
        if 'WS' in df.columns and 'WS_max' in df.columns:
            inconsistent = (df['WS'] > df['WS_max']).sum()
            if inconsistent > 0:
                self.logger.warning(f"Found {inconsistent} cases where WS > WS_max")
                
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> None:
        """데이터 품질 지표 계산"""
        
        total_records = len(df)
        quality_metrics = {}
        
        for col in self.standard_columns[1:]:  # Timestamp 제외
            if col in df.columns:
                valid_count = df[col].notna().sum()
                completeness = (valid_count / total_records) * 100
                quality_metrics[col] = {
                    'completeness': round(completeness, 2),
                    'missing_count': total_records - valid_count
                }
                
        # 품질 지표 로깅 (RN은 Record Number로 처리)
        self.logger.info("Data quality metrics:")
        for col, metrics in quality_metrics.items():
            desc = self.column_descriptions.get(col, col)
            
            # RN(Record Number)의 경우 특별한 설명 추가
            if col == 'RN':
                self.logger.info(f"  {desc} ({col}): {metrics['completeness']}% complete, {metrics['missing_count']} missing (레코드 개수)")
            elif metrics['completeness'] < 90:
                self.logger.warning(f"  {desc} ({col}): {metrics['completeness']}% complete, {metrics['missing_count']} missing")
            else:
                self.logger.info(f"  {desc} ({col}): {metrics['completeness']}% complete, {metrics['missing_count']} missing")
            
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
            
        # 2. 일일 평균 데이터
        with ProcessTimer(self.logger, "Generating daily average file"):
            daily_df = self._create_daily_summary(df)
            if not daily_df.empty:
                daily_file = output_path / f"{station_id}_CRNP_daily.xlsx"
                self.file_handler.save_dataframe(daily_df, str(daily_file), index=True)
                output_files['daily_format'] = str(daily_file)
                
        # 3. 품질 보고서
        with ProcessTimer(self.logger, "Generating quality report"):
            quality_report = self._generate_quality_report(df)
            report_file = output_path / f"{station_id}_CRNP_quality_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(quality_report)
                
            output_files['quality_report'] = str(report_file)
            
        return output_files
        
    def _create_daily_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """일일 요약 데이터 생성"""
        
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
            
        # 날짜별 그룹화
        df_daily = df.copy()
        df_daily['date'] = df_daily['timestamp'].dt.date
        
        # 수치 컬럼들만 선택
        numeric_columns = ['RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
        existing_numeric = [col for col in numeric_columns if col in df_daily.columns]
        
        if not existing_numeric:
            return pd.DataFrame()
            
        # 일일 통계 계산
        daily_stats = df_daily.groupby('date')[existing_numeric].agg({
            col: ['mean', 'std', 'min', 'max', 'count'] for col in existing_numeric
        })
        
        # 다중 인덱스 평면화
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        
        return daily_stats
        
    def _generate_quality_report(self, df: pd.DataFrame) -> str:
        """품질 보고서 생성"""
        
        lines = []
        lines.append("=" * 60)
        lines.append("CRNP DATA QUALITY REPORT")
        lines.append("=" * 60)
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Station: {self.station_config['station_info']['name']}")
        lines.append(f"Total Records: {len(df)}")
        
        if 'timestamp' in df.columns:
            lines.append(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
        lines.append("")
        lines.append("DATA COMPLETENESS:")
        lines.append("-" * 30)
        
        total_records = len(df)
        for col in self.standard_columns[1:]:  # Timestamp 제외
            if col in df.columns:
                valid_count = df[col].notna().sum()
                completeness = (valid_count / total_records) * 100
                desc = self.column_descriptions.get(col, col)
                
                # RN(Record Number) 특별 표시
                if col == 'RN':
                    lines.append(f"{desc:15} ({col:8}): {completeness:6.2f}% ({valid_count:5}/{total_records}) [Record Number]")
                else:
                    lines.append(f"{desc:15} ({col:8}): {completeness:6.2f}% ({valid_count:5}/{total_records})")
                
        lines.append("")
        lines.append("DATA SUMMARY STATISTICS:")
        lines.append("-" * 30)
        
        # RN 제외하고 통계 생성
        numeric_columns = ['Ta', 'RH', 'Pa', 'N_counts']
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                desc = self.column_descriptions.get(col, col)
                stats = df[col].describe()
                lines.append(f"{desc} ({col}):")
                lines.append(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                lines.append(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                lines.append("")
        
        # RN(Record Number) 별도 처리
        if 'RN' in df.columns and df['RN'].notna().sum() > 0:
            rn_stats = df['RN'].describe()
            lines.append("레코드 번호 (RN):")
            lines.append(f"  Total Records: {rn_stats['count']:.0f}")
            lines.append(f"  Range: {rn_stats['min']:.0f} - {rn_stats['max']:.0f}")
            lines.append("")
                
        lines.append("=" * 60)
        
        return "\n".join(lines)
        
    def _log_processing_summary(self, df: pd.DataFrame) -> None:
        """처리 요약 로깅"""
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "Unknown",
            'completeness': {}
        }
        
        # 데이터 완성도 계산
        for col in self.standard_columns[1:]:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                summary['completeness'][col] = round(completeness, 1)
                
        self.logger.info(f"CRNP processing summary: {summary['total_records']} records, {summary['date_range']}")
        
        # 주요 변수들의 완성도 로깅
        key_variables = ['N_counts', 'Ta', 'Pa', 'RH']
        for var in key_variables:
            if var in summary['completeness']:
                self.logger.info(f"  {var}: {summary['completeness'][var]}% complete")


# 사용 예시
if __name__ == "__main__":
    # 테스트용 설정
    test_station_config = {
        'station_info': {'id': 'HC', 'name': 'Hongcheon Station'},
        'data_paths': {
            'crnp_folder': 'data/input/HC/crnp/'
        }
    }
    
    test_processing_config = {}
    
    # CRNPProcessor 테스트
    from ..core.logger import setup_logger
    
    logger = setup_logger("CRNPProcessor_Test")
    processor = CRNPProcessor(test_station_config, test_processing_config, logger)
    
    try:
        output_files = processor.process_crnp_data("data/output/HC/preprocessed/")
        print("✅ CRNP 처리 완료!")
        print("생성된 파일들:")
        for output_type, file_path in output_files.items():
            print(f"  {output_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ CRNP 처리 실패: {e}")
        import traceback
        traceback.print_exc()