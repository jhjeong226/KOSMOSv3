# src/preprocessing/data_validator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats

from ..core.logger import CRNPLogger


class DataValidator:
    """데이터 품질 검증 및 이상값 탐지 클래스"""
    
    def __init__(self, logger: Optional[CRNPLogger] = None):
        self.logger = logger or CRNPLogger("DataValidator")
        
        # 기본 임계값 설정
        self.thresholds = {
            'theta_v': {'min': 0.0, 'max': 0.8},      # 체적수분함량 범위
            'temperature': {'min': -40, 'max': 50},    # 기온 범위 (°C)
            'humidity': {'min': 0, 'max': 100},        # 상대습도 범위 (%)
            'pressure': {'min': 800, 'max': 1100},     # 기압 범위 (hPa)
            'neutron_counts': {'min': 0, 'max': 10000}, # 중성자 카운트 범위
            'wind_speed': {'min': 0, 'max': 50},       # 풍속 범위 (m/s)
            'wind_direction': {'min': 0, 'max': 360}   # 풍향 범위 (도)
        }
        
        # 이상값 탐지 방법별 매개변수
        self.outlier_params = {
            'mad': {'threshold': 3},           # Median Absolute Deviation
            'iqr': {'multiplier': 1.5},        # Interquartile Range
            'zscore': {'threshold': 3},        # Z-score
            'isolation_forest': {'contamination': 0.1}
        }
        
    def validate_fdr_data(self, df: pd.DataFrame, sensor_id: str = "") -> Dict[str, Any]:
        """FDR 센서 데이터 종합 검증"""
        validation_results = {
            'sensor_id': sensor_id,
            'total_records': len(df),
            'validation_timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        self.logger.info(f"Starting FDR data validation for {sensor_id}")
        
        # 1. 기본 구조 검증
        structure_issues = self._validate_fdr_structure(df)
        validation_results['issues'].extend(structure_issues)
        
        # 2. 날짜/시간 검증
        datetime_issues = self._validate_datetime_column(df, 'Date')
        validation_results['issues'].extend(datetime_issues)
        
        # 3. 토양수분 데이터 검증
        theta_columns = [col for col in df.columns if 'theta_v' in col or 'Water Content' in col]
        for col in theta_columns:
            if col in df.columns:
                theta_issues = self._validate_theta_v_data(df, col)
                validation_results['issues'].extend(theta_issues)
                
        # 4. 데이터 연속성 검증
        continuity_issues = self._validate_data_continuity(df, 'Date')
        validation_results['issues'].extend(continuity_issues)
        
        # 5. 이상값 탐지
        for col in theta_columns:
            if col in df.columns:
                outlier_issues = self._detect_outliers(df, col, method='mad')
                validation_results['issues'].extend(outlier_issues)
                
        # 검증 결과 요약
        validation_results['severity_counts'] = self._count_issues_by_severity(validation_results['issues'])
        validation_results['is_valid'] = len([i for i in validation_results['issues'] if i['severity'] == 'critical']) == 0
        
        self.logger.info(f"FDR validation complete: {len(validation_results['issues'])} issues found")
        return validation_results
        
    def validate_crnp_data(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """CRNP 데이터 종합 검증"""
        validation_results = {
            'total_records': len(df),
            'validation_timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        self.logger.info("Starting CRNP data validation")
        
        # 컬럼명 매핑 (표준화)
        column_mapping = {
            'Timestamp': 'timestamp',
            'RN': 'rain',
            'Ta': 'temperature', 
            'RH': 'humidity',
            'Pa': 'pressure',
            'WS': 'wind_speed',
            'WS_max': 'wind_speed_max',
            'WD_VCT': 'wind_direction',
            'N_counts': 'neutron_counts'
        }
        
        if len(columns) == len(df.columns):
            df.columns = columns
            
        # 1. 기본 구조 검증
        structure_issues = self._validate_crnp_structure(df, columns)
        validation_results['issues'].extend(structure_issues)
        
        # 2. 날짜/시간 검증  
        datetime_issues = self._validate_datetime_column(df, columns[0])
        validation_results['issues'].extend(datetime_issues)
        
        # 3. 각 변수별 범위 검증
        for i, col in enumerate(columns[1:], 1):  # 첫 번째는 timestamp이므로 제외
            if i < len(df.columns):
                standard_name = column_mapping.get(col, col)
                range_issues = self._validate_variable_range(df, df.columns[i], standard_name)
                validation_results['issues'].extend(range_issues)
                
        # 4. 중성자 카운트 특별 검증
        if 'N_counts' in columns:
            neutron_col_idx = columns.index('N_counts')
            if neutron_col_idx < len(df.columns):
                neutron_issues = self._validate_neutron_counts(df, df.columns[neutron_col_idx])
                validation_results['issues'].extend(neutron_issues)
                
        # 5. 기상 데이터 상관관계 검증
        weather_issues = self._validate_weather_correlations(df, columns)
        validation_results['issues'].extend(weather_issues)
        
        validation_results['severity_counts'] = self._count_issues_by_severity(validation_results['issues'])
        validation_results['is_valid'] = len([i for i in validation_results['issues'] if i['severity'] == 'critical']) == 0
        
        self.logger.info(f"CRNP validation complete: {len(validation_results['issues'])} issues found")
        return validation_results
        
    def _validate_fdr_structure(self, df: pd.DataFrame) -> List[Dict]:
        """FDR 데이터 구조 검증"""
        issues = []
        
        # 필수 컬럼 존재 확인
        required_patterns = ['Timestamps', 'Water Content']
        columns = df.columns.tolist()
        
        for pattern in required_patterns:
            if not any(pattern in col for col in columns):
                issues.append({
                    'type': 'missing_column',
                    'severity': 'critical',
                    'message': f"Required column pattern '{pattern}' not found",
                    'column': pattern
                })
                
        # Water Content 컬럼 개수 확인 (보통 3개 깊이)
        water_content_cols = [col for col in columns if 'Water Content' in col]
        if len(water_content_cols) < 3:
            issues.append({
                'type': 'insufficient_sensors',
                'severity': 'warning',
                'message': f"Expected 3 water content sensors, found {len(water_content_cols)}",
                'found_columns': water_content_cols
            })
            
        return issues
        
    def _validate_crnp_structure(self, df: pd.DataFrame, expected_columns: List[str]) -> List[Dict]:
        """CRNP 데이터 구조 검증"""
        issues = []
        
        # 컬럼 개수 확인
        if len(df.columns) != len(expected_columns):
            issues.append({
                'type': 'column_count_mismatch',
                'severity': 'warning',
                'message': f"Expected {len(expected_columns)} columns, found {len(df.columns)}",
                'expected': expected_columns,
                'found': len(df.columns)
            })
            
        return issues
        
    def _validate_datetime_column(self, df: pd.DataFrame, datetime_col: str) -> List[Dict]:
        """날짜/시간 컬럼 검증"""
        issues = []
        
        if datetime_col not in df.columns:
            issues.append({
                'type': 'missing_datetime_column',
                'severity': 'critical',
                'message': f"Datetime column '{datetime_col}' not found",
                'column': datetime_col
            })
            return issues
            
        # 날짜 형식 변환 시도
        try:
            datetime_series = pd.to_datetime(df[datetime_col], errors='coerce')
            null_count = datetime_series.isna().sum()
            
            if null_count > 0:
                issues.append({
                    'type': 'invalid_datetime_format',
                    'severity': 'warning' if null_count < len(df) * 0.1 else 'critical',
                    'message': f"Found {null_count} invalid datetime values in {datetime_col}",
                    'column': datetime_col,
                    'count': null_count
                })
                
            # 날짜 범위 확인
            valid_dates = datetime_series.dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                # 미래 날짜 확인
                if max_date > datetime.now():
                    issues.append({
                        'type': 'future_dates',
                        'severity': 'warning',
                        'message': f"Found future dates in {datetime_col}. Latest: {max_date}",
                        'column': datetime_col,
                        'latest_date': max_date.isoformat()
                    })
                    
                # 너무 오래된 날짜 확인
                if min_date < datetime.now() - timedelta(days=365*5):  # 5년 이전
                    issues.append({
                        'type': 'very_old_dates',
                        'severity': 'warning',
                        'message': f"Found very old dates in {datetime_col}. Earliest: {min_date}",
                        'column': datetime_col,
                        'earliest_date': min_date.isoformat()
                    })
                    
        except Exception as e:
            issues.append({
                'type': 'datetime_processing_error',
                'severity': 'critical',
                'message': f"Error processing datetime column {datetime_col}: {str(e)}",
                'column': datetime_col
            })
            
        return issues
        
    def _validate_theta_v_data(self, df: pd.DataFrame, column: str) -> List[Dict]:
        """토양수분 데이터 검증"""
        issues = []
        
        if column not in df.columns:
            return issues
            
        data = pd.to_numeric(df[column], errors='coerce')
        
        # 범위 검증
        threshold = self.thresholds['theta_v']
        out_of_range = (data < threshold['min']) | (data > threshold['max'])
        out_of_range_count = out_of_range.sum()
        
        if out_of_range_count > 0:
            issues.append({
                'type': 'value_out_of_range',
                'severity': 'warning' if out_of_range_count < len(data) * 0.05 else 'critical',
                'message': f"Found {out_of_range_count} values outside valid range [{threshold['min']}, {threshold['max']}] in {column}",
                'column': column,
                'count': out_of_range_count,
                'percentage': round(out_of_range_count / len(data) * 100, 2)
            })
            
        # NaN 값 확인
        nan_count = data.isna().sum()
        if nan_count > 0:
            issues.append({
                'type': 'missing_values',
                'severity': 'warning' if nan_count < len(data) * 0.1 else 'critical',
                'message': f"Found {nan_count} missing values in {column}",
                'column': column,
                'count': nan_count,
                'percentage': round(nan_count / len(data) * 100, 2)
            })
            
        return issues
        
    def _validate_variable_range(self, df: pd.DataFrame, column: str, variable_type: str) -> List[Dict]:
        """변수별 범위 검증"""
        issues = []
        
        if column not in df.columns or variable_type not in self.thresholds:
            return issues
            
        data = pd.to_numeric(df[column], errors='coerce')
        threshold = self.thresholds[variable_type]
        
        out_of_range = (data < threshold['min']) | (data > threshold['max'])
        out_of_range_count = out_of_range.sum()
        
        if out_of_range_count > 0:
            issues.append({
                'type': 'value_out_of_range',
                'severity': 'warning',
                'message': f"Found {out_of_range_count} values outside expected range [{threshold['min']}, {threshold['max']}] in {variable_type}",
                'column': column,
                'variable_type': variable_type,
                'count': out_of_range_count
            })
            
        return issues
        
    def _validate_neutron_counts(self, df: pd.DataFrame, column: str) -> List[Dict]:
        """중성자 카운트 특별 검증"""
        issues = []
        
        if column not in df.columns:
            return issues
            
        data = pd.to_numeric(df[column], errors='coerce')
        
        # 0 또는 음수 값 확인
        non_positive = (data <= 0)
        non_positive_count = non_positive.sum()
        
        if non_positive_count > 0:
            issues.append({
                'type': 'non_positive_neutron_counts',
                'severity': 'critical',
                'message': f"Found {non_positive_count} non-positive neutron count values",
                'column': column,
                'count': non_positive_count
            })
            
        # 급격한 변화 감지
        if len(data) > 1:
            diff = data.diff().abs()
            mean_diff = diff.mean()
            large_changes = diff > (mean_diff + 3 * diff.std())
            large_changes_count = large_changes.sum()
            
            if large_changes_count > len(data) * 0.01:  # 1% 이상
                issues.append({
                    'type': 'sudden_neutron_changes',
                    'severity': 'warning',
                    'message': f"Found {large_changes_count} sudden changes in neutron counts",
                    'column': column,
                    'count': large_changes_count
                })
                
        return issues
        
    def _validate_data_continuity(self, df: pd.DataFrame, datetime_col: str) -> List[Dict]:
        """데이터 연속성 검증"""
        issues = []
        
        if datetime_col not in df.columns:
            return issues
            
        try:
            dates = pd.to_datetime(df[datetime_col], errors='coerce').dropna().sort_values()
            
            if len(dates) < 2:
                return issues
                
            # 시간 간격 분석
            time_diffs = dates.diff()[1:]  # 첫 번째 NaT 제외
            
            # 가장 일반적인 간격 찾기
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                expected_interval = mode_diff.iloc[0]
                
                # 예상 간격보다 긴 간격 찾기 (데이터 공백)
                long_gaps = time_diffs[time_diffs > expected_interval * 2]
                
                if len(long_gaps) > 0:
                    issues.append({
                        'type': 'data_gaps',
                        'severity': 'warning',
                        'message': f"Found {len(long_gaps)} data gaps longer than expected interval",
                        'column': datetime_col,
                        'gap_count': len(long_gaps),
                        'expected_interval': str(expected_interval),
                        'max_gap': str(long_gaps.max())
                    })
                    
        except Exception as e:
            issues.append({
                'type': 'continuity_check_error',
                'severity': 'warning',
                'message': f"Error checking data continuity: {str(e)}",
                'column': datetime_col
            })
            
        return issues
        
    def _validate_weather_correlations(self, df: pd.DataFrame, columns: List[str]) -> List[Dict]:
        """기상 데이터 상관관계 검증"""
        issues = []
        
        # 온도와 습도 역상관 확인 (일반적으로 온도가 높으면 상대습도는 낮음)
        if 'Ta' in columns and 'RH' in columns:
            try:
                ta_idx = columns.index('Ta')
                rh_idx = columns.index('RH')
                
                if ta_idx < len(df.columns) and rh_idx < len(df.columns):
                    ta_data = pd.to_numeric(df.iloc[:, ta_idx], errors='coerce')
                    rh_data = pd.to_numeric(df.iloc[:, rh_idx], errors='coerce')
                    
                    valid_mask = ~(ta_data.isna() | rh_data.isna())
                    if valid_mask.sum() > 10:  # 최소 10개 유효 데이터
                        correlation = ta_data[valid_mask].corr(rh_data[valid_mask])
                        
                        if correlation > 0.3:  # 강한 양의 상관관계는 이상
                            issues.append({
                                'type': 'unexpected_weather_correlation',
                                'severity': 'warning',
                                'message': f"Unexpected positive correlation between temperature and humidity: {correlation:.3f}",
                                'correlation': correlation
                            })
                            
            except Exception as e:
                self.logger.debug(f"Weather correlation check failed: {e}")
                
        return issues
        
    def _detect_outliers(self, df: pd.DataFrame, column: str, 
                        method: str = 'mad', **kwargs) -> List[Dict]:
        """이상값 탐지"""
        issues = []
        
        if column not in df.columns:
            return issues
            
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if len(data) == 0:
            return issues
            
        outliers = []
        
        if method == 'mad':
            # Median Absolute Deviation
            threshold = kwargs.get('threshold', self.outlier_params['mad']['threshold'])
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            
            if mad > 0:
                mad_scores = np.abs(data - median) / mad
                outliers = data[mad_scores > threshold]
                
        elif method == 'iqr':
            # Interquartile Range
            multiplier = kwargs.get('multiplier', self.outlier_params['iqr']['multiplier'])
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            # Z-score
            threshold = kwargs.get('threshold', self.outlier_params['zscore']['threshold'])
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > threshold]
            
        if len(outliers) > 0:
            outlier_percentage = len(outliers) / len(data) * 100
            severity = 'critical' if outlier_percentage > 5 else 'warning'
            
            issues.append({
                'type': 'outliers_detected',
                'severity': severity,
                'message': f"Detected {len(outliers)} outliers in {column} using {method} method",
                'column': column,
                'method': method,
                'outlier_count': len(outliers),
                'percentage': round(outlier_percentage, 2),
                'outlier_range': f"[{outliers.min():.3f}, {outliers.max():.3f}]"
            })
            
        return issues
        
    def _count_issues_by_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """심각도별 이슈 개수 집계"""
        severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'info')
            if severity in severity_counts:
                severity_counts[severity] += 1
                
        return severity_counts
        
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """검증 결과 보고서 생성"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Validation Date: {validation_results['validation_timestamp']}")
        report_lines.append(f"Total Records: {validation_results['total_records']}")
        
        if 'sensor_id' in validation_results:
            report_lines.append(f"Sensor ID: {validation_results['sensor_id']}")
            
        report_lines.append(f"Overall Status: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
        report_lines.append("")
        
        # 심각도별 요약
        severity_counts = validation_results['severity_counts']
        report_lines.append("Issue Summary:")
        report_lines.append(f"  Critical: {severity_counts['critical']}")
        report_lines.append(f"  Warning:  {severity_counts['warning']}")
        report_lines.append(f"  Info:     {severity_counts['info']}")
        report_lines.append("")
        
        # 상세 이슈 목록
        if validation_results['issues']:
            report_lines.append("Detailed Issues:")
            report_lines.append("-" * 40)
            
            for i, issue in enumerate(validation_results['issues'], 1):
                severity_marker = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'warning' else "🔵"
                report_lines.append(f"{i}. {severity_marker} [{issue['severity'].upper()}] {issue['type']}")
                report_lines.append(f"   {issue['message']}")
                if 'column' in issue:
                    report_lines.append(f"   Column: {issue['column']}")
                report_lines.append("")
        else:
            report_lines.append("No issues found! ✅")
            
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


# 사용 예시
if __name__ == "__main__":
    from ..core.logger import setup_logger
    
    # 로거 설정
    logger = setup_logger("DataValidator_Test")
    
    # DataValidator 인스턴스 생성
    validator = DataValidator(logger)
    
    # 테스트 데이터 생성
    test_fdr_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100, freq='H'),
        'theta_v_d1': np.random.normal(0.25, 0.05, 100),
        'theta_v_d2': np.random.normal(0.30, 0.05, 100),
        'theta_v_d3': np.random.normal(0.35, 0.05, 100)
    })
    
    # FDR 데이터 검증
    validation_result = validator.validate_fdr_data(test_fdr_data, "TEST_SENSOR")
    
    # 보고서 생성
    report = validator.generate_validation_report(validation_result)
    print(report)
    
    print("DataValidator 구현 완료!")