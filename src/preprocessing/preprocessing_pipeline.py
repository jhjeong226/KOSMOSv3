# src/preprocessing/preprocessing_pipeline.py

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from src.core.config_manager import ConfigManager
from src.core.logger import CRNPLogger, ProcessTimer
from src.preprocessing.fdr_processor import FDRProcessor
from src.preprocessing.crnp_processor import CRNPProcessor


class PreprocessingPipeline:
    """데이터 전처리 통합 파이프라인 - 간소화된 버전"""
    
    def __init__(self, config_root: str = "config"):
        self.config_manager = ConfigManager(config_root)
        self.main_logger = CRNPLogger("PreprocessingPipeline")
        self.results = {}
        
    def run_station_preprocessing(self, station_id: str, 
                                output_base_dir: str = "data/output") -> Dict[str, Any]:
        """단일 관측소 전처리 실행"""
        
        with ProcessTimer(self.main_logger, f"Station {station_id} Preprocessing"):
            
            try:
                # 1. 설정 로드
                station_config, processing_config = self._load_configurations(station_id)
                
                # 2. 관측소별 로거 생성
                station_logger = self.main_logger.create_station_logger(station_id)
                
                # 3. 출력 디렉토리 설정
                output_dir = Path(output_base_dir) / station_id / "preprocessed"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 4. FDR 데이터 처리
                fdr_results = self._process_fdr_data(
                    station_config, processing_config, station_logger, str(output_dir)
                )
                
                # 5. CRNP 데이터 처리
                crnp_results = self._process_crnp_data(
                    station_config, processing_config, station_logger, str(output_dir)
                )
                
                # 6. 결과 통합
                combined_results = self._combine_results(
                    station_id, fdr_results, crnp_results, str(output_dir)
                )
                
                # 7. 메타데이터 저장
                self._save_metadata(combined_results, str(output_dir))
                
                self.results[station_id] = combined_results
                station_logger.info("Preprocessing completed successfully")
                
                return combined_results
                
            except Exception as e:
                self.main_logger.log_error_with_context(e, f"Station {station_id} preprocessing")
                raise
                
    def run_multiple_stations(self, station_ids: List[str], 
                             output_base_dir: str = "data/output",
                             parallel: bool = False) -> Dict[str, Any]:
        """다중 관측소 전처리 실행"""
        
        if parallel and len(station_ids) > 1:
            return self._run_parallel_preprocessing(station_ids, output_base_dir)
        else:
            return self._run_sequential_preprocessing(station_ids, output_base_dir)
            
    def _load_configurations(self, station_id: str) -> tuple:
        """설정 파일들 로드"""
        
        try:
            station_config = self.config_manager.load_station_config(station_id)
            processing_config = self.config_manager.load_processing_config()
            
            # 필수 경로 검증
            self._validate_station_paths(station_config)
            
            return station_config, processing_config
            
        except Exception as e:
            self.main_logger.log_error_with_context(e, f"Loading configuration for {station_id}")
            raise
            
    def _validate_station_paths(self, station_config: Dict) -> None:
        """관측소 경로 유효성 검증"""
        
        paths_to_check = [
            ('fdr_folder', 'FDR data folder'),
            ('crnp_folder', 'CRNP data folder')
        ]
        
        missing_paths = []
        
        for path_key, description in paths_to_check:
            path = station_config['data_paths'].get(path_key)
            if path and not os.path.exists(path):
                missing_paths.append(f"{description}: {path}")
                
        if missing_paths:
            self.main_logger.warning(f"Missing paths: {', '.join(missing_paths)}")
            
    def _process_fdr_data(self, station_config: Dict, processing_config: Dict,
                         logger: CRNPLogger, output_dir: str) -> Dict[str, Any]:
        """FDR 데이터 처리"""
        
        try:
            fdr_processor = FDRProcessor(station_config, processing_config, logger)
            output_files = fdr_processor.process_all_fdr_data(output_dir)
            
            # 처리 요약 생성
            summary = self._get_fdr_summary(output_files.get('input_format'))
            
            logger.info(f"FDR processing completed: {len(output_files)} files")
            
            return {
                'status': 'success',
                'output_files': output_files,
                'summary': summary,
                'processor': 'FDRProcessor'
            }
            
        except Exception as e:
            logger.log_error_with_context(e, "FDR data processing")
            return {
                'status': 'failed',
                'error': str(e),
                'output_files': {},
                'summary': {},
                'processor': 'FDRProcessor'
            }
            
    def _process_crnp_data(self, station_config: Dict, processing_config: Dict,
                          logger: CRNPLogger, output_dir: str) -> Dict[str, Any]:
        """CRNP 데이터 처리"""
        
        try:
            crnp_processor = CRNPProcessor(station_config, processing_config, logger)
            output_files = crnp_processor.process_crnp_data(output_dir)
            
            # 처리 요약 생성
            summary = self._get_crnp_summary(output_files.get('input_format'))
            
            logger.info(f"CRNP processing completed: {len(output_files)} files")
            
            return {
                'status': 'success',
                'output_files': output_files,
                'summary': summary,
                'processor': 'CRNPProcessor'
            }
            
        except Exception as e:
            logger.log_error_with_context(e, "CRNP data processing")
            return {
                'status': 'failed',
                'error': str(e),
                'output_files': {},
                'summary': {},
                'processor': 'CRNPProcessor'
            }
            
    def _get_fdr_summary(self, input_file: Optional[str]) -> Dict:
        """FDR 처리 결과 요약 생성"""
        
        if not input_file or not os.path.exists(input_file):
            return {}
            
        try:
            import pandas as pd
            df = pd.read_excel(input_file)
            
            return {
                'total_records': len(df),
                'sensors': df['id'].nunique() if 'id' in df.columns else 0,
                'depths': sorted(df['FDR_depth'].unique()) if 'FDR_depth' in df.columns else [],
                'date_range': {
                    'start': str(df['Date'].min()) if 'Date' in df.columns else None,
                    'end': str(df['Date'].max()) if 'Date' in df.columns else None
                }
            }
            
        except Exception as e:
            self.main_logger.warning(f"Could not generate FDR summary: {e}")
            return {}
            
    def _get_crnp_summary(self, input_file: Optional[str]) -> Dict:
        """CRNP 처리 결과 요약 생성"""
        
        if not input_file or not os.path.exists(input_file):
            return {}
            
        try:
            import pandas as pd
            df = pd.read_excel(input_file)
            
            summary = {
                'total_records': len(df),
                'date_range': {
                    'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                    'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None
                }
            }
            
            # 주요 변수들의 완성도 계산
            key_columns = ['N_counts', 'Ta', 'RH', 'Pa']
            completeness = {}
            for col in key_columns:
                if col in df.columns:
                    complete_pct = (df[col].notna().sum() / len(df)) * 100
                    completeness[col] = round(complete_pct, 1)
                    
            summary['data_completeness'] = completeness
            return summary
            
        except Exception as e:
            self.main_logger.warning(f"Could not generate CRNP summary: {e}")
            return {}
            
    def _combine_results(self, station_id: str, fdr_results: Dict, 
                        crnp_results: Dict, output_dir: str) -> Dict[str, Any]:
        """FDR과 CRNP 결과 통합"""
        
        combined_results = {
            'station_id': station_id,
            'processing_timestamp': datetime.now().isoformat(),
            'overall_status': 'success',
            'fdr': fdr_results,
            'crnp': crnp_results,
            'output_directory': output_dir
        }
        
        # 전체 상태 결정
        if fdr_results.get('status') == 'failed' or crnp_results.get('status') == 'failed':
            combined_results['overall_status'] = 'partial_success'
            
        if fdr_results.get('status') == 'failed' and crnp_results.get('status') == 'failed':
            combined_results['overall_status'] = 'failed'
            
        return combined_results
        
    def _save_metadata(self, results: Dict, output_dir: str) -> None:
        """처리 메타데이터 저장"""
        
        metadata_file = Path(output_dir) / "preprocessing_metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
            self.main_logger.log_file_operation("save", str(metadata_file), "success")
            
        except Exception as e:
            self.main_logger.log_error_with_context(e, f"Saving metadata")
            
    def _run_sequential_preprocessing(self, station_ids: List[str], 
                                    output_base_dir: str) -> Dict[str, Any]:
        """순차적 다중 관측소 처리"""
        
        results = {}
        
        for station_id in station_ids:
            try:
                station_result = self.run_station_preprocessing(station_id, output_base_dir)
                results[station_id] = station_result
                
            except Exception as e:
                self.main_logger.log_error_with_context(e, f"Processing station {station_id}")
                results[station_id] = {
                    'station_id': station_id,
                    'overall_status': 'failed',
                    'error': str(e)
                }
                
        return results
        
    def _run_parallel_preprocessing(self, station_ids: List[str], 
                                  output_base_dir: str) -> Dict[str, Any]:
        """병렬 다중 관측소 처리"""
        
        import concurrent.futures
        import threading
        
        results = {}
        results_lock = threading.Lock()
        
        def process_station(station_id):
            try:
                result = self.run_station_preprocessing(station_id, output_base_dir)
                with results_lock:
                    results[station_id] = result
            except Exception as e:
                with results_lock:
                    results[station_id] = {
                        'station_id': station_id,
                        'overall_status': 'failed',
                        'error': str(e)
                    }
                    
        max_workers = min(len(station_ids), 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_station, station_id): station_id 
                      for station_id in station_ids}
            
            for future in concurrent.futures.as_completed(futures):
                station_id = futures[future]
                try:
                    future.result()
                    self.main_logger.info(f"Completed {station_id}")
                except Exception as e:
                    self.main_logger.log_error_with_context(e, f"Parallel processing of {station_id}")
                    
        return results
        
    def generate_summary_report(self, output_dir: str = "data/output") -> str:
        """전체 처리 결과 요약 보고서 생성"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("CRNP DATA PREPROCESSING SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Stations Processed: {len(self.results)}")
        lines.append("")
        
        for station_id, result in self.results.items():
            lines.append(f"STATION: {station_id}")
            lines.append("-" * 40)
            lines.append(f"Overall Status: {result.get('overall_status', 'Unknown')}")
            
            # FDR 요약
            fdr_status = result.get('fdr', {}).get('status', 'Unknown')
            lines.append(f"FDR Processing: {fdr_status}")
            
            fdr_summary = result.get('fdr', {}).get('summary', {})
            if fdr_summary:
                lines.append(f"  - Sensors: {fdr_summary.get('sensors', 0)}")
                lines.append(f"  - Records: {fdr_summary.get('total_records', 0)}")
                
            # CRNP 요약
            crnp_status = result.get('crnp', {}).get('status', 'Unknown')
            lines.append(f"CRNP Processing: {crnp_status}")
            
            crnp_summary = result.get('crnp', {}).get('summary', {})
            if crnp_summary:
                lines.append(f"  - Records: {crnp_summary.get('total_records', 0)}")
                
            lines.append("")
            
        lines.append("=" * 80)
        
        report_content = "\n".join(lines)
        
        # 보고서 파일 저장
        report_file = Path(output_dir) / "preprocessing_summary_report.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.main_logger.info(f"Summary report saved")
        except Exception as e:
            self.main_logger.warning(f"Could not save summary report: {e}")
            
        return report_content


# 실행 스크립트
def main():
    """메인 실행 함수"""
    
    print("🚀 CRNP 데이터 전처리 파이프라인 시작")
    print("=" * 60)
    
    # 파이프라인 초기화
    pipeline = PreprocessingPipeline()
    
    # 단일 관측소 처리
    try:
        print("📍 관측소 데이터 처리 중...")
        
        hc_results = pipeline.run_station_preprocessing("HC")
        
        if hc_results['overall_status'] == 'success':
            print("✅ 관측소 처리 완료!")
        else:
            print(f"⚠️  관측소 처리 부분 완료: {hc_results['overall_status']}")
            
        # 결과 출력
        print("\n📊 처리 결과:")
        print(f"  FDR 처리: {hc_results['fdr']['status']}")
        print(f"  CRNP 처리: {hc_results['crnp']['status']}")
        
        # 요약 보고서 생성
        print("\n📋 요약 보고서 생성 중...")
        summary_report = pipeline.generate_summary_report()
        print("요약 보고서가 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    print("🎯 다음 단계: 캘리브레이션 모듈 구현")


if __name__ == "__main__":
    main()