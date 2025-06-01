# src/core/logger.py

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class CRNPLogger:
    """CRNP 시스템용 간소화된 로깅 클래스"""
    
    def __init__(self, 
                 name: str = "CRNP",
                 level: str = "INFO",
                 log_dir: str = "logs",
                 save_to_file: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir)
        self.save_to_file = save_to_file
        
        # 로그 디렉토리 생성
        if save_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # 포매터 설정
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러 추가
        self._add_console_handler()
        
        # 파일 핸들러 추가
        if save_to_file:
            self._add_file_handler(max_file_size, backup_count)
            
        # 프로세스 컨텍스트
        self.process_context = {}
        
    def _add_console_handler(self):
        """콘솔 핸들러 추가"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        # 콘솔용 간단한 포매터
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def _add_file_handler(self, max_file_size: int, backup_count: int):
        """파일 핸들러 추가"""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
    def set_context(self, **kwargs):
        """로깅 컨텍스트 설정"""
        self.process_context.update(kwargs)
        
    def clear_context(self):
        """로깅 컨텍스트 초기화"""
        self.process_context.clear()
        
    def _format_message(self, message: str) -> str:
        """컨텍스트 정보를 포함한 메시지 포맷팅"""
        if self.process_context:
            context_str = " | ".join([f"{k}={v}" for k, v in self.process_context.items()])
            return f"[{context_str}] {message}"
        return message
        
    def debug(self, message: str, **kwargs):
        """DEBUG 레벨 로깅"""
        self.logger.debug(self._format_message(message), **kwargs)
        
    def info(self, message: str, **kwargs):
        """INFO 레벨 로깅"""
        self.logger.info(self._format_message(message), **kwargs)
        
    def warning(self, message: str, **kwargs):
        """WARNING 레벨 로깅"""
        self.logger.warning(self._format_message(message), **kwargs)
        
    def error(self, message: str, **kwargs):
        """ERROR 레벨 로깅"""
        self.logger.error(self._format_message(message), **kwargs)
        
    def critical(self, message: str, **kwargs):
        """CRITICAL 레벨 로깅"""
        self.logger.critical(self._format_message(message), **kwargs)
        
    def log_process_start(self, process_name: str, **details):
        """프로세스 시작 로깅 - 주요 프로세스만"""
        if any(keyword in process_name.lower() for keyword in ['preprocessing', 'calibration', 'calculation', 'validation']):
            self.set_context(process=process_name)
            self.info(f"Starting {process_name}")
        
    def log_process_end(self, process_name: str, duration: Optional[float] = None, **results):
        """프로세스 종료 로깅 - 주요 프로세스만"""
        if any(keyword in process_name.lower() for keyword in ['preprocessing', 'calibration', 'calculation', 'validation']):
            duration_str = f" ({duration:.1f}s)" if duration else ""
            self.info(f"Completed {process_name}{duration_str}")
        
    def log_data_summary(self, data_type: str, count: int, **metadata):
        """데이터 요약 로깅 - 중요한 데이터만"""
        if any(keyword in data_type.lower() for keyword in ['fdr', 'crnp', 'matched', 'calibration', 'daily']):
            self.info(f"{data_type}: {count} records")
        
    def log_file_operation(self, operation: str, file_path: str, status: str = "success", **details):
        """파일 작업 로깅 - 저장 작업만"""
        if operation == "save" and status == "success":
            filename = os.path.basename(file_path)
            self.info(f"Saved: {filename}")
        
    def log_calibration_result(self, N0: float, metrics: Dict[str, Any]):
        """캘리브레이션 결과 로깅"""
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        self.info(f"Calibration result: N0={N0:.1f}, R²={r2:.3f}, RMSE={rmse:.4f}")
        
    def log_validation_result(self, metrics: Dict[str, Any]):
        """검증 결과 로깅"""
        r2 = metrics.get('R2', 0)
        rmse = metrics.get('RMSE', 0)
        self.info(f"Validation result: R²={r2:.3f}, RMSE={rmse:.4f}")
        
    def log_error_with_context(self, error: Exception, context: str = ""):
        """에러와 컨텍스트 정보 로깅"""
        error_msg = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
        self.error(error_msg)
        
    def create_station_logger(self, station_id: str) -> 'CRNPLogger':
        """관측소별 로거 생성"""
        station_log_dir = self.log_dir / station_id
        station_logger = CRNPLogger(
            name=f"{self.name}_{station_id}",
            level=logging.getLevelName(self.level),
            log_dir=str(station_log_dir),
            save_to_file=self.save_to_file
        )
        station_logger.set_context(station_id=station_id)
        return station_logger


class ProcessTimer:
    """프로세스 실행 시간 측정을 위한 컨텍스트 매니저"""
    
    def __init__(self, logger: CRNPLogger, process_name: str, **details):
        self.logger = logger
        self.process_name = process_name
        self.details = details
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log_process_start(self.process_name, **self.details)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log_process_end(self.process_name, duration=duration)
        else:
            self.logger.log_error_with_context(exc_val, f"{self.process_name}")


def setup_logger(name: str = "CRNP", 
                config: Optional[Dict[str, Any]] = None) -> CRNPLogger:
    """로거 설정 및 초기화"""
    if config is None:
        config = {
            'level': 'INFO',
            'save_to_file': True,
            'log_dir': 'logs'
        }
    
    logger = CRNPLogger(
        name=name,
        level=config.get('level', 'INFO'),
        log_dir=config.get('log_dir', 'logs'),
        save_to_file=config.get('save_to_file', True)
    )
    
    return logger