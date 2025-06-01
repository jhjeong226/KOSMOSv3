# src/visualization/visualization_manager.py

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import webbrowser

from ..core.logger import CRNPLogger, ProcessTimer
from ..utils.file_handler import FileHandler
from .neutron_plots import NeutronPlotter
from .soil_moisture_plots import SoilMoisturePlotter
from .validation_plots import ValidationPlotter


class VisualizationManager:
    """CRNP 시각화를 총괄하는 매니저 클래스"""
    
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.logger = CRNPLogger(f"VisualizationManager_{station_id}")
        
        # 기본 경로 설정
        self.output_dir = Path(f"data/output/{station_id}")
        self.viz_dir = self.output_dir / "visualization"
        
        # 플로터 초기화
        self.neutron_plotter = NeutronPlotter(station_id, self.logger)
        self.sm_plotter = SoilMoisturePlotter(station_id, self.logger)
        self.validation_plotter = ValidationPlotter(station_id, self.logger)
        
        # 생성된 플롯 파일들 추적
        self.generated_plots = {}
        
    def generate_all_plots(self, include_validation: bool = True) -> Dict[str, Any]:
        """모든 플롯 생성"""
        
        with ProcessTimer(self.logger, f"Generating all visualizations for {self.station_id}"):
            
            # 출력 디렉토리 생성
            self.viz_dir.mkdir(parents=True, exist_ok=True)
            
            results = {
                'station_id': self.station_id,
                'generation_timestamp': datetime.now().isoformat(),
                'plots_generated': {},
                'html_report': None,
                'total_plots': 0
            }
            
            try:
                # 1. 중성자 관련 플롯
                neutron_plots = self._generate_neutron_plots()
                results['plots_generated']['neutron'] = neutron_plots
                
                # 2. 토양수분 관련 플롯
                sm_plots = self._generate_soil_moisture_plots()
                results['plots_generated']['soil_moisture'] = sm_plots
                
                # 3. 검증 플롯 (선택사항)
                if include_validation:
                    validation_plots = self._generate_validation_plots()
                    results['plots_generated']['validation'] = validation_plots
                    
                # 4. HTML 리포트 생성
                html_report = self._generate_html_report(results['plots_generated'])
                results['html_report'] = html_report
                
                # 5. 총 플롯 수 계산
                total_plots = sum(len(plots) for plots in results['plots_generated'].values())
                results['total_plots'] = total_plots
                
                # 6. 결과 메타데이터 저장
                self._save_visualization_metadata(results)
                
                self.logger.info(f"Generated {total_plots} plots successfully")
                return results
                
            except Exception as e:
                self.logger.log_error_with_context(e, f"Visualization generation for {self.station_id}")
                raise
                
    def _generate_neutron_plots(self) -> Dict[str, str]:
        """중성자 관련 플롯 생성"""
        
        try:
            # CRNP 전처리 데이터 확인
            crnp_file = self.output_dir / "preprocessed" / f"{self.station_id}_CRNP_input.xlsx"
            
            if not crnp_file.exists():
                self.logger.warning("CRNP preprocessed data not found, skipping neutron plots")
                return {}
                
            # 캘리브레이션 결과에서 보정된 중성자 데이터 확인
            calibration_dir = self.output_dir / "calibration"
            calibration_files = list(calibration_dir.glob("*debug_data.xlsx"))
            
            neutron_plots = {}
            
            if calibration_files:
                # 캘리브레이션 디버깅 데이터 사용 (보정 계수 포함)
                cal_debug_file = calibration_files[0]
                cal_data = pd.read_excel(cal_debug_file)
                
                # 간단한 중성자 데이터 구성
                neutron_data = cal_data.copy()
                neutron_data.index = pd.to_datetime(neutron_data['date'])
                
                # 필요한 컬럼 확인 및 생성
                if 'Daily_N' in neutron_data.columns:
                    neutron_data['total_raw_counts'] = neutron_data['Daily_N']
                    neutron_data['total_corrected_neutrons'] = neutron_data['Daily_N']
                    
                # 보정계수는 1.0으로 기본 설정 (실제 값이 없는 경우)
                for factor in ['fi', 'fp', 'fw', 'fb']:
                    if factor not in neutron_data.columns:
                        neutron_data[factor] = 1.0
                        
            else:
                # 전처리 데이터만 사용
                crnp_data = pd.read_excel(crnp_file)
                neutron_data = crnp_data.copy()
                
                if 'timestamp' in neutron_data.columns:
                    neutron_data.index = pd.to_datetime(neutron_data['timestamp'])
                    
                if 'N_counts' in neutron_data.columns:
                    neutron_data['total_raw_counts'] = neutron_data['N_counts']
                    neutron_data['total_corrected_neutrons'] = neutron_data['N_counts']
                    
                # 기본 보정계수
                for factor in ['fi', 'fp', 'fw', 'fb']:
                    neutron_data[factor] = 1.0
                    
            # 플롯 생성
            if len(neutron_data) > 0:
                neutron_plots = self.neutron_plotter.plot_neutron_timeseries(
                    neutron_data, 
                    output_dir=str(self.viz_dir),
                    show_corrections=True
                )
                
                # 통계 플롯 추가
                stats_plot = self.neutron_plotter.plot_neutron_statistics(
                    neutron_data,
                    output_dir=str(self.viz_dir)
                )
                if stats_plot:
                    neutron_plots['neutron_statistics'] = stats_plot
                    
                # 보정 요약 플롯
                summary_plot = self.neutron_plotter.plot_correction_summary(
                    neutron_data,
                    output_dir=str(self.viz_dir)
                )
                if summary_plot:
                    neutron_plots['correction_summary'] = summary_plot
                    
            return neutron_plots
            
        except Exception as e:
            self.logger.warning(f"Failed to generate neutron plots: {e}")
            return {}
            
    def _generate_soil_moisture_plots(self) -> Dict[str, str]:
        """토양수분 관련 플롯 생성"""
        
        try:
            # 토양수분 계산 결과 확인
            sm_file = self.output_dir / "soil_moisture" / f"{self.station_id}_soil_moisture.xlsx"
            
            if not sm_file.exists():
                self.logger.warning("Soil moisture data not found, skipping soil moisture plots")
                return {}
                
            # 데이터 로드
            sm_data = pd.read_excel(sm_file, index_col=0)
            sm_data.index = pd.to_datetime(sm_data.index)
            
            # 기상 데이터 (CRNP 전처리 데이터에서)
            weather_data = None
            crnp_file = self.output_dir / "preprocessed" / f"{self.station_id}_CRNP_input.xlsx"
            
            if crnp_file.exists():
                crnp_data = pd.read_excel(crnp_file)
                if 'timestamp' in crnp_data.columns:
                    crnp_data.index = pd.to_datetime(crnp_data['timestamp'])
                    weather_data = crnp_data[['Ta', 'RH', 'Pa']].dropna()
                    
            # 플롯 생성
            sm_plots = self.sm_plotter.plot_soil_moisture_timeseries(
                sm_data,
                weather_data=weather_data,
                output_dir=str(self.viz_dir)
            )
            
            # 계절별 패턴 분석 (데이터가 충분한 경우)
            if len(sm_data) >= 90:  # 최소 3개월 데이터
                seasonal_plot = self.sm_plotter.plot_seasonal_patterns(
                    sm_data,
                    output_dir=str(self.viz_dir)
                )
                if seasonal_plot:
                    sm_plots['seasonal_patterns'] = seasonal_plot
                    
            return sm_plots
            
        except Exception as e:
            self.logger.warning(f"Failed to generate soil moisture plots: {e}")
            return {}
            
    def _generate_validation_plots(self) -> Dict[str, str]:
        """검증 관련 플롯 생성"""
        
        try:
            # 토양수분 데이터
            sm_file = self.output_dir / "soil_moisture" / f"{self.station_id}_soil_moisture.xlsx"
            
            # FDR 데이터
            fdr_file = self.output_dir / "preprocessed" / f"{self.station_id}_FDR_input.xlsx"
            
            if not sm_file.exists() or not fdr_file.exists():
                self.logger.warning("Missing data for validation plots")
                return {}
                
            # 데이터 로드
            sm_data = pd.read_excel(sm_file, index_col=0)
            sm_data.index = pd.to_datetime(sm_data.index)
            
            fdr_data = pd.read_excel(fdr_file)
            fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
            
            # 검증 플롯 생성
            validation_plots = self.validation_plotter.plot_validation_comparison(
                sm_data,
                fdr_data,
                output_dir=str(self.viz_dir)
            )
            
            return validation_plots
            
        except Exception as e:
            self.logger.warning(f"Failed to generate validation plots: {e}")
            return {}
            
    def _generate_html_report(self, plots_data: Dict[str, Dict[str, str]]) -> str:
        """HTML 리포트 생성"""
        
        try:
            # HTML 템플릿 생성
            html_content = self._create_html_template(plots_data)
            
            # HTML 파일 저장
            html_file = self.viz_dir / f"{self.station_id}_visualization_report.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.log_file_operation("save", str(html_file), "success")
            
            # 상대 경로 반환
            return str(html_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate HTML report: {e}")
            return ""
            
    def _create_html_template(self, plots_data: Dict[str, Dict[str, str]]) -> str:
        """HTML 템플릿 생성"""
        
        # 기본 메타데이터 수집
        station_info = self._get_station_metadata()
        
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRNP 분석 결과 - {self.station_id} 관측소</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section h2 {{
            color: #2E86AB;
            border-bottom: 3px solid #F18F01;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 1.8em;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 5px solid #F18F01;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .metadata-label {{
            font-weight: 600;
            color: #495057;
        }}
        .metadata-value {{
            color: #2E86AB;
            font-weight: 500;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .plot-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .plot-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .plot-card h3 {{
            background: #f8f9fa;
            margin: 0;
            padding: 15px 20px;
            font-size: 1.1em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .plot-card .plot-description {{
            padding: 15px 20px;
            font-size: 0.9em;
            color: #6c757d;
            background: #f8f9fa;
        }}
        .nav-tabs {{
            display: flex;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 20px;
        }}
        .nav-tab {{
            padding: 12px 24px;
            background: #f8f9fa;
            border: none;
            cursor: pointer;
            font-size: 1em;
            color: #495057;
            border-radius: 8px 8px 0 0;
            margin-right: 2px;
            transition: all 0.2s;
        }}
        .nav-tab.active {{
            background: #2E86AB;
            color: white;
        }}
        .nav-tab:hover:not(.active) {{
            background: #e9ecef;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .status-error {{
            background: #f8d7da;
            color: #721c24;
        }}
        @media (max-width: 768px) {{
            .plot-grid {{
                grid-template-columns: 1fr;
            }}
            .nav-tabs {{
                flex-wrap: wrap;
            }}
            .nav-tab {{
                flex: 1;
                min-width: 120px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.station_id} 관측소</h1>
            <p>CRNP 토양수분 분석 결과</p>
            <p>생성일: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</p>
        </div>
        
        <div class="content">
            <!-- 메타데이터 섹션 -->
            <div class="section">
                <h2>📊 분석 개요</h2>
                <div class="metadata">
                    <div class="metadata-grid">
                        {self._generate_metadata_html(station_info)}
                    </div>
                </div>
            </div>
            
            <!-- 탭 네비게이션 -->
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="showTab('neutron')">🛰️ 중성자 분석</button>
                <button class="nav-tab" onclick="showTab('soil-moisture')">💧 토양수분</button>
                <button class="nav-tab" onclick="showTab('validation')">✅ 검증</button>
            </div>
            
            <!-- 중성자 분석 탭 -->
            <div id="neutron" class="tab-content active">
                <div class="section">
                    <h2>🛰️ 중성자 카운트 분석</h2>
                    {self._generate_plots_html(plots_data.get('neutron', {}), 'neutron')}
                </div>
            </div>
            
            <!-- 토양수분 탭 -->
            <div id="soil-moisture" class="tab-content">
                <div class="section">
                    <h2>💧 토양수분 분석</h2>
                    {self._generate_plots_html(plots_data.get('soil_moisture', {}), 'soil_moisture')}
                </div>
            </div>
            
            <!-- 검증 탭 -->
            <div id="validation" class="tab-content">
                <div class="section">
                    <h2>✅ 모델 검증</h2>
                    {self._generate_plots_html(plots_data.get('validation', {}), 'validation')}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>CRNP (Cosmic Ray Neutron Probe) 토양수분 모니터링 시스템</p>
            <p>이 리포트는 자동으로 생성되었습니다.</p>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {{
            // 모든 탭 내용 숨기기
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // 모든 탭 버튼 비활성화
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // 선택된 탭 활성화
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
        """
        
        return html
        
    def _generate_metadata_html(self, station_info: Dict) -> str:
        """메타데이터 HTML 생성"""
        
        metadata_items = [
            ("관측소 ID", station_info.get('station_id', 'N/A')),
            ("위치", f"{station_info.get('coordinates', {}).get('latitude', 'N/A'):.4f}, {station_info.get('coordinates', {}).get('longitude', 'N/A'):.4f}"),
            ("토양 벌크밀도", f"{station_info.get('soil_bulk_density', 'N/A')} g/cm³"),
            ("점토함량", f"{station_info.get('clay_content', 'N/A')*100:.1f}%"),
            ("캘리브레이션 상태", station_info.get('calibration_status', 'Unknown')),
            ("데이터 기간", station_info.get('data_period', 'N/A')),
            ("총 플롯 수", str(station_info.get('total_plots', 0))),
            ("생성 시각", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ]
        
        html_items = []
        for label, value in metadata_items:
            # 상태에 따른 배지 스타일
            if label == "캘리브레이션 상태":
                if "Available" in str(value) or "성공" in str(value):
                    value = f'<span class="status-badge status-success">{value}</span>'
                elif "Warning" in str(value) or "경고" in str(value):
                    value = f'<span class="status-badge status-warning">{value}</span>'
                else:
                    value = f'<span class="status-badge status-error">{value}</span>'
                    
            html_items.append(f"""
                <div class="metadata-item">
                    <span class="metadata-label">{label}</span>
                    <span class="metadata-value">{value}</span>
                </div>
            """)
            
        return ''.join(html_items)
        
    def _generate_plots_html(self, plots: Dict[str, str], category: str) -> str:
        """플롯 HTML 생성"""
        
        if not plots:
            return """
            <div class="plot-card">
                <h3>데이터 없음</h3>
                <div class="plot-description">
                    이 섹션에 대한 데이터가 없거나 플롯 생성에 실패했습니다.
                </div>
            </div>
            """
            
        plot_descriptions = {
            'raw_neutron': '원시 중성자 카운트의 시계열 변화',
            'corrected_neutron': '보정된 중성자 카운트의 시계열 변화',
            'neutron_comparison': '원시 vs 보정 중성자 카운트 비교',
            'correction_factors': '중성자 보정계수들의 시계열 변화',
            'neutron_statistics': '중성자 카운트의 통계적 특성 분석',
            'correction_summary': '보정계수들의 요약 통계',
            
            'vwc_timeseries': '체적수분함량(VWC)의 시계열 변화',
            'vwc_uncertainty': '불확실성을 포함한 VWC 시계열',
            'sensing_depth': 'CRNP 센싱 깊이의 시계열 변화',
            'storage': '토양수분 저장량의 시계열 변화',
            'vwc_weather': '기상조건과 함께 표시된 VWC',
            'sm_dashboard': '토양수분 종합 대시보드',
            'seasonal_patterns': 'VWC의 계절별 패턴 분석',
            
            'validation_timeseries': 'CRNP vs 지점센서 시계열 비교',
            'validation_scatter': 'CRNP vs 지점센서 산점도',
            'validation_residuals': '모델 잔차 분석',
            'validation_metrics': '검증 성능지표 요약',
            'validation_depth': '깊이별 검증 결과'
        }
        
        html_plots = []
        for plot_key, plot_path in plots.items():
            # 상대 경로로 변환
            rel_path = os.path.relpath(plot_path, self.viz_dir)
            
            description = plot_descriptions.get(plot_key, '분석 결과 플롯')
            
            # 플롯 제목 생성
            title = plot_key.replace('_', ' ').title()
            
            html_plots.append(f"""
                <div class="plot-card">
                    <h3>{title}</h3>
                    <img src="{rel_path}" alt="{title}" loading="lazy">
                    <div class="plot-description">{description}</div>
                </div>
            """)
            
        return f'<div class="plot-grid">{"".join(html_plots)}</div>'
        
    def _get_station_metadata(self) -> Dict:
        """관측소 메타데이터 수집"""
        
        metadata = {
            'station_id': self.station_id,
            'coordinates': {},
            'calibration_status': 'Unknown',
            'data_period': 'N/A',
            'total_plots': 0
        }
        
        try:
            # 캘리브레이션 상태 확인
            cal_file = self.output_dir / "calibration" / f"{self.station_id}_calibration_result.json"
            if cal_file.exists():
                with open(cal_file, 'r', encoding='utf-8') as f:
                    cal_data = json.load(f)
                    
                metadata['coordinates'] = cal_data.get('coordinates', {})
                metadata['soil_bulk_density'] = cal_data.get('soil_bulk_density', 'N/A')
                metadata['clay_content'] = cal_data.get('clay_content', 'N/A')
                metadata['calibration_status'] = 'Available'
                
                # 성능 지표
                metrics = cal_data.get('performance_metrics', {})
                if metrics:
                    r2 = metrics.get('R2', 0)
                    metadata['calibration_status'] = f'Available (R²={r2:.3f})'
                    
            # 토양수분 데이터 기간 확인
            sm_file = self.output_dir / "soil_moisture" / f"{self.station_id}_soil_moisture.xlsx"
            if sm_file.exists():
                sm_data = pd.read_excel(sm_file, index_col=0)
                if len(sm_data) > 0:
                    start_date = pd.to_datetime(sm_data.index.min()).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(sm_data.index.max()).strftime('%Y-%m-%d')
                    metadata['data_period'] = f'{start_date} ~ {end_date}'
                    
        except Exception as e:
            self.logger.warning(f"Error collecting metadata: {e}")
            
        return metadata
        
    def _save_visualization_metadata(self, results: Dict) -> None:
        """시각화 메타데이터 저장"""
        
        metadata_file = self.viz_dir / f"{self.station_id}_visualization_metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.log_file_operation("save", str(metadata_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save visualization metadata: {e}")
            
    def get_visualization_status(self) -> Dict[str, Any]:
        """시각화 상태 확인"""
        
        status = {
            'station_id': self.station_id,
            'plots_available': False,
            'html_report_available': False,
            'plots_count': 0,
            'generation_date': None
        }
        
        # 메타데이터 파일 확인
        metadata_file = self.viz_dir / f"{self.station_id}_visualization_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                status.update({
                    'plots_available': True,
                    'plots_count': metadata.get('total_plots', 0),
                    'generation_date': metadata.get('generation_timestamp'),
                    'plots_by_category': metadata.get('plots_generated', {})
                })
                
                # HTML 리포트 확인
                html_file = self.viz_dir / f"{self.station_id}_visualization_report.html"
                status['html_report_available'] = html_file.exists()
                
                if status['html_report_available']:
                    status['html_report_path'] = str(html_file)
                    
            except Exception as e:
                self.logger.warning(f"Error reading visualization metadata: {e}")
                
        return status
        
    def open_html_report(self) -> bool:
        """HTML 리포트를 기본 브라우저에서 열기"""
        
        html_file = self.viz_dir / f"{self.station_id}_visualization_report.html"
        
        if html_file.exists():
            try:
                # 절대 경로로 변환하여 브라우저에서 열기
                file_url = f"file://{html_file.absolute()}"
                webbrowser.open(file_url)
                self.logger.info(f"Opened HTML report in browser: {html_file}")
                return True
            except Exception as e:
                self.logger.warning(f"Could not open HTML report: {e}")
                return False
        else:
            self.logger.warning("HTML report not found")
            return False