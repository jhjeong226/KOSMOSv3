# web/app.py

"""
CRNP 웹 대시보드
Flask 기반의 CRNP 분석 결과 웹 인터페이스

실행 방법:
    python web/app.py
    또는
    python scripts/run_dashboard.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, jsonify, request, send_file, abort
import pandas as pd

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logger import setup_logger
from src.visualization.visualization_manager import VisualizationManager
from src.calibration.calibration_manager import CalibrationManager
from src.calculation.soil_moisture_manager import SoilMoistureManager
from src.validation.validation_manager import ValidationManager


class CRNPDashboard:
    """CRNP 웹 대시보드 클래스"""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.logger = setup_logger("CRNPDashboard")
        self.project_root = project_root
        
        # 라우트 설정
        self._setup_routes()
        
        # 사용 가능한 관측소 목록
        self.available_stations = self._discover_stations()
        
    def _setup_routes(self):
        """라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 페이지"""
            return render_template('dashboard.html', 
                                 stations=self.available_stations,
                                 current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        @self.app.route('/api/stations')
        def api_stations():
            """관측소 목록 API"""
            return jsonify({
                'stations': self.available_stations,
                'count': len(self.available_stations)
            })
        
        @self.app.route('/api/station/<station_id>/status')
        def api_station_status(station_id: str):
            """관측소 상태 API"""
            try:
                status = self._get_station_status(station_id)
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/station/<station_id>/data')
        def api_station_data(station_id: str):
            """관측소 데이터 API"""
            try:
                data_type = request.args.get('type', 'soil_moisture')
                days = int(request.args.get('days', 30))
                
                data = self._get_station_data(station_id, data_type, days)
                return jsonify(data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/station/<station_id>/plots')
        def api_station_plots(station_id: str):
            """관측소 플롯 목록 API"""
            try:
                plots = self._get_station_plots(station_id)
                return jsonify(plots)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/station/<station_id>/generate_plots', methods=['POST'])
        def api_generate_plots(station_id: str):
            """플롯 생성 API"""
            try:
                include_validation = request.json.get('include_validation', True)
                
                viz_manager = VisualizationManager(station_id)
                result = viz_manager.generate_all_plots(include_validation=include_validation)
                
                return jsonify({
                    'success': True,
                    'total_plots': result.get('total_plots', 0),
                    'html_report': result.get('html_report')
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
                
        @self.app.route('/plots/<station_id>/<filename>')
        def serve_plot(station_id: str, filename: str):
            """플롯 파일 서빙"""
            try:
                plot_path = self.project_root / f"data/output/{station_id}/visualization/{filename}"
                
                if plot_path.exists() and plot_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                    return send_file(plot_path)
                else:
                    abort(404)
            except Exception as e:
                abort(404)
                
        @self.app.route('/station/<station_id>')
        def station_detail(station_id: str):
            """관측소 상세 페이지"""
            try:
                if station_id not in self.available_stations:
                    abort(404)
                    
                status = self._get_station_status(station_id)
                return render_template('station_detail.html', 
                                     station_id=station_id,
                                     status=status)
            except Exception as e:
                abort(500)
                
        @self.app.route('/reports/<station_id>')
        def station_reports(station_id: str):
            """관측소 리포트 페이지"""
            try:
                if station_id not in self.available_stations:
                    abort(404)
                    
                html_report_path = self.project_root / f"data/output/{station_id}/visualization/{station_id}_visualization_report.html"
                
                if html_report_path.exists():
                    return send_file(html_report_path)
                else:
                    return render_template('no_report.html', station_id=station_id)
            except Exception as e:
                abort(500)
                
    def _discover_stations(self) -> List[str]:
        """사용 가능한 관측소 목록 발견"""
        stations = []
        
        output_dir = self.project_root / "data" / "output"
        
        if output_dir.exists():
            for station_dir in output_dir.iterdir():
                if station_dir.is_dir():
                    station_id = station_dir.name
                    
                    # 최소한의 데이터가 있는지 확인
                    preprocessed_dir = station_dir / "preprocessed"
                    if preprocessed_dir.exists() and list(preprocessed_dir.glob("*.xlsx")):
                        stations.append(station_id)
                        
        return sorted(stations)
        
    def _get_station_status(self, station_id: str) -> Dict[str, Any]:
        """관측소 상태 정보 수집"""
        
        status = {
            'station_id': station_id,
            'data_available': False,
            'preprocessing': {'status': 'unknown'},
            'calibration': {'status': 'unknown'},
            'soil_moisture': {'status': 'unknown'},
            'validation': {'status': 'unknown'},
            'visualization': {'status': 'unknown'},
            'last_updated': None
        }
        
        base_dir = self.project_root / f"data/output/{station_id}"
        
        try:
            # 1. 전처리 상태
            preprocessed_dir = base_dir / "preprocessed"
            fdr_file = preprocessed_dir / f"{station_id}_FDR_input.xlsx"
            crnp_file = preprocessed_dir / f"{station_id}_CRNP_input.xlsx"
            
            if fdr_file.exists() and crnp_file.exists():
                status['preprocessing']['status'] = 'completed'
                status['data_available'] = True
                
                # 데이터 기본 정보
                try:
                    fdr_data = pd.read_excel(fdr_file)
                    crnp_data = pd.read_excel(crnp_file)
                    
                    status['preprocessing']['fdr_records'] = len(fdr_data)
                    status['preprocessing']['crnp_records'] = len(crnp_data)
                    
                    if 'Date' in fdr_data.columns:
                        fdr_dates = pd.to_datetime(fdr_data['Date'])
                        status['preprocessing']['fdr_date_range'] = {
                            'start': fdr_dates.min().strftime('%Y-%m-%d'),
                            'end': fdr_dates.max().strftime('%Y-%m-%d')
                        }
                        
                    if 'timestamp' in crnp_data.columns:
                        crnp_dates = pd.to_datetime(crnp_data['timestamp'])
                        status['preprocessing']['crnp_date_range'] = {
                            'start': crnp_dates.min().strftime('%Y-%m-%d'),
                            'end': crnp_dates.max().strftime('%Y-%m-%d')
                        }
                except:
                    pass
            else:
                status['preprocessing']['status'] = 'missing'
                
            # 2. 캘리브레이션 상태
            try:
                cal_manager = CalibrationManager(station_id)
                cal_status = cal_manager.get_calibration_status()
                
                if cal_status['calibration_available']:
                    status['calibration']['status'] = 'completed'
                    status['calibration']['N0'] = cal_status.get('N0_rdt')
                    
                    metrics = cal_status.get('performance_metrics', {})
                    if metrics:
                        status['calibration']['R2'] = metrics.get('R2')
                        status['calibration']['RMSE'] = metrics.get('RMSE')
                else:
                    status['calibration']['status'] = 'missing'
            except:
                status['calibration']['status'] = 'error'
                
            # 3. 토양수분 계산 상태
            try:
                sm_manager = SoilMoistureManager(station_id)
                sm_status = sm_manager.get_calculation_status()
                
                if sm_status['calculation_available']:
                    status['soil_moisture']['status'] = 'completed'
                    status['soil_moisture']['records'] = sm_status.get('data_records')
                    
                    data_summary = sm_status.get('data_summary', {})
                    if data_summary:
                        vwc_stats = data_summary.get('vwc_statistics', {})
                        status['soil_moisture']['vwc_mean'] = vwc_stats.get('mean')
                        status['soil_moisture']['vwc_std'] = vwc_stats.get('std')
                else:
                    status['soil_moisture']['status'] = 'missing'
            except:
                status['soil_moisture']['status'] = 'error'
                
            # 4. 검증 상태
            try:
                val_manager = ValidationManager(station_id)
                val_status = val_manager.get_validation_status()
                
                if val_status['validation_available']:
                    status['validation']['status'] = 'completed'
                    
                    overall_metrics = val_status.get('overall_metrics', {})
                    if overall_metrics:
                        status['validation']['R2'] = overall_metrics.get('R2')
                        status['validation']['RMSE'] = overall_metrics.get('RMSE')
                else:
                    status['validation']['status'] = 'missing'
            except:
                status['validation']['status'] = 'error'
                
            # 5. 시각화 상태
            try:
                viz_manager = VisualizationManager(station_id)
                viz_status = viz_manager.get_visualization_status()
                
                if viz_status['plots_available']:
                    status['visualization']['status'] = 'completed'
                    status['visualization']['plots_count'] = viz_status.get('plots_count', 0)
                    status['visualization']['html_available'] = viz_status.get('html_report_available', False)
                else:
                    status['visualization']['status'] = 'missing'
            except:
                status['visualization']['status'] = 'error'
                
            # 6. 마지막 업데이트 시간
            latest_files = []
            for subdir in ['preprocessed', 'calibration', 'soil_moisture', 'validation', 'visualization']:
                subdir_path = base_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.glob('*'):
                        if file_path.is_file():
                            latest_files.append(file_path.stat().st_mtime)
                            
            if latest_files:
                latest_time = max(latest_files)
                status['last_updated'] = datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
                
        except Exception as e:
            self.logger.warning(f"Error getting station status for {station_id}: {e}")
            
        return status
        
    def _get_station_data(self, station_id: str, data_type: str, days: int) -> Dict[str, Any]:
        """관측소 데이터 조회"""
        
        data = {
            'station_id': station_id,
            'data_type': data_type,
            'days': days,
            'records': [],
            'error': None
        }
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if data_type == 'soil_moisture':
                # 토양수분 데이터
                sm_file = self.project_root / f"data/output/{station_id}/soil_moisture/{station_id}_soil_moisture.xlsx"
                
                if sm_file.exists():
                    df = pd.read_excel(sm_file, index_col=0)
                    df.index = pd.to_datetime(df.index)
                    
                    # 기간 필터링
                    mask = df.index >= start_date
                    df_filtered = df[mask]
                    
                    # JSON 직렬화 가능한 형태로 변환
                    records = []
                    for idx, row in df_filtered.iterrows():
                        record = {
                            'date': idx.strftime('%Y-%m-%d'),
                            'timestamp': idx.isoformat()
                        }
                        
                        for col in df_filtered.columns:
                            value = row[col]
                            if pd.notna(value):
                                record[col] = float(value)
                            else:
                                record[col] = None
                                
                        records.append(record)
                        
                    data['records'] = records
                    data['count'] = len(records)
                    
            elif data_type == 'neutron':
                # 중성자 카운트 데이터
                crnp_file = self.project_root / f"data/output/{station_id}/preprocessed/{station_id}_CRNP_input.xlsx"
                
                if crnp_file.exists():
                    df = pd.read_excel(crnp_file)
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        
                        # 기간 필터링
                        mask = df.index >= start_date
                        df_filtered = df[mask]
                        
                        # 필요한 컬럼만 선택
                        cols_to_include = ['N_counts', 'Ta', 'RH', 'Pa']
                        available_cols = [col for col in cols_to_include if col in df_filtered.columns]
                        
                        if available_cols:
                            df_subset = df_filtered[available_cols]
                            
                            records = []
                            for idx, row in df_subset.iterrows():
                                record = {
                                    'date': idx.strftime('%Y-%m-%d'),
                                    'timestamp': idx.isoformat()
                                }
                                
                                for col in available_cols:
                                    value = row[col]
                                    if pd.notna(value):
                                        record[col] = float(value)
                                    else:
                                        record[col] = None
                                        
                                records.append(record)
                                
                            data['records'] = records
                            data['count'] = len(records)
                            
        except Exception as e:
            data['error'] = str(e)
            
        return data
        
    def _get_station_plots(self, station_id: str) -> Dict[str, Any]:
        """관측소 플롯 목록 조회"""
        
        plots = {
            'station_id': station_id,
            'categories': {},
            'total_count': 0
        }
        
        viz_dir = self.project_root / f"data/output/{station_id}/visualization"
        
        if viz_dir.exists():
            plot_files = list(viz_dir.glob("*.png"))
            
            # 카테고리별로 분류
            categories = {
                'neutron': [],
                'soil_moisture': [],
                'validation': [],
                'other': []
            }
            
            for plot_file in plot_files:
                filename = plot_file.name
                plot_info = {
                    'filename': filename,
                    'title': filename.replace('_', ' ').replace('.png', '').title(),
                    'url': f'/plots/{station_id}/{filename}',
                    'size': plot_file.stat().st_size
                }
                
                # 카테고리 분류
                if 'neutron' in filename.lower() or 'correction' in filename.lower():
                    categories['neutron'].append(plot_info)
                elif 'vwc' in filename.lower() or 'moisture' in filename.lower() or 'storage' in filename.lower():
                    categories['soil_moisture'].append(plot_info)
                elif 'validation' in filename.lower():
                    categories['validation'].append(plot_info)
                else:
                    categories['other'].append(plot_info)
                    
            plots['categories'] = categories
            plots['total_count'] = len(plot_files)
            
        return plots
        
    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        """웹 서버 실행"""
        self.logger.info(f"Starting CRNP Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# Flask 템플릿 파일들
def create_templates():
    """템플릿 파일들 생성"""
    
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # 1. 기본 레이아웃
    base_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}CRNP Dashboard{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #f8f9fa; }
        .navbar-brand { font-weight: bold; }
        .card { box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); margin-bottom: 1rem; }
        .status-badge { font-size: 0.75rem; }
        .metric-card { text-align: center; }
        .metric-value { font-size: 2rem; font-weight: bold; }
        .metric-label { color: #6c757d; }
        .plot-container { max-width: 100%; overflow: auto; }
        .plot-container img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-satellite-dish me-2"></i>CRNP Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text">
                    <i class="fas fa-clock me-1"></i>{{ current_time }}
                </span>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
    """
    
    # 2. 메인 대시보드 페이지
    dashboard_template = """
{% extends "base.html" %}

{% block title %}CRNP Dashboard - Home{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h1 class="mb-4">
            <i class="fas fa-tachometer-alt me-2"></i>CRNP 관측소 대시보드
        </h1>
    </div>
</div>

<div class="row">
    {% for station_id in stations %}
    <div class="col-lg-6 col-xl-4 mb-4">
        <div class="card h-100">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">
                    <i class="fas fa-map-marker-alt me-2"></i>{{ station_id }} 관측소
                </h5>
                <span class="badge bg-secondary status-badge" id="status-{{ station_id }}">
                    <i class="fas fa-spinner fa-spin"></i> 로딩중
                </span>
            </div>
            <div class="card-body">
                <div id="station-info-{{ station_id }}">
                    <div class="text-center">
                        <div class="spinner-border spinner-border-sm" role="status"></div>
                        <p class="mt-2 text-muted">상태 확인 중...</p>
                    </div>
                </div>
            </div>
            <div class="card-footer">
                <div class="btn-group w-100" role="group">
                    <a href="/station/{{ station_id }}" class="btn btn-primary btn-sm">
                        <i class="fas fa-chart-line me-1"></i>상세
                    </a>
                    <a href="/reports/{{ station_id }}" class="btn btn-success btn-sm" target="_blank">
                        <i class="fas fa-file-alt me-1"></i>리포트
                    </a>
                    <button class="btn btn-outline-secondary btn-sm" onclick="generatePlots('{{ station_id }}')">
                        <i class="fas fa-sync me-1"></i>갱신
                    </button>
                </div>
            </div>
        </div>
    </div>
    {% endfor %}
</div>

{% if not stations %}
<div class="row">
    <div class="col-12">
        <div class="alert alert-info text-center">
            <i class="fas fa-info-circle me-2"></i>
            사용 가능한 관측소가 없습니다. 먼저 데이터를 전처리해주세요.
        </div>
    </div>
</div>
{% endif %}
{% endblock %}

{% block scripts %}
<script>
// 페이지 로드 시 모든 관측소 상태 확인
document.addEventListener('DOMContentLoaded', function() {
    {% for station_id in stations %}
    loadStationStatus('{{ station_id }}');
    {% endfor %}
});

async function loadStationStatus(stationId) {
    try {
        const response = await fetch(`/api/station/${stationId}/status`);
        const status = await response.json();
        
        updateStationCard(stationId, status);
    } catch (error) {
        console.error(`Error loading status for ${stationId}:`, error);
        document.getElementById(`status-${stationId}`).innerHTML = 
            '<i class="fas fa-exclamation-triangle"></i> 오류';
        document.getElementById(`status-${stationId}`).className = 'badge bg-danger status-badge';
    }
}

function updateStationCard(stationId, status) {
    const statusBadge = document.getElementById(`status-${stationId}`);
    const infoDiv = document.getElementById(`station-info-${stationId}`);
    
    // 전체 상태 결정
    let overallStatus = 'success';
    let statusText = '정상';
    let statusIcon = 'fas fa-check-circle';
    
    if (status.preprocessing.status !== 'completed' || 
        status.calibration.status !== 'completed' ||
        status.soil_moisture.status !== 'completed') {
        overallStatus = 'warning';
        statusText = '불완전';
        statusIcon = 'fas fa-exclamation-triangle';
    }
    
    statusBadge.innerHTML = `<i class="${statusIcon}"></i> ${statusText}`;
    statusBadge.className = `badge bg-${overallStatus} status-badge`;
    
    // 상세 정보 업데이트
    let infoHtml = `
        <div class="row g-2">
            <div class="col-6">
                <div class="metric-card">
                    <div class="metric-value text-primary">${status.preprocessing.fdr_records || 0}</div>
                    <div class="metric-label">FDR 레코드</div>
                </div>
            </div>
            <div class="col-6">
                <div class="metric-card">
                    <div class="metric-value text-info">${status.preprocessing.crnp_records || 0}</div>
                    <div class="metric-label">CRNP 레코드</div>
                </div>
            </div>
        </div>
        
        <div class="mt-2">
            <small class="text-muted">
                <i class="fas fa-clock me-1"></i>
                최종 업데이트: ${status.last_updated || 'N/A'}
            </small>
        </div>
        
        <div class="mt-2">
            <div class="progress" style="height: 6px;">
                <div class="progress-bar bg-primary" style="width: ${getProgressWidth(status)}%"></div>
            </div>
            <small class="text-muted">처리 완료도</small>
        </div>
    `;
    
    infoDiv.innerHTML = infoHtml;
}

function getProgressWidth(status) {
    let completed = 0;
    const total = 5; // preprocessing, calibration, soil_moisture, validation, visualization
    
    if (status.preprocessing.status === 'completed') completed++;
    if (status.calibration.status === 'completed') completed++;
    if (status.soil_moisture.status === 'completed') completed++;
    if (status.validation.status === 'completed') completed++;
    if (status.visualization.status === 'completed') completed++;
    
    return (completed / total) * 100;
}

async function generatePlots(stationId) {
    const button = event.target;
    const originalHtml = button.innerHTML;
    
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>생성중';
    button.disabled = true;
    
    try {
        const response = await fetch(`/api/station/${stationId}/generate_plots`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                include_validation: true
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(`플롯 생성 완료! ${result.total_plots}개의 플롯이 생성되었습니다.`);
            loadStationStatus(stationId);
        } else {
            alert(`플롯 생성 실패: ${result.error}`);
        }
    } catch (error) {
        alert(`오류가 발생했습니다: ${error.message}`);
    } finally {
        button.innerHTML = originalHtml;
        button.disabled = false;
    }
}
</script>
{% endblock %}
    """
    
    # 3. 관측소 상세 페이지
    station_detail_template = """
{% extends "base.html" %}

{% block title %}{{ station_id }} 관측소 - CRNP Dashboard{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1>
                <i class="fas fa-map-marker-alt me-2"></i>{{ station_id }} 관측소
            </h1>
            <div>
                <a href="/" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i>뒤로
                </a>
                <a href="/reports/{{ station_id }}" class="btn btn-success" target="_blank">
                    <i class="fas fa-file-alt me-1"></i>전체 리포트
                </a>
            </div>
        </div>
    </div>
</div>

<!-- 상태 카드들 -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6 mb-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-database fa-2x text-primary mb-2"></i>
                <h5>전처리</h5>
                <span class="badge bg-secondary" id="preprocessing-status">로딩중</span>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 mb-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-cogs fa-2x text-info mb-2"></i>
                <h5>캘리브레이션</h5>
                <span class="badge bg-secondary" id="calibration-status">로딩중</span>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 mb-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-tint fa-2x text-success mb-2"></i>
                <h5>토양수분</h5>
                <span class="badge bg-secondary" id="soil-moisture-status">로딩중</span>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 mb-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-chart-line fa-2x text-warning mb-2"></i>
                <h5>시각화</h5>
                <span class="badge bg-secondary" id="visualization-status">로딩중</span>
            </div>
        </div>
    </div>
</div>

<!-- 데이터 차트 -->
<div class="row mb-4">
    <div class="col-lg-6 mb-3">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-tint me-2"></i>토양수분 변화
                </h5>
            </div>
            <div class="card-body">
                <canvas id="soilMoistureChart" height="200"></canvas>
            </div>
        </div>
    </div>
    <div class="col-lg-6 mb-3">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-satellite-dish me-2"></i>중성자 카운트
                </h5>
            </div>
            <div class="card-body">
                <canvas id="neutronChart" height="200"></canvas>
            </div>
        </div>
    </div>
</div>

<!-- 플롯 갤러리 -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-images me-2"></i>분석 결과 플롯
                </h5>
            </div>
            <div class="card-body" id="plots-gallery">
                <div class="text-center">
                    <div class="spinner-border" role="status"></div>
                    <p class="mt-2">플롯 로딩 중...</p>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let soilMoistureChart = null;
let neutronChart = null;

document.addEventListener('DOMContentLoaded', function() {
    loadStationData();
    loadPlots();
});

async function loadStationData() {
    try {
        // 상태 정보 로드
        const statusResponse = await fetch(`/api/station/{{ station_id }}/status`);
        const status = await statusResponse.json();
        updateStatusCards(status);
        
        // 토양수분 데이터 로드
        const smResponse = await fetch(`/api/station/{{ station_id }}/data?type=soil_moisture&days=30`);
        const smData = await smResponse.json();
        createSoilMoistureChart(smData);
        
        // 중성자 데이터 로드
        const neutronResponse = await fetch(`/api/station/{{ station_id }}/data?type=neutron&days=30`);
        const neutronData = await neutronResponse.json();
        createNeutronChart(neutronData);
        
    } catch (error) {
        console.error('Error loading station data:', error);
    }
}

function updateStatusCards(status) {
    const statusMap = {
        'completed': { class: 'bg-success', text: '완료' },
        'missing': { class: 'bg-warning', text: '미완료' },
        'error': { class: 'bg-danger', text: '오류' },
        'unknown': { class: 'bg-secondary', text: '알 수 없음' }
    };
    
    ['preprocessing', 'calibration', 'soil-moisture', 'visualization'].forEach(step => {
        const element = document.getElementById(`${step}-status`);
        const stepStatus = status[step.replace('-', '_')].status;
        const statusInfo = statusMap[stepStatus] || statusMap['unknown'];
        
        element.className = `badge ${statusInfo.class}`;
        element.textContent = statusInfo.text;
    });
}

function createSoilMoistureChart(data) {
    const ctx = document.getElementById('soilMoistureChart').getContext('2d');
    
    if (soilMoistureChart) {
        soilMoistureChart.destroy();
    }
    
    const labels = data.records.map(r => r.date);
    const vwcData = data.records.map(r => r.VWC);
    
    soilMoistureChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'VWC (m³/m³)',
                data: vwcData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function createNeutronChart(data) {
    const ctx = document.getElementById('neutronChart').getContext('2d');
    
    if (neutronChart) {
        neutronChart.destroy();
    }
    
    const labels = data.records.map(r => r.date);
    const neutronData = data.records.map(r => r.N_counts);
    
    neutronChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Neutron Counts',
                data: neutronData,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

async function loadPlots() {
    try {
        const response = await fetch(`/api/station/{{ station_id }}/plots`);
        const plotsData = await response.json();
        
        displayPlots(plotsData);
    } catch (error) {
        document.getElementById('plots-gallery').innerHTML = 
            '<div class="alert alert-warning">플롯을 로드할 수 없습니다.</div>';
    }
}

function displayPlots(plotsData) {
    const gallery = document.getElementById('plots-gallery');
    
    if (plotsData.total_count === 0) {
        gallery.innerHTML = `
            <div class="alert alert-info text-center">
                <i class="fas fa-info-circle me-2"></i>
                생성된 플롯이 없습니다. 
                <button class="btn btn-primary btn-sm ms-2" onclick="generatePlots()">
                    <i class="fas fa-sync me-1"></i>플롯 생성
                </button>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    Object.entries(plotsData.categories).forEach(([category, plots]) => {
        if (plots.length > 0) {
            html += `
                <h6 class="mt-3 mb-2">${category.toUpperCase()}</h6>
                <div class="row">
            `;
            
            plots.forEach(plot => {
                html += `
                    <div class="col-lg-4 col-md-6 mb-3">
                        <div class="card">
                            <div class="plot-container">
                                <img src="${plot.url}" class="card-img-top" alt="${plot.title}">
                            </div>
                            <div class="card-body p-2">
                                <small class="text-muted">${plot.title}</small>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
        }
    });
    
    gallery.innerHTML = html;
}

async function generatePlots() {
    // 플롯 생성 로직 (메인 페이지와 동일)
    // ... (생략)
}
</script>
{% endblock %}
    """
    
    # 4. 기본 레이아웃 파일 저장
    with open(templates_dir / "base.html", 'w', encoding='utf-8') as f:
        f.write(base_template)
        
    with open(templates_dir / "dashboard.html", 'w', encoding='utf-8') as f:
        f.write(dashboard_template)
        
    with open(templates_dir / "station_detail.html", 'w', encoding='utf-8') as f:
        f.write(station_detail_template)
        
    # 5. 리포트 없음 페이지
    no_report_template = """
{% extends "base.html" %}

{% block title %}리포트 없음 - {{ station_id }}{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-file-times fa-4x text-muted mb-3"></i>
                <h4>리포트를 찾을 수 없습니다</h4>
                <p class="text-muted">{{ station_id }} 관측소의 시각화 리포트가 아직 생성되지 않았습니다.</p>
                <a href="/station/{{ station_id }}" class="btn btn-primary">
                    <i class="fas fa-chart-line me-1"></i>플롯 생성하기
                </a>
                <a href="/" class="btn btn-outline-secondary ms-2">
                    <i class="fas fa-home me-1"></i>홈으로
                </a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    """
    
    with open(templates_dir / "no_report.html", 'w', encoding='utf-8') as f:
        f.write(no_report_template)
        

if __name__ == "__main__":
    # 템플릿 파일 생성
    create_templates()
    
    # 대시보드 실행
    dashboard = CRNPDashboard()
    dashboard.run(debug=True)