# CRNP 데이터 처리 시스템

## 개요

CRNP (Cosmic Ray Neutron Probe) 데이터 처리 시스템은 우주선 중성자 탐지기를 이용한 토양수분 모니터링 데이터를 전처리, 캘리브레이션, 분석하는 통합 파이프라인입니다.

## 주요 기능

- **데이터 전처리**: FDR 센서 및 CRNP 데이터 자동 처리
- **캘리브레이션**: N0 매개변수 최적화 및 중성자 보정
- **토양수분 계산**: VWC (체적수분함량) 및 불확실성 산출
- **모델 검증**: 지점센서와의 비교 분석
- **시각화**: 분석 결과의 자동 차트 생성
- **통합 파이프라인**: 전체 과정의 일괄 처리

## 시스템 요구사항

### 필수 요구사항
- Python 3.8 이상
- 최소 4GB RAM
- 1GB 이상 디스크 공간

### 지원 운영체제
- Windows 10/11
- macOS 10.14 이상
- Linux (Ubuntu 18.04 이상)

## 설치 방법

### 1. 환경 설정 (자동)
```bash
python setup_environment.py
```

### 2. 환경 설정 (수동)
```bash
# 필수 라이브러리 설치
pip install pandas>=1.5.0 numpy>=1.20.0 openpyxl>=3.0.0
pip install PyYAML>=6.0 matplotlib>=3.5.0 scipy>=1.7.0

# 추가 라이브러리 (권장)
pip install chardet seaborn scikit-learn psutil joblib
```

### 3. 디렉토리 구조 확인
```
crnp_system/
├── src/                    # 핵심 소스코드
│   ├── core/              # 핵심 모듈 (설정, 로깅)
│   ├── preprocessing/     # 데이터 전처리
│   ├── calibration/       # 캘리브레이션
│   ├── calculation/       # 토양수분 계산
│   ├── validation/        # 모델 검증
│   ├── visualization/     # 시각화
│   └── utils/            # 유틸리티
├── scripts/               # 실행 스크립트
├── config/               # 설정 파일
├── data/                 # 데이터 폴더
│   ├── input/           # 입력 데이터
│   └── output/          # 출력 결과
└── logs/                # 로그 파일
```

## 데이터 준비

### 1. 디렉토리 구조 생성
관측소별로 다음 구조로 데이터를 배치:

```
data/input/{관측소ID}/
├── fdr/                  # FDR 센서 데이터 (.csv)
├── crnp/                # CRNP 데이터 (.xlsx, .csv)
└── geo_info.xlsx        # 지리정보 (선택사항)
```

### 2. 지원 파일 형식

#### FDR 센서 데이터
- **형식**: CSV 파일
- **명명규칙**: 
  - PC 관측소: `z6-{loc_key}(...)-Configuration....csv`
  - HC 관측소: `HC-{site_name}(z6-{loc_key})(...)-Configuration....csv`
- **필수 컬럼**: Timestamps, Water Content (multiple depths)

#### CRNP 데이터
- **형식**: Excel (.xlsx) 또는 CSV 파일 (TOA5 형식)
- **필수 컬럼**: 
  - Timestamp/TIMESTAMP (시간)
  - N_counts/HI_NeutronCts_Tot (중성자 카운트)
  - Ta/Air_Temp_Avg (기온)
  - RH/RH_Avg (상대습도)
  - Pa/Air_Press_Avg (기압)

### 3. 설정 파일 생성
```bash
# 관측소 설정 템플릿 생성
python scripts/run_preprocessing.py --setup-station HC
python scripts/run_preprocessing.py --setup-station PC
```

## 사용 방법

### 1. 전체 파이프라인 실행 (권장)
```bash
# 전체 과정 자동 실행
python scripts/run_crnp_pipeline.py --station HC --all

# 특정 단계만 실행
python scripts/run_crnp_pipeline.py --station HC --steps preprocessing calibration

# 상태 확인
python scripts/run_crnp_pipeline.py --station HC --status
```

### 2. 단계별 실행

#### 2.1 데이터 전처리
```bash
# 데이터 파일 확인
python scripts/run_preprocessing.py --station HC --check-only

# 전처리 실행
python scripts/run_preprocessing.py --station HC
```

#### 2.2 캘리브레이션
```bash
# 캘리브레이션 실행 (기본 기간)
python scripts/run_calibration.py --station HC

# 사용자 지정 기간
python scripts/run_calibration.py --station HC --start 2024-08-17 --end 2024-08-25

# 자동 기간 최적화
python scripts/run_calibration.py --station HC --auto-optimize

# 상태 확인
python scripts/run_calibration.py --station HC --status
```

#### 2.3 토양수분 계산
```bash
# 토양수분 계산 실행
python scripts/run_soil_moisture.py --station HC

# 특정 기간 계산
python scripts/run_soil_moisture.py --station HC --start 2024-08-01 --end 2024-12-31

# 상태 확인
python scripts/run_soil_moisture.py --station HC --status
```

#### 2.4 시각화
```bash
# 간단한 시각화 생성
python scripts/run_visualization.py --station HC

# 데이터 가용성만 확인
python scripts/run_visualization.py --station HC --check-only
```

## 설정 관리

### 관측소별 설정 (config/stations/{station_id}.yaml)
```yaml
station_info:
  id: "HC"
  name: "Hongcheon Station"
  description: "홍천 관측소"

coordinates:
  latitude: 37.7049111
  longitude: 128.0316412
  altitude: null

soil_properties:
  bulk_density: 1.44
  clay_content: 0.35
  lattice_water: null  # 자동 계산

sensor_configuration:
  depths: [10, 30, 60]
  fdr_sensor_count: 3

data_paths:
  crnp_folder: "data/input/HC/crnp/"
  fdr_folder: "data/input/HC/fdr/"
```

### 처리 옵션 설정 (config/processing_options.yaml)
```yaml
corrections:
  incoming_flux: true    # fi 보정
  pressure: true         # fp 보정  
  humidity: true         # fw 보정
  biomass: false         # fb 보정

calibration:
  optimization_method: "Nelder-Mead"
  initial_N0: 1000
  weighting_method: "Schron_2017"

calculation:
  exclude_periods:
    winter_months: [12, 1, 2]
    custom_dates: []
  
  smoothing:
    enabled: false
    method: "savitzky_golay"
    window: 11
    order: 3
```

## 출력 결과

### 1. 디렉토리 구조
```
data/output/{관측소ID}/
├── preprocessed/         # 전처리 결과
├── calibration/         # 캘리브레이션 결과  
├── soil_moisture/       # 토양수분 계산 결과
├── validation/          # 검증 결과
└── visualization/       # 시각화 결과
```

### 2. 주요 출력 파일

#### 전처리 결과
- `{station}_FDR_input.xlsx`: FDR 센서 데이터
- `{station}_CRNP_input.xlsx`: CRNP 데이터
- `{station}_FDR_daily_avg.xlsx`: 일평균 FDR 데이터

#### 캘리브레이션 결과
- `{station}_calibration_result.json`: 캘리브레이션 매개변수
- `{station}_Parameters.xlsx`: 매개변수 Excel 파일
- `{station}_calibration_diagnostics.png`: 진단 그래프
- `{station}_calibration_debug_data.xlsx`: 디버깅 데이터

#### 토양수분 계산 결과
- `{station}_soil_moisture.xlsx`: 토양수분 시계열
- `{station}_calculation_metadata.json`: 계산 메타데이터
- `{station}_calculation_report.txt`: 계산 보고서

#### 검증 결과
- `{station}_validation_result.json`: 검증 성능지표
- `{station}_validation_data.xlsx`: 매칭된 검증 데이터
- `{station}_validation_report.txt`: 검증 보고서

#### 시각화 결과
- `{station}_neutron_comparison.png`: 중성자 카운트 비교
- `{station}_correction_factors.png`: 보정계수 시계열
- `{station}_vwc_timeseries.png`: VWC 시계열
- `{station}_soil_moisture_comparison.png`: 토양수분 비교
- `{station}_soil_moisture_scatter.png`: 산점도

## 문제 해결

### 일반적인 문제들

#### 1. 데이터 파일을 찾을 수 없음
```bash
# 데이터 파일 확인
python scripts/run_preprocessing.py --station HC --check-only

# 파일 배치 가이드 확인
python scripts/run_preprocessing.py --station HC
```

#### 2. 캘리브레이션 실패
```bash
# 데이터 품질 확인
python scripts/run_calibration.py --station HC --status

# 자동 기간 최적화 시도
python scripts/run_calibration.py --station HC --auto-optimize
```

#### 3. 중성자 카운트 데이터 없음
- TOA5 파일의 중성자 검출기 컬럼명 확인
- 가능한 컬럼명: `HI_NeutronCts_Tot`, `NeutronCounts_Tot`, `CRNP_Tot`

#### 4. 메모리 부족
- 대용량 데이터 처리 시 RAM 부족 가능
- 처리 기간을 단축하거나 시스템 메모리 증설 필요

### 로그 확인
```bash
# 로그 파일 위치
logs/{모듈명}_{날짜}.log

# 실시간 로그 확인 (Linux/macOS)
tail -f logs/PreprocessingPipeline_*.log
```

## 개발자 가이드

### 코드 구조

#### 핵심 모듈
- `src/core/config_manager.py`: 설정 관리
- `src/core/logger.py`: 로깅 시스템
- `src/preprocessing/`: 데이터 전처리
- `src/calibration/`: 캘리브레이션 엔진
- `src/calculation/`: 토양수분 계산
- `src/validation/`: 모델 검증
- `src/visualization/`: 시각화

#### 주요 클래스
- `ConfigManager`: 설정 파일 관리
- `PreprocessingPipeline`: 전처리 파이프라인
- `CalibrationManager`: 캘리브레이션 관리
- `SoilMoistureManager`: 토양수분 계산 관리
- `ValidationManager`: 검증 관리

### 새로운 관측소 추가
1. 설정 파일 생성: `config/stations/{new_station}.yaml`
2. 데이터 디렉토리 생성: `data/input/{new_station}/`
3. 파일 패턴 수정 (필요시): `src/utils/file_handler.py`

### 커스터마이징
- 보정 방법 추가: `src/calibration/neutron_correction.py`
- 새로운 시각화: `src/visualization/`
- 데이터 형식 지원: `src/utils/file_handler.py`

## 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

## 지원

### 기술 지원
- 이슈 리포트: GitHub Issues
- 문서: `docs/` 폴더 참조

### 참고 자료
- Schröder et al. (2017): CRNP 측정 원리
- Kohli et al. (2015): 중성자 보정 방법
- Andreasen et al. (2017): 캘리브레이션 방법론

## 업데이트 이력

### v1.0.0 (2024-12-XX)
- 초기 릴리스
- 기본 전처리, 캘리브레이션, 계산 기능
- TOA5 형식 지원
- 통합 파이프라인 구현
- 간단한 시각화 기능

## 예제 실행

### 빠른 시작
```bash
# 1. 환경 설정
python setup_environment.py

# 2. 관측소 설정 생성
python scripts/run_preprocessing.py --setup-station HC

# 3. 데이터 파일 배치 (수동)
# data/input/HC/fdr/ 에 FDR CSV 파일들 복사
# data/input/HC/crnp/ 에 CRNP Excel/CSV 파일 복사

# 4. 전체 파이프라인 실행
python scripts/run_crnp_pipeline.py --station HC --all

# 5. 결과 확인
# data/output/HC/ 폴더에서 결과 확인
```

### 단계별 실행 예제
```bash
# 데이터 확인
python scripts/run_preprocessing.py --station HC --check-only

# 전처리
python scripts/run_preprocessing.py --station HC

# 캘리브레이션 (자동 최적화)
python scripts/run_calibration.py --station HC --auto-optimize

# 토양수분 계산
python scripts/run_soil_moisture.py --station HC

# 시각화
python scripts/run_visualization.py --station HC
```

이 시스템을 통해 CRNP 데이터의 전체 처리 과정을 자동화하고, 신뢰할 수 있는 토양수분 모니터링 결과를 얻을 수 있습니다.