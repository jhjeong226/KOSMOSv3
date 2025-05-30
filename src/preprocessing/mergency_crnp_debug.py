# scripts/emergency_crnp_debug.py

"""
CRNP 데이터 긴급 진단 및 수정 스크립트
문제를 즉시 찾아서 해결합니다.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def emergency_diagnose_crnp(station_id: str = "PC"):
    """CRNP 문제 긴급 진단"""
    
    print("🚨 CRNP 긴급 진단 시작")
    print("=" * 60)
    
    # 1. 원본 CRNP 파일 확인
    print("1️⃣ 원본 CRNP 파일 확인")
    crnp_folder = project_root / "data" / "input" / station_id / "crnp"
    
    if not crnp_folder.exists():
        print(f"❌ CRNP 폴더 없음: {crnp_folder}")
        return False
        
    crnp_files = list(crnp_folder.glob("*.xlsx")) + list(crnp_folder.glob("*.csv"))
    if not crnp_files:
        print(f"❌ CRNP 파일 없음: {crnp_folder}")
        return False
        
    crnp_file = crnp_files[0]
    print(f"✅ CRNP 파일 발견: {crnp_file.name}")
    
    # 2. 원본 파일 직접 분석
    print("\n2️⃣ 원본 파일 내용 분석")
    
    # TOA5 헤더 읽기
    header_df = pd.read_excel(crnp_file, header=None, nrows=6)
    print("헤더 구조:")
    for i in range(len(header_df)):
        row_data = header_df.iloc[i, :].tolist()[:8]  # 처음 8개만
        print(f"  행 {i}: {row_data}")
    
    # 실제 데이터 읽기 (4행 스킵)
    data_df = pd.read_excel(crnp_file, skiprows=4)
    print(f"\n데이터 크기: {data_df.shape}")
    print(f"원본 컬럼: {list(data_df.columns)}")
    
    if len(data_df) > 0:
        print("첫 3행 데이터:")
        for i in range(min(3, len(data_df))):
            row_data = data_df.iloc[i, :].tolist()[:8]
            print(f"  [{i}] {row_data}")
    
    # 3. 전처리된 파일 확인
    print("\n3️⃣ 전처리된 파일 확인")
    processed_file = project_root / "data" / "output" / station_id / "preprocessed" / f"{station_id}_CRNP_input.xlsx"
    
    if processed_file.exists():
        print(f"✅ 전처리 파일 존재: {processed_file}")
        
        processed_df = pd.read_excel(processed_file)
        print(f"전처리 데이터 크기: {processed_df.shape}")
        print(f"전처리 컬럼: {list(processed_df.columns)}")
        
        if len(processed_df) > 0:
            print("전처리 첫 3행:")
            for i in range(min(3, len(processed_df))):
                row_data = processed_df.iloc[i, :].tolist()[:5]
                print(f"  [{i}] {row_data}")
                
            # 타임스탬프 분석
            if 'timestamp' in processed_df.columns:
                ts_data = processed_df['timestamp']
                print(f"\n타임스탬프 분석:")
                print(f"  유효한 타임스탬프: {ts_data.notna().sum()}/{len(ts_data)}")
                if ts_data.notna().sum() > 0:
                    print(f"  날짜 범위: {ts_data.min()} ~ {ts_data.max()}")
                    
            # 중성자 카운트 분석
            if 'N_counts' in processed_df.columns:
                neutron_data = processed_df['N_counts']
                print(f"\n중성자 카운트 분석:")
                print(f"  유효한 값: {neutron_data.notna().sum()}/{len(neutron_data)}")
                if neutron_data.notna().sum() > 0:
                    print(f"  범위: {neutron_data.min()} ~ {neutron_data.max()}")
                    print(f"  평균: {neutron_data.mean():.1f}")
                    
        else:
            print("❌ 전처리 파일이 비어있음!")
    else:
        print(f"❌ 전처리 파일 없음: {processed_file}")
        
    # 4. 캘리브레이션 기간 vs 데이터 기간 비교
    print("\n4️⃣ 캘리브레이션 기간 vs 데이터 기간")
    
    cal_start = pd.to_datetime('2024-08-17')
    cal_end = pd.to_datetime('2024-08-25')
    print(f"캘리브레이션 기간: {cal_start.date()} ~ {cal_end.date()}")
    
    if processed_file.exists() and len(processed_df) > 0 and 'timestamp' in processed_df.columns:
        data_start = processed_df['timestamp'].min()
        data_end = processed_df['timestamp'].max()
        print(f"실제 데이터 기간: {data_start.date()} ~ {data_end.date()}")
        
        # 겹치는 기간 확인
        overlap_mask = (processed_df['timestamp'] >= cal_start) & (processed_df['timestamp'] <= cal_end)
        overlap_count = overlap_mask.sum()
        print(f"겹치는 데이터: {overlap_count}개")
        
        if overlap_count == 0:
            print("⚠️ 캘리브레이션 기간에 데이터가 없음!")
            print("해결책: 캘리브레이션 기간을 실제 데이터 기간으로 조정")
            
    return True

def emergency_fix_crnp(station_id: str = "PC"):
    """CRNP 문제 긴급 수정"""
    
    print("\n🔧 CRNP 긴급 수정 시작")
    print("=" * 60)
    
    # 원본 파일 직접 처리
    crnp_folder = project_root / "data" / "input" / station_id / "crnp"
    crnp_files = list(crnp_folder.glob("*.xlsx")) + list(crnp_folder.glob("*.csv"))
    crnp_file = crnp_files[0]
    
    print(f"원본 파일 처리: {crnp_file.name}")
    
    # 1. 헤더에서 실제 컬럼명 추출
    header_df = pd.read_excel(crnp_file, header=None, nrows=6)
    
    # 행 1에서 컬럼명 추출 (NaN이 아닌 것들)
    actual_columns = []
    if len(header_df) > 1:
        row1_data = header_df.iloc[1, :].tolist()
        actual_columns = [str(col) for col in row1_data if pd.notna(col)]
        
    print(f"추출된 컬럼명: {actual_columns}")
    
    # 2. 데이터 읽기 (4행 스킵)
    data_df = pd.read_excel(crnp_file, skiprows=4)
    
    # 3. 컬럼명 적용
    if actual_columns and len(actual_columns) <= len(data_df.columns):
        final_columns = actual_columns + [f"Col_{i}" for i in range(len(actual_columns), len(data_df.columns))]
        data_df.columns = final_columns[:len(data_df.columns)]
        print(f"적용된 컬럼명: {list(data_df.columns)}")
    
    # 4. 표준 컬럼 매핑 (수동으로 확실하게)
    standard_mapping = {}
    
    # 타임스탬프 찾기
    for col in data_df.columns:
        col_lower = str(col).lower()
        if 'timestamp' in col_lower or 'time' in col_lower:
            standard_mapping['Timestamp'] = col
            break
    
    # 기본 기상 변수들
    for col in data_df.columns:
        col_lower = str(col).lower()
        if 'temp' in col_lower and 'Timestamp' not in standard_mapping.values():
            standard_mapping['Ta'] = col
        elif 'rh' in col_lower and 'humidity' not in [v for v in standard_mapping.values()]:
            standard_mapping['RH'] = col
        elif 'press' in col_lower:
            standard_mapping['Pa'] = col
        elif 'record' in col_lower:
            standard_mapping['RN'] = col
            
    # 중성자 카운트 찾기 (가장 중요!)
    neutron_col = None
    for col in data_df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['neutron', 'cosmic', 'crnp', 'count']):
            # 숫자 데이터인지 확인
            try:
                numeric_data = pd.to_numeric(data_df[col], errors='coerce')
                if numeric_data.notna().sum() > 0:
                    neutron_col = col
                    standard_mapping['N_counts'] = col
                    break
            except:
                continue
                
    # 마지막 컬럼이 숫자면 중성자 카운트일 가능성
    if not neutron_col and len(data_df.columns) > 0:
        last_col = data_df.columns[-1]
        try:
            numeric_data = pd.to_numeric(data_df[last_col], errors='coerce')
            if numeric_data.notna().sum() > 0 and numeric_data.mean() > 10:
                neutron_col = last_col
                standard_mapping['N_counts'] = last_col
                print(f"마지막 컬럼을 중성자 카운트로 사용: {last_col}")
        except:
            pass
    
    print(f"매핑 결과: {standard_mapping}")
    
    # 5. 표준 데이터프레임 생성
    standard_columns = ['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts']
    final_df = pd.DataFrame()
    
    for std_col in standard_columns:
        if std_col in standard_mapping:
            final_df[std_col] = data_df[standard_mapping[std_col]]
        else:
            final_df[std_col] = np.nan
            
    # 6. 타임스탬프 처리 (이미 datetime인 경우)
    if 'Timestamp' in final_df.columns:
        if final_df['Timestamp'].dtype == 'object':
            # datetime 객체인지 확인
            sample = final_df['Timestamp'].iloc[0] if len(final_df) > 0 else None
            if isinstance(sample, pd.Timestamp):
                final_df['timestamp'] = final_df['Timestamp']
            else:
                final_df['timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')
        else:
            final_df['timestamp'] = final_df['Timestamp']
    
    # 유효한 타임스탬프만 유지
    initial_count = len(final_df)
    final_df = final_df.dropna(subset=['timestamp'])
    print(f"타임스탬프 필터링: {initial_count} → {len(final_df)} 레코드")
    
    if len(final_df) > 0:
        print(f"최종 데이터 범위: {final_df['timestamp'].min()} ~ {final_df['timestamp'].max()}")
        
        # 중성자 카운트 확인
        if 'N_counts' in final_df.columns:
            neutron_valid = final_df['N_counts'].notna().sum()
            print(f"유효한 중성자 카운트: {neutron_valid}/{len(final_df)}")
            
        # 7. 파일 저장
        output_dir = project_root / "data" / "output" / station_id / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{station_id}_CRNP_input.xlsx"
        
        final_df.to_excel(output_file, index=False)
        print(f"✅ 수정된 파일 저장: {output_file}")
        
        return True
    else:
        print("❌ 처리 후 데이터가 없음")
        return False

def main():
    """메인 실행"""
    
    import argparse
    parser = argparse.ArgumentParser(description="CRNP 긴급 진단 및 수정")
    parser.add_argument("--station", "-s", default="PC", help="관측소 ID")
    parser.add_argument("--fix", "-f", action="store_true", help="문제 자동 수정")
    
    args = parser.parse_args()
    
    # 진단 실행
    diagnosis_ok = emergency_diagnose_crnp(args.station)
    
    if not diagnosis_ok:
        print("❌ 진단 실패")
        return 1
        
    # 수정 실행
    if args.fix:
        fix_ok = emergency_fix_crnp(args.station)
        if fix_ok:
            print("\n✅ 수정 완료! 이제 캘리브레이션을 다시 시도하세요:")
            print(f"python scripts/run_calibration.py --station {args.station} --start 2024-08-01 --end 2024-08-02")
        else:
            print("❌ 수정 실패")
            return 1
    else:
        print("\n💡 수정하려면:")
        print(f"python scripts/emergency_crnp_debug.py --station {args.station} --fix")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())