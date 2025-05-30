# scripts/debug_crnp_structure.py

"""
CRNP 파일 구조 디버깅 스크립트
실제 CRNP 파일의 구조를 확인하여 타임스탬프 파싱 문제를 해결합니다.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import chardet

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def detect_file_encoding(file_path: str) -> str:
    """파일 인코딩 감지"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def analyze_crnp_file(file_path: str, station_id: str):
    """단일 CRNP 파일 상세 분석"""
    print(f"\n{'='*80}")
    print(f"CRNP 파일 분석: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        # 파일 기본 정보
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📁 파일 정보:")
        print(f"   크기: {file_size:.2f} MB")
        print(f"   확장자: {file_ext}")
        
        if file_ext in ['.csv', '.txt']:
            encoding = detect_file_encoding(file_path)
            print(f"   인코딩: {encoding}")
        
        # 1. 파일 헤더 구조 분석 (처음 10줄)
        print(f"\n📋 헤더 구조 분석 (처음 10줄):")
        print("-" * 50)
        
        if file_ext in ['.xlsx', '.xls']:
            # Excel 파일
            header_df = pd.read_excel(file_path, header=None, nrows=10)
        else:
            # CSV 파일
            encoding = detect_file_encoding(file_path)
            header_df = pd.read_csv(file_path, header=None, nrows=10, encoding=encoding)
        
        for i, row in header_df.iterrows():
            row_data = [str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in row.values[:5]]
            print(f"   행 {i}: {row_data}")
        
        # 2. TOA5 형식 감지
        print(f"\n🔍 TOA5 형식 감지:")
        toa5_detected = False
        if len(header_df) >= 4:
            first_cell = str(header_df.iloc[0, 0]).upper()
            if 'TOA5' in first_cell:
                toa5_detected = True
                print("   ✅ TOA5 형식 감지됨 (Campbell Scientific 로거)")
                print(f"   첫 번째 셀: {first_cell}")
                
                # TOA5 메타데이터 분석
                if len(header_df) >= 2:
                    station_name = str(header_df.iloc[0, 1]) if len(header_df.columns) > 1 else "Unknown"
                    model_name = str(header_df.iloc[0, 2]) if len(header_df.columns) > 2 else "Unknown"
                    print(f"   관측소: {station_name}")
                    print(f"   모델: {model_name}")
                
                # 컬럼명 (3번째 행)
                if len(header_df) >= 3:
                    column_names = header_df.iloc[2, :].tolist()
                    print(f"   컬럼명 (행 2): {column_names[:8]}...")
                    
                # 단위 (4번째 행)
                if len(header_df) >= 4:
                    units = header_df.iloc[3, :].tolist()
                    print(f"   단위 (행 3): {units[:8]}...")
            else:
                print("   ❌ TOA5 형식 아님")
        else:
            print("   ❌ 헤더가 너무 짧음")
        
        # 3. 적절한 헤더 행 설정으로 데이터 로드
        print(f"\n📊 데이터 로드 테스트:")
        
        skip_rows = 4 if toa5_detected else 0
        print(f"   헤더 스킵: {skip_rows}행")
        
        if file_ext in ['.xlsx', '.xls']:
            if toa5_detected:
                df = pd.read_excel(file_path, skiprows=skip_rows, nrows=20)
            else:
                df = pd.read_excel(file_path, nrows=20)
        else:
            encoding = detect_file_encoding(file_path)
            if toa5_detected:
                df = pd.read_csv(file_path, skiprows=skip_rows, nrows=20, encoding=encoding)
            else:
                df = pd.read_csv(file_path, nrows=20, encoding=encoding)
        
        print(f"   로드된 데이터 크기: {df.shape}")
        print(f"   컬럼 수: {len(df.columns)}")
        print(f"   컬럼명: {list(df.columns)}")
        
        # 4. 타임스탬프 컬럼 분석
        print(f"\n⏰ 타임스탬프 분석:")
        
        # 첫 번째 컬럼이 타임스탬프일 가능성이 높음
        timestamp_col = df.columns[0]
        timestamp_data = df[timestamp_col]
        
        print(f"   타임스탬프 컬럼: '{timestamp_col}'")
        print(f"   샘플 값들:")
        
        for i, val in enumerate(timestamp_data.head(10)):
            print(f"     [{i}] {repr(val)} (타입: {type(val).__name__})")
        
        # 5. 타임스탬프 형식 추론
        print(f"\n🔧 타임스탬프 형식 추론:")
        
        sample_values = timestamp_data.head(5).tolist()
        
        # 숫자 형식 확인
        try:
            numeric_series = pd.to_numeric(timestamp_data, errors='coerce')
            numeric_count = numeric_series.notna().sum()
            numeric_ratio = numeric_count / len(timestamp_data) * 100
            
            print(f"   숫자 변환 가능: {numeric_count}/{len(timestamp_data)} ({numeric_ratio:.1f}%)")
            
            if numeric_ratio > 80:
                print("   ✅ Excel 숫자 형식 타임스탬프로 추정")
                
                # Excel 날짜 변환 테스트
                if numeric_count > 0:
                    sample_numeric = numeric_series.dropna().iloc[0]
                    print(f"   샘플 숫자값: {sample_numeric}")
                    
                    # Excel epoch 변환 테스트
                    base_date = pd.to_datetime('1899-12-30')
                    converted_date = base_date + pd.to_timedelta(sample_numeric, unit='D')
                    print(f"   Excel 변환 결과: {converted_date}")
                    
                    # 전체 변환 테스트
                    converted_series = base_date + pd.to_timedelta(numeric_series, unit='D')
                    valid_range = ((converted_series >= pd.to_datetime('2020-01-01')) & 
                                  (converted_series <= pd.to_datetime('2030-12-31')))
                    valid_count = valid_range.sum()
                    print(f"   유효한 날짜 범위: {valid_count}/{len(timestamp_data)} ({valid_count/len(timestamp_data)*100:.1f}%)")
                    
                    if valid_count > 0:
                        print(f"   변환된 날짜 범위: {converted_series.min()} ~ {converted_series.max()}")
            else:
                print("   ❌ 숫자 형식 아님")
                
        except Exception as e:
            print(f"   ❌ 숫자 변환 실패: {e}")
        
        # 문자열 형식 확인
        print(f"\n   문자열 형식 분석:")
        string_samples = [str(val) for val in sample_values[:5]]
        for i, sample in enumerate(string_samples):
            print(f"     [{i}] '{sample}' (길이: {len(sample)})")
        
        # 일반적인 날짜 형식 테스트
        common_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
        ]
        
        for fmt in common_formats[:4]:  # 주요 형식만 테스트
            try:
                parsed = pd.to_datetime(timestamp_data, format=fmt, errors='coerce')
                valid_count = parsed.notna().sum()
                if valid_count > 0:
                    print(f"   형식 '{fmt}': {valid_count}/{len(timestamp_data)} 성공")
                    if valid_count > len(timestamp_data) * 0.8:
                        print(f"     ✅ 추천 형식! 날짜 범위: {parsed.min()} ~ {parsed.max()}")
            except:
                continue
        
        # 6. 데이터 품질 확인
        print(f"\n📈 데이터 품질 확인:")
        
        for col in df.columns[:8]:  # 처음 8개 컬럼만
            non_null_count = df[col].notna().sum()
            completeness = non_null_count / len(df) * 100
            print(f"   {col}: {completeness:.1f}% 완성도 ({non_null_count}/{len(df)})")
        
        # 7. 추천 처리 방법
        print(f"\n💡 추천 처리 방법:")
        
        if toa5_detected:
            print("   1. TOA5 형식이므로 4행 스킵 (skiprows=4)")
            print("   2. 표준 CRNP 컬럼명 강제 적용")
            
            if numeric_ratio > 80:
                print("   3. Excel 숫자 타임스탬프 변환 적용")
                print("      base_date = pd.to_datetime('1899-12-30')")
                print("      timestamp = base_date + pd.to_timedelta(numeric_values, unit='D')")
            else:
                print("   3. 문자열 타임스탬프 파싱 적용")
        else:
            print("   1. 헤더 스킵 없음")
            print("   2. 컬럼명 자동 매핑 또는 위치 기반 매핑")
            print("   3. 첫 번째 컬럼을 타임스탬프로 처리")
        
        return True
        
    except Exception as e:
        print(f"❌ 파일 분석 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRNP 파일 구조 디버깅")
    parser.add_argument("--station", "-s", default="PC", help="관측소 ID (HC or PC)")
    parser.add_argument("--file", "-f", help="특정 파일만 분석")
    
    args = parser.parse_args()
    
    print("🔍 CRNP 파일 구조 디버깅 시작")
    print("="*80)
    
    # CRNP 폴더 경로
    crnp_folder = project_root / "data" / "input" / args.station / "crnp"
    
    if not crnp_folder.exists():
        print(f"❌ CRNP 폴더가 없습니다: {crnp_folder}")
        return
    
    # 지원하는 파일 찾기
    excel_files = list(crnp_folder.glob("*.xlsx")) + list(crnp_folder.glob("*.xls"))
    csv_files = list(crnp_folder.glob("*.csv"))
    all_files = excel_files + csv_files
    
    if not all_files:
        print(f"❌ CRNP 데이터 파일이 없습니다: {crnp_folder}")
        return
    
    print(f"📁 찾은 CRNP 파일: {len(all_files)}개")
    for i, file_path in enumerate(all_files):
        print(f"   [{i}] {file_path.name} ({file_path.stat().st_size/(1024*1024):.1f} MB)")
    
    if args.file:
        # 특정 파일만 분석
        target_file = crnp_folder / args.file
        if target_file.exists():
            analyze_crnp_file(str(target_file), args.station)
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {target_file}")
    else:
        # 첫 번째 파일 분석
        if all_files:
            analyze_crnp_file(str(all_files[0]), args.station)
            
            if len(all_files) > 1:
                print(f"\n💡 나머지 {len(all_files)-1}개 파일이 더 있습니다.")
                print("특정 파일 분석: python scripts/debug_crnp_structure.py --file 파일명.xlsx")
    
    print(f"\n🎯 다음 단계:")
    print("1. 위 분석 결과를 바탕으로 CRNP 파일 형식 확인")
    print("2. CRNPProcessor의 파싱 로직 수정")
    print("3. 타임스탬프 변환 방법 적용")
    print("4. 전처리 재실행")

if __name__ == "__main__":
    main()