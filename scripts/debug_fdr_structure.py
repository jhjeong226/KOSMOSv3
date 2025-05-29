# scripts/debug_fdr_structure.py

"""
FDR 파일 구조 디버깅 스크립트
실제 FDR 파일의 구조를 확인하여 전처리 문제를 해결합니다.
"""

import pandas as pd
import os
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_fdr_file(file_path: str, station_id: str):
    """단일 FDR 파일 분석"""
    print(f"\n{'='*60}")
    print(f"파일 분석: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        # 1. 기존 방식으로 읽기 시도 (2행 스킵)
        print("📋 2행 스킵 후 구조 (기존 방식):")
        df_skip2 = pd.read_csv(file_path, skiprows=2, nrows=10)
        print(f"컬럼 수: {len(df_skip2.columns)}")
        print(f"컬럼명: {list(df_skip2.columns)}")
        
        # 2. 필요한 컬럼 확인
        required_columns = ['Timestamps', ' m3/m3 Water Content', ' m3/m3 Water Content.1', ' m3/m3 Water Content.2']
        missing_columns = [col for col in required_columns if col not in df_skip2.columns]
        
        if missing_columns:
            print(f"❌ 누락된 필수 컬럼: {missing_columns}")
            print("📋 사용 가능한 컬럼:")
            for i, col in enumerate(df_skip2.columns):
                print(f"  {i}: '{col}'")
        else:
            print("✅ 모든 필수 컬럼이 존재합니다!")
            
        # 3. 첫 5행 데이터 확인
        print("\n📊 첫 5행 데이터:")
        if not missing_columns:
            sample_data = df_skip2[required_columns].head()
            print(sample_data)
        else:
            print(df_skip2.head())
            
        # 4. 전체 파일 행 수 확인
        total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
        print(f"\n📊 총 행 수: {total_lines}")
        
        return len(missing_columns) == 0
        
    except Exception as e:
        print(f"❌ 파일 분석 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FDR 파일 구조 디버깅")
    parser.add_argument("--station", "-s", default="HC", help="관측소 ID (HC or PC)")
    parser.add_argument("--file", "-f", help="특정 파일만 분석")
    
    args = parser.parse_args()
    
    print("🔍 FDR 파일 구조 디버깅 시작")
    print("="*60)
    
    # FDR 폴더 경로
    fdr_folder = project_root / "data" / "input" / args.station / "fdr"
    
    if not fdr_folder.exists():
        print(f"❌ FDR 폴더가 없습니다: {fdr_folder}")
        return
    
    # CSV 파일 찾기
    csv_files = list(fdr_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ CSV 파일이 없습니다: {fdr_folder}")
        return
    
    print(f"📁 찾은 CSV 파일: {len(csv_files)}개")
    
    if args.file:
        # 특정 파일만 분석
        target_file = fdr_folder / args.file
        if target_file.exists():
            analyze_fdr_file(str(target_file), args.station)
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {target_file}")
    else:
        # 처음 3개 파일만 분석
        for i, file_path in enumerate(csv_files[:3]):
            analyze_fdr_file(str(file_path), args.station)
            
        print(f"\n💡 나머지 {len(csv_files)-3}개 파일이 더 있습니다.")
        print("특정 파일 분석: python scripts/debug_fdr_structure.py --file 파일명.csv")
    
    print(f"\n🎯 다음 단계:")
    print("1. 위 분석 결과를 바탕으로 컬럼명과 스킵 행 수 확인")
    print("2. FDRProcessor의 컬럼 매핑 수정")
    print("3. DataValidator의 검증 기준 완화")

if __name__ == "__main__":
    main()