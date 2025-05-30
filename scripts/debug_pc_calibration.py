# scripts/debug_pc_calibration.py

"""
PC 관측소 캘리브레이션 문제 디버깅 스크립트
실제 데이터 기간과 매칭 문제를 확인합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_pc_calibration():
    """PC 관측소 캘리브레이션 디버깅"""
    
    print("🔍 PC 관측소 캘리브레이션 디버깅")
    print("=" * 60)
    
    # 1. 파일 경로 설정
    output_dir = project_root / "data" / "output" / "PC" / "preprocessed"
    fdr_file = output_dir / "PC_FDR_input.xlsx"
    crnp_file = output_dir / "PC_CRNP_input.xlsx"
    
    if not fdr_file.exists():
        print(f"❌ FDR 파일 없음: {fdr_file}")
        return
        
    if not crnp_file.exists():
        print(f"❌ CRNP 파일 없음: {crnp_file}")
        return
        
    # 2. FDR 데이터 분석
    print("\n📊 FDR 데이터 분석:")
    fdr_data = pd.read_excel(fdr_file)
    print(f"  총 레코드: {len(fdr_data)}")
    print(f"  컬럼: {list(fdr_data.columns)}")
    
    if 'Date' in fdr_data.columns:
        fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
        fdr_date_min = fdr_data['Date'].min()
        fdr_date_max = fdr_data['Date'].max()
        print(f"  날짜 범위: {fdr_date_min.date()} ~ {fdr_date_max.date()}")
        
        # 필수 컬럼 확인
        required_cols = ['theta_v', 'FDR_depth', 'distance_from_station']
        missing_cols = [col for col in required_cols if col not in fdr_data.columns]
        if missing_cols:
            print(f"  ❌ 누락 컬럼: {missing_cols}")
        else:
            print(f"  ✅ 필수 컬럼 모두 존재")
            
        # 깊이별 데이터 확인
        if 'FDR_depth' in fdr_data.columns:
            depths = sorted(fdr_data['FDR_depth'].unique())
            print(f"  측정 깊이: {depths}")
            
        # 센서별 데이터 확인
        if 'id' in fdr_data.columns:
            sensors = sorted(fdr_data['id'].unique())
            print(f"  센서 ID: {sensors[:5]}{'...' if len(sensors) > 5 else ''} (총 {len(sensors)}개)")
    
    # 3. CRNP 데이터 분석
    print("\n🛰️  CRNP 데이터 분석:")
    crnp_data = pd.read_excel(crnp_file)
    print(f"  총 레코드: {len(crnp_data)}")
    print(f"  컬럼: {list(crnp_data.columns)}")
    
    # 타임스탬프 처리
    if 'timestamp' in crnp_data.columns:
        crnp_data['timestamp'] = pd.to_datetime(crnp_data['timestamp'], errors='coerce')
        valid_timestamps = crnp_data['timestamp'].notna().sum()
        print(f"  유효 타임스탬프: {valid_timestamps}/{len(crnp_data)}")
        
        if valid_timestamps > 0:
            crnp_date_min = crnp_data['timestamp'].min()
            crnp_date_max = crnp_data['timestamp'].max()
            print(f"  날짜 범위: {crnp_date_min.date()} ~ {crnp_date_max.date()}")
    
    # 4. 기간 겹침 확인
    print("\n🔗 데이터 기간 겹침 확인:")
    
    if 'Date' in fdr_data.columns and 'timestamp' in crnp_data.columns:
        fdr_dates = set(fdr_data['Date'].dt.date)
        crnp_dates = set(crnp_data['timestamp'].dt.date)
        
        overlap_dates = fdr_dates.intersection(crnp_dates)
        
        print(f"  FDR 날짜 수: {len(fdr_dates)}")
        print(f"  CRNP 날짜 수: {len(crnp_dates)}")
        print(f"  겹치는 날짜: {len(overlap_dates)}")
        
        if overlap_dates:
            min_overlap = min(overlap_dates)
            max_overlap = max(overlap_dates)
            print(f"  겹치는 기간: {min_overlap} ~ {max_overlap}")
            
            # 캘리브레이션 추천 기간
            if len(overlap_dates) >= 7:
                # 충분한 겹치는 기간이 있으면 일주일 단위로 추천
                sorted_overlap = sorted(overlap_dates)
                recommended_start = sorted_overlap[0]
                recommended_end = min(sorted_overlap[6], max_overlap)  # 최소 7일
                print(f"  🎯 추천 캘리브레이션 기간: {recommended_start} ~ {recommended_end}")
            else:
                print(f"  ⚠️  겹치는 기간이 너무 짧음 ({len(overlap_dates)}일)")
        else:
            print(f"  ❌ 겹치는 날짜가 없음!")
    
    # 5. 토양수분 데이터 품질 확인
    print("\n💧 토양수분 데이터 품질:")
    if 'theta_v' in fdr_data.columns:
        valid_theta = fdr_data['theta_v'].notna().sum()
        total_theta = len(fdr_data)
        completeness = (valid_theta / total_theta) * 100
        
        print(f"  유효 데이터: {valid_theta}/{total_theta} ({completeness:.1f}%)")
        
        if valid_theta > 0:
            theta_mean = fdr_data['theta_v'].mean()
            theta_std = fdr_data['theta_v'].std()
            theta_min = fdr_data['theta_v'].min()
            theta_max = fdr_data['theta_v'].max()
            
            print(f"  평균: {theta_mean:.3f}")
            print(f"  표준편차: {theta_std:.3f}")
            print(f"  범위: {theta_min:.3f} ~ {theta_max:.3f}")
            
            # 이상값 확인
            outliers = ((fdr_data['theta_v'] < 0) | (fdr_data['theta_v'] > 1)).sum()
            if outliers > 0:
                print(f"  ⚠️  이상값: {outliers}개 (범위 밖)")
    
    # 6. 중성자 카운트 데이터 확인
    print("\n⚛️  중성자 카운트 데이터:")
    if 'N_counts' in crnp_data.columns:
        valid_neutrons = crnp_data['N_counts'].notna().sum()
        total_neutrons = len(crnp_data)
        n_completeness = (valid_neutrons / total_neutrons) * 100
        
        print(f"  유효 데이터: {valid_neutrons}/{total_neutrons} ({n_completeness:.1f}%)")
        
        if valid_neutrons > 0:
            n_mean = crnp_data['N_counts'].mean()
            n_std = crnp_data['N_counts'].std()
            n_min = crnp_data['N_counts'].min()
            n_max = crnp_data['N_counts'].max()
            
            print(f"  평균: {n_mean:.1f}")
            print(f"  표준편차: {n_std:.1f}")
            print(f"  범위: {n_min:.1f} ~ {n_max:.1f}")
    
    # 7. 해결책 제안
    print("\n🎯 해결책 제안:")
    
    if 'overlap_dates' in locals() and overlap_dates:
        if len(overlap_dates) >= 3:
            # 겹치는 기간으로 캘리브레이션 실행
            sorted_overlap = sorted(overlap_dates)
            start_date = sorted_overlap[0]
            end_date = sorted_overlap[-1]
            
            print(f"1. 겹치는 기간으로 캘리브레이션 실행:")
            print(f"   python scripts/run_calibration.py --station PC --start {start_date} --end {end_date}")
        else:
            print(f"1. 데이터 기간이 부족합니다. 추가 데이터 확보 필요")
    else:
        print(f"1. FDR과 CRNP 데이터 기간이 겹치지 않습니다.")
        print(f"2. 데이터 재확인 또는 다른 기간의 데이터 필요")
    
    print(f"2. 강제 재처리 (문제 해결 후):")
    print(f"   python scripts/run_calibration.py --station PC --force")


if __name__ == "__main__":
    debug_pc_calibration()