# scripts/debug_calibration_data.py

"""
캘리브레이션 데이터를 직접 확인하는 간단한 스크립트
R² = -111이 나오는 원인을 찾아봅시다.
"""

import pandas as pd
import numpy as np
import crnpy
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_calibration_data():
    """캘리브레이션 데이터 직접 디버깅"""
    
    print("🔍 캘리브레이션 데이터 직접 확인")
    print("=" * 60)
    
    # 1. 데이터 로드
    output_dir = project_root / "data" / "output" / "PC" / "preprocessed"
    fdr_file = output_dir / "PC_FDR_input.xlsx"
    crnp_file = output_dir / "PC_CRNP_input.xlsx"
    
    print("📂 데이터 로드 중...")
    fdr_data = pd.read_excel(fdr_file)
    crnp_data = pd.read_excel(crnp_file)
    
    print(f"  FDR: {len(fdr_data)} records")
    print(f"  CRNP: {len(crnp_data)} records")
    
    # 2. 캘리브레이션 기간 데이터 필터링
    cal_start = '2024-08-17'
    cal_end = '2024-08-25'
    
    print(f"\n📅 캘리브레이션 기간: {cal_start} ~ {cal_end}")
    
    # FDR 필터링
    fdr_data['Date'] = pd.to_datetime(fdr_data['Date'])
    fdr_cal = fdr_data[(fdr_data['Date'] >= cal_start) & (fdr_data['Date'] <= cal_end)]
    
    # CRNP 필터링
    crnp_data['timestamp'] = pd.to_datetime(crnp_data['timestamp'])
    crnp_cal = crnp_data[(crnp_data['timestamp'] >= cal_start) & (crnp_data['timestamp'] <= cal_end)]
    
    print(f"  필터링된 FDR: {len(fdr_cal)} records")
    print(f"  필터링된 CRNP: {len(crnp_cal)} records")
    
    # 3. FDR 토양수분 확인
    print(f"\n🌱 FDR 토양수분 데이터:")
    if 'theta_v' in fdr_cal.columns:
        print(f"  범위: {fdr_cal['theta_v'].min():.3f} ~ {fdr_cal['theta_v'].max():.3f}")
        print(f"  평균: {fdr_cal['theta_v'].mean():.3f}")
        print(f"  표준편차: {fdr_cal['theta_v'].std():.3f}")
        
        # 일별 평균 계산
        fdr_cal['date'] = fdr_cal['Date'].dt.date
        fdr_daily = fdr_cal.groupby('date')['theta_v'].mean()
        
        print(f"  일별 데이터: {len(fdr_daily)}일")
        print(f"  일별 범위: {fdr_daily.min():.3f} ~ {fdr_daily.max():.3f}")
    else:
        print("  ❌ theta_v 컬럼 없음!")
        return
    
    # 4. CRNP 중성자 카운트 확인
    print(f"\n⚛️  CRNP 중성자 카운트:")
    if 'N_counts' in crnp_cal.columns:
        print(f"  범위: {crnp_cal['N_counts'].min():.1f} ~ {crnp_cal['N_counts'].max():.1f}")
        print(f"  평균: {crnp_cal['N_counts'].mean():.1f}")
        print(f"  표준편차: {crnp_cal['N_counts'].std():.1f}")
        
        # 일별 평균 계산
        crnp_cal['date'] = crnp_cal['timestamp'].dt.date
        crnp_daily = crnp_cal.groupby('date')['N_counts'].mean()
        
        print(f"  일별 데이터: {len(crnp_daily)}일")
        print(f"  일별 범위: {crnp_daily.min():.1f} ~ {crnp_daily.max():.1f}")
    else:
        print("  ❌ N_counts 컬럼 없음!")
        return
    
    # 5. 매칭된 일별 데이터 생성
    print(f"\n🔗 일별 데이터 매칭:")
    
    # 공통 날짜 찾기
    fdr_dates = set(fdr_daily.index)
    crnp_dates = set(crnp_daily.index)
    common_dates = fdr_dates.intersection(crnp_dates)
    
    print(f"  공통 날짜: {len(common_dates)}일")
    
    if len(common_dates) == 0:
        print("  ❌ 공통 날짜가 없습니다!")
        return
    
    # 매칭된 데이터 생성
    matched_data = []
    for date in sorted(common_dates):
        matched_data.append({
            'date': date,
            'Field_SM': fdr_daily[date],
            'Daily_N': crnp_daily[date]
        })
    
    matched_df = pd.DataFrame(matched_data)
    print(f"  매칭 완료: {len(matched_df)}일")
    
    # 6. crnpy로 VWC 계산 테스트
    print(f"\n🧪 crnpy VWC 계산 테스트:")
    
    # 기본 매개변수
    bulk_density = 1.2
    lattice_water = crnpy.lattice_water(clay_content=0.35)
    
    print(f"  벌크밀도: {bulk_density}")
    print(f"  격자수: {lattice_water:.4f}")
    
    # 여러 N0 값으로 테스트
    test_N0_values = [1000, 1500, 2000, 2500, 3000]
    
    print(f"\n📊 N0 값별 VWC 계산 결과:")
    print("N0\t\tVWC 범위\t\tVWC 평균\t\tRMSE")
    print("-" * 60)
    
    best_rmse = 1e6
    best_N0 = 1000
    
    for N0 in test_N0_values:
        try:
            # VWC 계산
            vwc = crnpy.counts_to_vwc(
                matched_df['Daily_N'],
                N0=N0,
                bulk_density=bulk_density,
                Wlat=lattice_water,
                Wsoc=0.01
            )
            
            # RMSE 계산
            rmse = np.sqrt(np.mean((vwc - matched_df['Field_SM']) ** 2))
            
            print(f"{N0}\t\t{vwc.min():.3f}-{vwc.max():.3f}\t\t{vwc.mean():.3f}\t\t{rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_N0 = N0
                best_vwc = vwc
                
        except Exception as e:
            print(f"{N0}\t\t❌ 오류: {e}")
    
    print(f"\n🎯 최적 결과: N0={best_N0}, RMSE={best_rmse:.4f}")
    
    # 7. R² 계산 확인
    if 'best_vwc' in locals():
        print(f"\n📈 성능 지표 상세 계산:")
        
        observed = matched_df['Field_SM'].values
        predicted = best_vwc
        
        print(f"  관측값 (FDR): {observed.min():.3f} ~ {observed.max():.3f} (평균: {observed.mean():.3f})")
        print(f"  예측값 (CRNP): {predicted.min():.3f} ~ {predicted.max():.3f} (평균: {predicted.mean():.3f})")
        
        # R² 계산 과정
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        
        print(f"  SS_res (잔차제곱합): {ss_res:.6f}")
        print(f"  SS_tot (총제곱합): {ss_tot:.6f}")
        
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
            print(f"  R² = 1 - (SS_res/SS_tot) = {r2:.6f}")
            
            if r2 < -10:
                print("  🚨 R²가 극단적으로 낮습니다!")
                print("  💡 가능한 원인:")
                print("     1. 예측값이 관측값과 완전히 다른 범위")
                print("     2. 중성자 카운트 → VWC 변환 공식 문제")
                print("     3. 매개변수 (N0, 벌크밀도, 격자수) 부적절")
        else:
            print("  ⚠️  SS_tot이 0입니다 (관측값이 모두 동일)")
    
    # 8. 간단한 시각화
    print(f"\n🎨 결과 시각화 생성...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 시계열 비교
        axes[0,0].plot(matched_df['date'], matched_df['Field_SM'], 'bo-', label='FDR')
        if 'best_vwc' in locals():
            axes[0,0].plot(matched_df['date'], best_vwc, 'ro-', label='CRNP')
        axes[0,0].set_title('시계열 비교')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 산점도
        if 'best_vwc' in locals():
            axes[0,1].scatter(matched_df['Field_SM'], best_vwc, s=100)
            min_val = min(matched_df['Field_SM'].min(), best_vwc.min())
            max_val = max(matched_df['Field_SM'].max(), best_vwc.max())
            axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
            axes[0,1].set_xlabel('FDR SM')
            axes[0,1].set_ylabel('CRNP VWC')
            axes[0,1].set_title('산점도')
            axes[0,1].legend()
        
        # 중성자 카운트
        axes[1,0].plot(matched_df['date'], matched_df['Daily_N'], 'go-')
        axes[1,0].set_title('중성자 카운트')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 히스토그램
        axes[1,1].hist(matched_df['Field_SM'], alpha=0.7, label='FDR', bins=10)
        if 'best_vwc' in locals():
            axes[1,1].hist(best_vwc, alpha=0.7, label='CRNP', bins=10)
        axes[1,1].set_title('분포 비교')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # 저장
        plot_file = project_root / "debug_calibration_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 시각화 저장: {plot_file}")
        
    except Exception as e:
        print(f"  ❌ 시각화 실패: {e}")
    
    # 9. 결과 데이터 저장
    print(f"\n💾 결과 데이터 저장...")
    
    try:
        # 디버깅 데이터 저장
        if 'best_vwc' in locals():
            debug_df = matched_df.copy()
            debug_df['CRNP_VWC'] = best_vwc
            debug_df['Residuals'] = best_vwc - matched_df['Field_SM']
            debug_df['N0_used'] = best_N0
            
            debug_file = project_root / "debug_calibration_data.xlsx"
            debug_df.to_excel(debug_file, index=False)
            
            print(f"  ✅ 데이터 저장: {debug_file}")
        
    except Exception as e:
        print(f"  ❌ 데이터 저장 실패: {e}")
    
    print(f"\n🎯 결론:")
    print(f"  매칭된 일수: {len(matched_df)}일")
    print(f"  최적 N0: {best_N0}")
    print(f"  최적 RMSE: {best_rmse:.4f}")
    
    if best_rmse > 0.1:
        print(f"  ⚠️  RMSE가 높습니다. 데이터 품질이나 매개변수를 확인해보세요.")
    
    print(f"\n📁 생성된 파일:")
    print(f"  - debug_calibration_analysis.png (시각화)")
    print(f"  - debug_calibration_data.xlsx (데이터)")


if __name__ == "__main__":
    debug_calibration_data()