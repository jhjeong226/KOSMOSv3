#!/usr/bin/env python3
"""
run_ridgeline.py

범용 Ridgeline Plot 생성기
전처리된 토양수분 데이터로부터 ridgeline plot 생성

특징:
- 겨울철 데이터 자동 제외 (11, 12, 1, 2, 3월)
- 관측소 컬럼 자동 감지
- 깊이 자동 감지 (10cm, 30cm, 60cm 등 유연 지원)
- Average 분포 자동 추가 (모든 관측소 데이터 합집합)
- 다양한 파일 형식 및 경로 지원

사용법:
    python script/run_ridgeline.py --station HC
    python script/run_ridgeline.py --station PC --input-dir data/input
    python script/run_ridgeline.py --station HC --depths 10cm 20cm --output-dir plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import sys
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# 설정
plt.style.use('default')
sns.set_palette("husl")

class RidgelineGenerator:
    """범용 Ridgeline Plot 생성기"""
    
    def __init__(self, station_id: str, base_dir: str = "data"):
        self.station_id = station_id
        self.base_dir = Path(base_dir)
        
        # 경로 설정
        self.output_dir = self.base_dir / "output" / station_id
        self.input_dir = self.base_dir / "input"
        
        # 결과 저장용
        self.station_columns = {}
        self.processed_data = {}
        
        print(f"🔧 초기화: 관측소 {station_id}")
        print(f"📂 데이터 디렉토리: {self.output_dir}")
        print(f"📂 입력 디렉토리: {self.input_dir}")
        print(f"📊 기본 출력 경로: {self.output_dir}/visualization/")
    
    def find_fdr_files(self, depths: list = None) -> dict:
        """FDR 파일 자동 탐색 및 깊이 자동 감지"""
        
        found_files = {}
        auto_detected_depths = set()
        
        # 가능한 파일 패턴들
        possible_patterns = [
            # 전처리된 파일들
            f"{self.station_id}_FDR_daily_depths.xlsx",
            f"{self.station_id}_FDR_input.xlsx", 
            f"{self.station_id}_fdr_data.xlsx",
            # 원본 파일들
            "HC_FDR_daily_depths.xlsx",
            "FDR_daily_depths.xlsx",
            f"{self.station_id}_daily_depths.xlsx",
        ]
        
        # 탐색 경로들
        search_paths = [
            self.output_dir / "preprocessed",
            self.output_dir,
            self.input_dir,
            Path("."),  # 현재 디렉토리
            Path("data"),
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for pattern in possible_patterns:
                file_path = search_path / pattern
                
                if file_path.exists():
                    try:
                        # Excel 파일 시트 확인
                        xl_file = pd.ExcelFile(file_path)
                        available_sheets = xl_file.sheet_names
                        print(f"📋 발견된 파일: {file_path}")
                        print(f"📊 사용 가능한 시트: {available_sheets}")
                        
                        # 깊이 시트 자동 감지
                        detected_depths = self._detect_depth_sheets(available_sheets)
                        auto_detected_depths.update(detected_depths)
                        
                        # 요청된 깊이가 있으면 해당 깊이만, 없으면 감지된 모든 깊이
                        target_depths = depths if depths else list(detected_depths)
                        
                        for depth in target_depths:
                            # 정확한 매칭 먼저 시도
                            if depth in available_sheets:
                                found_files[depth] = file_path
                                print(f"✅ {depth} 데이터 발견: {file_path}")
                            else:
                                # 유사한 시트명 찾기
                                depth_num = ''.join(filter(str.isdigit, depth))
                                for sheet in available_sheets:
                                    sheet_num = ''.join(filter(str.isdigit, sheet))
                                    if depth_num and sheet_num and depth_num == sheet_num:
                                        found_files[depth] = file_path
                                        print(f"✅ {depth} 데이터 매칭: {file_path} (시트: {sheet})")
                                        break
                        
                        if found_files:
                            break
                            
                    except Exception as e:
                        print(f"⚠️ 파일 확인 실패 {file_path}: {e}")
                        continue
            
            if found_files:
                break
        
        if not found_files:
            if auto_detected_depths:
                print(f"💡 감지된 깊이들: {sorted(auto_detected_depths)}")
                print("💡 예시: --depths 10cm 30cm 60cm")
            print(f"❌ FDR 데이터를 찾을 수 없습니다.")
            print("📁 다음 위치들을 확인했습니다:")
            for path in search_paths:
                print(f"   - {path}")
        
        return found_files
    
    def _detect_depth_sheets(self, sheet_names: list) -> set:
        """시트명으로부터 깊이 정보 자동 감지"""
        
        detected_depths = set()
        
        # 패턴 1: "10cm", "20cm", "30cm" 형태
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if 'cm' in sheet_lower:
                # 숫자 + cm 패턴 추출
                import re
                match = re.search(r'(\d+)cm', sheet_lower)
                if match:
                    depth_num = match.group(1)
                    detected_depths.add(f"{depth_num}cm")
        
        # 패턴 2: 순수 숫자 시트명 ("10", "20", "30")
        for sheet in sheet_names:
            if sheet.isdigit():
                detected_depths.add(f"{sheet}cm")
        
        # 패턴 3: "depth_10", "layer_30" 등
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if any(keyword in sheet_lower for keyword in ['depth', 'layer', 'level']):
                import re
                match = re.search(r'(\d+)', sheet)
                if match:
                    depth_num = match.group(1)
                    detected_depths.add(f"{depth_num}cm")
        
        return detected_depths
    
    def detect_station_columns(self, df: pd.DataFrame) -> dict:
        """관측소 컬럼 자동 감지"""
        
        # Date 관련 컬럼 제외
        exclude_patterns = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        potential_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            if not any(pattern in col_lower for pattern in exclude_patterns):
                # 수치 데이터인지 확인
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    valid_ratio = numeric_data.notna().sum() / len(df)
                    
                    if valid_ratio > 0.5:  # 50% 이상이 유효한 수치
                        # 토양수분 범위인지 확인 (0.01 ~ 1.0)
                        valid_values = numeric_data.dropna()
                        if len(valid_values) > 0:
                            if (valid_values.min() >= 0.001 and valid_values.max() <= 1.0):
                                potential_cols.append(col)
                except:
                    continue
        
        print(f"🔍 감지된 관측소 컬럼: {potential_cols}")
        
        # 컬럼명을 관측소명으로 매핑 (기본적으로 컬럼명 그대로 사용)
        station_mapping = {}
        for col in potential_cols:
            # 컬럼명 정리 (공백, 특수문자 제거)
            clean_name = str(col).strip().replace(' ', '_')
            station_mapping[col] = clean_name
        
        return station_mapping
    
    def load_soil_moisture_data(self, file_path: Path, depth: str) -> pd.DataFrame:
        """토양수분 데이터 로드 및 처리"""
        
        try:
            # 시트명 결정
            xl_file = pd.ExcelFile(file_path)
            sheet_name = depth
            
            # 정확한 시트명이 없으면 유사한 시트 찾기
            if depth not in xl_file.sheet_names:
                depth_num = depth.replace('cm', '')
                for sheet in xl_file.sheet_names:
                    if depth_num in sheet or depth in sheet:
                        sheet_name = sheet
                        break
                else:
                    print(f"⚠️ {depth}에 해당하는 시트를 찾을 수 없습니다.")
                    return pd.DataFrame()
            
            # 데이터 로드
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"📊 원본 데이터 크기: {df.shape}")
            
            # Date 컬럼 처리
            date_cols = [col for col in df.columns if 'date' in str(col).lower()]
            date_index = None
            
            if date_cols:
                try:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    df.set_index(date_cols[0], inplace=True)
                    date_index = df.index
                    print(f"📅 날짜 범위: {date_index.min().strftime('%Y-%m-%d')} ~ {date_index.max().strftime('%Y-%m-%d')}")
                except:
                    print("⚠️ 날짜 컬럼 처리 실패, 월별 필터링 건너뜀")
                    date_index = None
            
            # 겨울철 데이터 제외 (11, 12, 1, 2, 3월)
            if date_index is not None:
                winter_months = [11, 12, 1, 2, 3]
                original_length = len(df)
                
                # 겨울철이 아닌 데이터만 선택
                growing_season_mask = ~date_index.month.isin(winter_months)
                df = df[growing_season_mask]
                
                filtered_length = len(df)
                excluded_count = original_length - filtered_length
                
                print(f"❄️ 겨울철 데이터 제외: {excluded_count:,}개 제거 ({excluded_count/original_length*100:.1f}%)")
                print(f"🌱 생육기간 데이터: {filtered_length:,}개 ({filtered_length/original_length*100:.1f}%)")
                
                if filtered_length > 0:
                    remaining_months = sorted(df.index.month.unique())
                    print(f"📅 포함된 월: {remaining_months}")
            
            # 관측소 컬럼 자동 감지
            station_mapping = self.detect_station_columns(df)
            
            if not station_mapping:
                print(f"❌ {depth} 데이터에서 유효한 관측소 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()
            
            # Long format으로 변환
            data_list = []
            
            for col_name, station_name in station_mapping.items():
                values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                
                # 현실적인 토양수분 범위 필터링
                values = values[(values >= 0.01) & (values <= 0.8)]
                
                if len(values) > 20:  # 최소 데이터 요구사항
                    for value in values:
                        data_list.append({
                            'Station': station_name,
                            'VWC': value
                        })
                    
                    print(f"✅ {station_name}: {len(values)} 데이터 포인트")
            
            if not data_list:
                print(f"⚠️ {depth} 데이터에서 유효한 토양수분 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()
            
            result_df = pd.DataFrame(data_list)
            
            # Average 분포 추가 (모든 관측소 데이터의 합집합)
            if len(result_df) > 100:  # 충분한 데이터가 있을 때만
                all_values = result_df['VWC'].values
                
                average_df = pd.DataFrame({
                    'Station': ['Average'] * len(all_values),
                    'VWC': all_values
                })
                
                result_df = pd.concat([result_df, average_df], ignore_index=True)
                print(f"📊 전체 평균 분포 추가: {len(all_values)} 포인트")
            
            print(f"✅ {depth} 최종 데이터: {len(result_df)} 포인트, {result_df['Station'].nunique()} 관측소")
            
            return result_df
            
        except Exception as e:
            print(f"❌ {depth} 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def create_ridgeline_plot(self, df: pd.DataFrame, depth: str, output_dir: Path) -> Path:
        """Ridgeline plot 생성"""
        
        if df.empty:
            print(f"⚠️ {depth} 데이터가 비어있어 플롯을 생성할 수 없습니다.")
            return None
        
        # 관측소 순서 (평균 기준 정렬, Average는 맨 아래로)
        station_means = df.groupby('Station')['VWC'].mean().sort_values(ascending=False)
        station_order = station_means.index.tolist()
        
        # Average를 맨 아래로
        if 'Average' in station_order:
            station_order.remove('Average')
            station_order.append('Average')
        
        n_stations = len(station_order)
        
        # 색상 설정
        colors = sns.color_palette("viridis", n_stations)
        
        # 그래프 설정 (1:1.5 비율)
        base_width = max(8, min(12, n_stations * 0.8))  # 기본 너비
        figsize = (base_width, base_width * 1.5)  # 1:1.5 비율 유지
        
        fig, axes = plt.subplots(n_stations, 1, figsize=figsize, sharex=True)
        
        if n_stations == 1:
            axes = [axes]
        
        # 전체 데이터 범위
        x_min, x_max = df['VWC'].quantile([0.005, 0.995])
        x_range = np.linspace(x_min, x_max, 100)
        
        # 각 관측소별 ridgeline 그리기
        for i, station in enumerate(station_order):
            ax = axes[i]
            
            # 해당 관측소 데이터
            station_data = df[df['Station'] == station]['VWC'].values
            
            # 커널 밀도 추정
            try:
                kde = stats.gaussian_kde(station_data)
                kde.set_bandwidth(kde.factor * 0.8)  # 약간 더 부드럽게
                kde_values = kde(x_range)
            except:
                # KDE 실패시 히스토그램 사용
                hist, bins = np.histogram(station_data, bins=30, density=True, 
                                        range=(x_min, x_max))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                kde_values = np.interp(x_range, bin_centers, hist)
            
            # 색상 선택
            if station == 'Average':
                color = '#e74c3c'  # 빨간색으로 강조
                alpha = 0.8
            else:
                color = colors[i]
                alpha = 0.7
            
            # Ridgeline 그리기
            ax.fill_between(x_range, 0, kde_values, 
                           color=color, alpha=alpha, linewidth=0)
            ax.plot(x_range, kde_values, color=color, linewidth=1.5)
            
            # 평균선
            mean_val = np.mean(station_data)
            max_density = np.max(kde_values)
            ax.vlines(mean_val, 0, max_density, 
                     colors='white', linewidth=2, alpha=0.9)
            
            # 축 설정
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, None)
            ax.set_yticks([])
            
            # 관측소명 표시
            ax.text(0.01, 0.8, station, transform=ax.transAxes,
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.8, edgecolor='none'))
            
            # 통계 정보
            stats_text = f"μ={mean_val:.3f}\nσ={np.std(station_data):.3f}\nn={len(station_data):,}"
            ax.text(0.99, 0.8, stats_text, transform=ax.transAxes,
                   fontsize=9, ha='right',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.7, edgecolor='none'))
            
            # 축 스타일링
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            if i < n_stations - 1:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
            else:
                ax.spines['bottom'].set_color('#cccccc')
                ax.tick_params(colors='#666666')
        
        # 제목 및 라벨
        title = f'{self.station_id} - {depth} Soil Moisture Ridge Plot (Growing Season)'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        axes[-1].set_xlabel('Volumetric Water Content (m³/m³)',
                           fontsize=12, color='#333333')
        
        # 레이아웃 조정
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0)
        
        # 배경색 설정
        fig.patch.set_facecolor('white')
        
        # 저장
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.station_id}_ridgeline_{depth}.png"
        
        fig.savefig(output_file, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"✅ 저장됨: {output_file}")
        
        # 메모리 정리
        plt.close(fig)
        
        return output_file
    
    def print_summary_stats(self, df: pd.DataFrame, depth: str):
        """요약 통계 출력"""
        
        print(f"\n{'='*60}")
        print(f"📊 {self.station_id} - {depth} SUMMARY STATISTICS (Growing Season)")
        print('='*60)
        print("❄️ 겨울철 데이터 제외됨 (11, 12, 1, 2, 3월)")
        print('-'*60)
        
        summary = df.groupby('Station')['VWC'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        summary['range'] = (summary['max'] - summary['min']).round(3)
        summary['cv%'] = (summary['std'] / summary['mean'] * 100).round(1)
        
        # 정렬
        summary = summary.sort_values('mean', ascending=False)
        
        print(summary)
        
        print(f"\n📈 Overall Statistics (Growing Season Only):")
        print(f"   Total data points: {len(df):,}")
        print(f"   Stations: {df['Station'].nunique() - (1 if 'Average' in df['Station'].values else 0)} + Average")
        print(f"   Overall mean: {df[df['Station'] != 'Average']['VWC'].mean():.3f} m³/m³")
        print(f"   Overall std: {df[df['Station'] != 'Average']['VWC'].std():.3f}")
        print(f"   Data range: {df['VWC'].min():.3f} - {df['VWC'].max():.3f} m³/m³")
        print(f"   Coefficient of variation: {(df[df['Station'] != 'Average']['VWC'].std()/df[df['Station'] != 'Average']['VWC'].mean()*100):.1f}%")
    
    def run(self, depths: list = None, output_dir: str = None):
        """메인 실행 함수"""
        
        if output_dir is None:
            # 기본 출력 경로: data/output/{station_id}/visualization/
            output_dir = self.output_dir / "visualization"
        else:
            output_dir = Path(output_dir)
        
        print(f"\n🚀 {self.station_id} 관측소 Ridgeline Plot 생성 시작")
        if depths:
            print(f"📋 지정된 깊이: {depths}")
        else:
            print(f"📋 자동 깊이 감지 모드")
        print(f"📁 출력 디렉토리: {output_dir}")
        
        # FDR 파일 탐색 (깊이 자동 감지 포함)
        fdr_files = self.find_fdr_files(depths)
        
        if not fdr_files:
            print("❌ 처리할 FDR 파일을 찾을 수 없습니다.")
            print("\n💡 힌트:")
            print("   1. 전처리가 완료되었는지 확인하세요")
            print("   2. FDR 파일이 다음 위치에 있는지 확인하세요:")
            print(f"      - {self.output_dir}/preprocessed/")
            print(f"      - {self.input_dir}/")
            return False
        
        # 실제 처리할 깊이 목록
        actual_depths = list(fdr_files.keys())
        print(f"📊 처리 예정 깊이: {actual_depths}")
        
        success_count = 0
        
        # 각 깊이별 처리
        for depth in actual_depths:
            print(f"\n🔍 {depth} 처리 중...")
            
            # 데이터 로드
            df = self.load_soil_moisture_data(fdr_files[depth], depth)
            
            if df.empty:
                print(f"⚠️ {depth} 데이터가 없어 건너뜁니다.")
                continue
            
            # 요약 통계
            self.print_summary_stats(df, depth)
            
            # Ridgeline plot 생성
            output_file = self.create_ridgeline_plot(df, depth, output_dir)
            
            if output_file:
                success_count += 1
                self.processed_data[depth] = df
        
        print(f"\n🎉 완료! {success_count}/{len(actual_depths)} 개 플롯 생성 성공")
        
        if success_count > 0:
            print(f"📁 결과 저장 위치: {output_dir}")
            print(f"📊 생성된 파일들:")
            for depth in actual_depths:
                if depth in self.processed_data:
                    filename = f"{self.station_id}_ridgeline_{depth}.png"
                    print(f"   ✅ {filename}")
            return True
        else:
            return False

def main():
    """명령줄 인터페이스"""
    
    parser = argparse.ArgumentParser(
        description="범용 Ridgeline Plot 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python script/run_ridgeline.py --station HC                    # 자동 깊이 감지
  python script/run_ridgeline.py --station PC --depths 10cm 30cm 60cm
  python script/run_ridgeline.py --station HC --base-dir /path/to/data
  python script/run_ridgeline.py --station MyStation --output-dir ./custom_results
        """
    )
    
    parser.add_argument('--station', '-s', required=True,
                       help='관측소 ID (예: HC, PC)')
    
    parser.add_argument('--depths', '-d', nargs='+',
                       help='처리할 깊이들 (예: 10cm 30cm 60cm). 미지정시 자동 감지')
    
    parser.add_argument('--base-dir', '-b', default='data',
                       help='데이터 기본 디렉토리 (기본값: data)')
    
    parser.add_argument('--output-dir', '-o',
                       help='출력 디렉토리 (기본값: data/output/{station_id}/visualization/)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🔧 설정:")
        print(f"   관측소: {args.station}")
        print(f"   깊이: {args.depths if args.depths else '자동 감지'}")
        print(f"   기본 디렉토리: {args.base_dir}")
        print(f"   출력 디렉토리: {args.output_dir}")
        print(f"   ❄️ 겨울철 데이터 제외: 11, 12, 1, 2, 3월")
        print(f"   📊 Average 분포 포함: 모든 관측소 데이터의 합집합")
    
    try:
        # 생성기 초기화
        generator = RidgelineGenerator(args.station, args.base_dir)
        
        # 실행
        success = generator.run(args.depths, args.output_dir)
        
        if success:
            print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
            sys.exit(0)
        else:
            print("\n❌ 일부 또는 모든 작업이 실패했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()