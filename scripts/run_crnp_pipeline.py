# scripts/run_crnp_pipeline.py

"""
CRNP 데이터 처리 통합 파이프라인 - 간단한 시각화 버전

전처리 → 캘리브레이션 → 토양수분 계산 → 검증 → 간단한 시각화를 
순차적으로 실행하는 통합 스크립트

사용법:
    python scripts/run_crnp_pipeline.py --station HC --all
    python scripts/run_crnp_pipeline.py --station HC --steps preprocessing calibration
    python scripts/run_crnp_pipeline.py --station HC --cal-period 2024-08-17 2024-08-25
    python scripts/run_crnp_pipeline.py --station HC --status
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.calibration.calibration_manager import CalibrationManager
from src.calculation.soil_moisture_manager import SoilMoistureManager
from src.validation.validation_manager import ValidationManager
from src.visualization.simple_plotter import create_simple_visualization  # 간단한 시각화 사용
from src.core.logger import setup_logger


class CRNPPipelineRunner:
    """CRNP 전체 파이프라인을 실행하는 클래스 - 순서 개선 버전 (검증 → 시각화)"""
    
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.logger = setup_logger(f"CRNPPipeline_{station_id}")
        
        # 각 단계별 매니저 초기화
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.calibration_manager = CalibrationManager(station_id)
        self.sm_manager = SoilMoistureManager(station_id)
        self.validation_manager = ValidationManager(station_id)
        
        # 파이프라인 결과 저장
        self.results = {}
        
    def run_full_pipeline(self, calibration_period: Optional[tuple] = None,
                         calculation_period: Optional[tuple] = None,
                         skip_validation: bool = False,
                         force_recalculation: bool = False) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        
        print(f"🚀 CRNP 전체 파이프라인 시작 - {self.station_id} 관측소")
        print("=" * 70)
        
        pipeline_start_time = datetime.now()
        
        try:
            # 1. 전처리
            print("📊 1단계: 데이터 전처리")
            preprocessing_result = self._run_preprocessing()
            self.results['preprocessing'] = preprocessing_result
            
            # 2. 캘리브레이션
            print("\n🔬 2단계: 캘리브레이션")
            calibration_result = self._run_calibration(calibration_period, force_recalculation)
            self.results['calibration'] = calibration_result
            
            # 3. 토양수분 계산
            print("\n💧 3단계: 토양수분 계산")
            sm_result = self._run_soil_moisture_calculation(calculation_period, force_recalculation)
            self.results['soil_moisture'] = sm_result
            
            # 4. 검증 (선택사항)
            if not skip_validation:
                print("\n✅ 4단계: 검증")
                validation_result = self._run_validation()
                self.results['validation'] = validation_result
            else:
                print("\n⏭️  4단계: 검증 (건너뜀)")
                self.results['validation'] = {'status': 'skipped'}
            
            # 5. 간단한 시각화 (검증 후 실행으로 모든 데이터 준비됨)
            print("\n🎨 5단계: 시각화 생성 (간단 버전)")
            viz_result = self._run_simple_visualization()
            self.results['visualization'] = viz_result
                
            # 6. 최종 결과 정리
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            final_result = {
                'station_id': self.station_id,
                'pipeline_completed': True,
                'completion_time': datetime.now().isoformat(),
                'total_duration_seconds': pipeline_duration,
                'steps_completed': len([r for r in self.results.values() if r.get('status') != 'failed']),
                'results': self.results
            }
            
            # 결과 저장
            self._save_pipeline_results(final_result)
            
            # 최종 요약 출력
            self._print_final_summary(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            print(f"\n❌ 파이프라인 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 실패 결과 저장
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            failed_result = {
                'station_id': self.station_id,
                'pipeline_completed': False,
                'failure_time': datetime.now().isoformat(),
                'total_duration_seconds': pipeline_duration,
                'error': str(e),
                'results': self.results
            }
            
            self._save_pipeline_results(failed_result)
            raise
            
    def run_specific_steps(self, steps: List[str], **kwargs) -> Dict[str, Any]:
        """특정 단계들만 실행"""
        
        print(f"🎯 CRNP 파이프라인 - {self.station_id} 관측소 (선택 단계)")
        print(f"실행 단계: {', '.join(steps)}")
        print("=" * 70)
        
        step_start_time = datetime.now()
        
        try:
            for step in steps:
                if step == 'preprocessing':
                    print("📊 전처리 실행")
                    self.results['preprocessing'] = self._run_preprocessing()
                    
                elif step == 'calibration':
                    print("🔬 캘리브레이션 실행")
                    cal_period = kwargs.get('calibration_period')
                    force = kwargs.get('force_recalculation', False)
                    self.results['calibration'] = self._run_calibration(cal_period, force)
                    
                elif step == 'soil_moisture':
                    print("💧 토양수분 계산 실행")
                    sm_period = kwargs.get('calculation_period')
                    force = kwargs.get('force_recalculation', False)
                    self.results['soil_moisture'] = self._run_soil_moisture_calculation(sm_period, force)
                    
                elif step == 'validation':
                    print("✅ 검증 실행")
                    self.results['validation'] = self._run_validation()
                    
                elif step == 'visualization':
                    print("🎨 간단한 시각화 생성")
                    self.results['visualization'] = self._run_simple_visualization()
                    
                else:
                    print(f"⚠️  알 수 없는 단계: {step}")
                    
            step_duration = (datetime.now() - step_start_time).total_seconds()
            
            result = {
                'station_id': self.station_id,
                'steps_executed': steps,
                'completion_time': datetime.now().isoformat(),
                'duration_seconds': step_duration,
                'results': self.results
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            print(f"\n❌ 단계 실행 실패: {e}")
            raise
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 확인"""
        
        status = {
            'station_id': self.station_id,
            'steps': {}
        }
        
        # 각 단계별 상태 확인
        try:
            # 전처리 상태
            preprocessing_status = self._check_preprocessing_status()
            status['steps']['preprocessing'] = preprocessing_status
            
            # 캘리브레이션 상태
            cal_status = self.calibration_manager.get_calibration_status()
            status['steps']['calibration'] = cal_status
            
            # 토양수분 계산 상태
            sm_status = self.sm_manager.get_calculation_status()
            status['steps']['soil_moisture'] = sm_status
            
            # 검증 상태
            val_status = self.validation_manager.get_validation_status()
            status['steps']['validation'] = val_status
            
            # 시각화 상태 (간단 버전)
            viz_status = self._check_simple_visualization_status()
            status['steps']['visualization'] = viz_status
            
            # 전체 완성도 계산
            completed_steps = sum(1 for step_status in status['steps'].values() 
                                if step_status.get('calibration_available') or 
                                   step_status.get('calculation_available') or
                                   step_status.get('validation_available') or
                                   step_status.get('plots_available') or
                                   step_status.get('preprocessing_available'))
            
            status['completion_percentage'] = (completed_steps / 5) * 100
            status['ready_for_next_step'] = self._determine_next_step(status['steps'])
            
        except Exception as e:
            self.logger.warning(f"Error checking pipeline status: {e}")
            
        return status
        
    def _run_preprocessing(self) -> Dict[str, Any]:
        """전처리 실행"""
        
        try:
            result = self.preprocessing_pipeline.run_station_preprocessing(self.station_id)
            
            if result.get('overall_status') == 'success':
                print("   ✅ 전처리 완료")
                return {'status': 'success', 'result': result}
            else:
                print("   ⚠️  전처리 부분 완료")
                return {'status': 'partial', 'result': result}
                
        except Exception as e:
            print(f"   ❌ 전처리 실패: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _run_calibration(self, calibration_period: Optional[tuple], force: bool) -> Dict[str, Any]:
        """캘리브레이션 실행"""
        
        try:
            cal_start, cal_end = calibration_period if calibration_period else (None, None)
            
            result = self.calibration_manager.run_calibration(
                calibration_start=cal_start,
                calibration_end=cal_end,
                force_recalibration=force
            )
            
            print(f"   ✅ 캘리브레이션 완료 (N0={result.get('N0_rdt', 0):.2f})")
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            print(f"   ❌ 캘리브레이션 실패: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _run_soil_moisture_calculation(self, calculation_period: Optional[tuple], force: bool) -> Dict[str, Any]:
        """토양수분 계산 실행"""
        
        try:
            calc_start, calc_end = calculation_period if calculation_period else (None, None)
            
            result = self.sm_manager.calculate_soil_moisture(
                calculation_start=calc_start,
                calculation_end=calc_end,
                force_recalculation=force
            )
            
            data_summary = result.get('data_summary', {})
            total_days = data_summary.get('total_days', 0)
            print(f"   ✅ 토양수분 계산 완료 ({total_days}일)")
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            print(f"   ❌ 토양수분 계산 실패: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _run_simple_visualization(self) -> Dict[str, Any]:
        """간단한 시각화 실행"""
        
        try:
            output_dir = f"data/output/{self.station_id}/visualization"
            plot_files = create_simple_visualization(self.station_id, output_dir)
            
            print(f"   ✅ 간단한 시각화 완료 ({len(plot_files)}개 플롯)")
            print("   📊 생성된 그래프:")
            
            plot_descriptions = {
                'neutron_comparison': '   • 중성자 카운트 비교',
                'correction_factors': '   • 보정계수 시계열',
                'vwc_timeseries': '   • VWC 시계열',
                'sm_timeseries': '   • 토양수분 비교',
                'sm_scatter': '   • 토양수분 산점도'
            }
            
            for plot_type in plot_files.keys():
                description = plot_descriptions.get(plot_type, f"   • {plot_type}")
                print(description)
            
            return {
                'status': 'success', 
                'result': {
                    'plot_files': plot_files,
                    'total_plots': len(plot_files),
                    'output_dir': output_dir
                }
            }
            
        except Exception as e:
            print(f"   ❌ 시각화 실패: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _run_validation(self) -> Dict[str, Any]:
        """검증 실행"""
        
        try:
            result = self.validation_manager.run_validation()
            
            overall_metrics = result.get('overall_metrics', {})
            r2 = overall_metrics.get('R2', 0)
            rmse = overall_metrics.get('RMSE', 0)
            print(f"   ✅ 검증 완료 (R²={r2:.3f}, RMSE={rmse:.3f})")
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            print(f"   ❌ 검증 실패: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _check_preprocessing_status(self) -> Dict[str, Any]:
        """전처리 상태 확인"""
        
        output_dir = Path(f"data/output/{self.station_id}/preprocessed")
        
        fdr_file = output_dir / f"{self.station_id}_FDR_input.xlsx"
        crnp_file = output_dir / f"{self.station_id}_CRNP_input.xlsx"
        
        return {
            'preprocessing_available': fdr_file.exists() and crnp_file.exists(),
            'fdr_file_exists': fdr_file.exists(),
            'crnp_file_exists': crnp_file.exists(),
            'output_directory': str(output_dir)
        }
        
    def _check_simple_visualization_status(self) -> Dict[str, Any]:
        """간단한 시각화 상태 확인"""
        
        viz_dir = Path(f"data/output/{self.station_id}/visualization")
        
        expected_plots = [
            f"{self.station_id}_neutron_comparison.png",
            f"{self.station_id}_correction_factors.png", 
            f"{self.station_id}_vwc_timeseries.png",
            f"{self.station_id}_soil_moisture_comparison.png",
            f"{self.station_id}_soil_moisture_scatter.png"
        ]
        
        existing_plots = []
        for plot_file in expected_plots:
            if (viz_dir / plot_file).exists():
                existing_plots.append(plot_file)
        
        return {
            'plots_available': len(existing_plots) > 0,
            'total_plots': len(existing_plots),
            'expected_plots': len(expected_plots),
            'plot_files': existing_plots,
            'output_directory': str(viz_dir)
        }
        
    def _determine_next_step(self, steps_status: Dict[str, Any]) -> str:
        """다음 실행 가능한 단계 결정"""
        
        if not steps_status.get('preprocessing', {}).get('preprocessing_available'):
            return 'preprocessing'
        elif not steps_status.get('calibration', {}).get('calibration_available'):
            return 'calibration'
        elif not steps_status.get('soil_moisture', {}).get('calculation_available'):
            return 'soil_moisture'
        elif not steps_status.get('validation', {}).get('validation_available'):
            return 'validation'
        elif not steps_status.get('visualization', {}).get('plots_available'):
            return 'visualization'
        else:
            return 'complete'
            
    def _save_pipeline_results(self, result: Dict[str, Any]) -> None:
        """파이프라인 결과 저장"""
        
        output_dir = Path(f"data/output/{self.station_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f"{self.station_id}_pipeline_result.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            self.logger.log_file_operation("save", str(result_file), "success")
            
        except Exception as e:
            self.logger.warning(f"Could not save pipeline results: {e}")
            
    def _print_final_summary(self, result: Dict[str, Any]) -> None:
        """최종 요약 출력"""
        
        print("\n" + "=" * 70)
        print("🎉 CRNP 파이프라인 완료!")
        print("=" * 70)
        
        duration = result.get('total_duration_seconds', 0)
        print(f"총 소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")
        print(f"완료된 단계: {result.get('steps_completed', 0)}/5")
        
        # 각 단계별 결과 요약
        results = result.get('results', {})
        
        print(f"\n📊 단계별 결과:")
        step_names = {
            'preprocessing': '전처리',
            'calibration': '캘리브레이션', 
            'soil_moisture': '토양수분 계산',
            'validation': '검증',
            'visualization': '시각화'
        }
        
        for step_key, step_name in step_names.items():
            if step_key in results:
                status = results[step_key].get('status', 'unknown')
                status_icon = {'success': '✅', 'partial': '⚠️', 'failed': '❌', 'skipped': '⏭️'}.get(status, '❓')
                print(f"  {step_name}: {status_icon}")
                
        # 주요 결과 출력
        if 'calibration' in results and results['calibration'].get('status') == 'success':
            cal_result = results['calibration']['result']
            N0 = cal_result.get('N0_rdt', 0)
            metrics = cal_result.get('performance_metrics', {})
            r2 = metrics.get('R2', 0)
            print(f"\n🔬 캘리브레이션 결과: N0={N0:.2f}, R²={r2:.3f}")
            
        if 'validation' in results and results['validation'].get('status') == 'success':
            val_result = results['validation']['result']
            val_metrics = val_result.get('overall_metrics', {})
            val_r2 = val_metrics.get('R2', 0)
            val_rmse = val_metrics.get('RMSE', 0)
            print(f"✅ 검증 결과: R²={val_r2:.3f}, RMSE={val_rmse:.3f}")
            
        if 'visualization' in results and results['visualization'].get('status') == 'success':
            viz_result = results['visualization']['result']
            total_plots = viz_result.get('total_plots', 0)
            print(f"🎨 시각화: {total_plots}개 간단한 그래프 생성")
                
        print(f"\n📁 결과 위치: data/output/{self.station_id}/")
        print("=" * 70)


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="CRNP 통합 파이프라인 실행 (간단한 시각화)")
    
    # 필수 인자
    parser.add_argument("--station", "-s", required=True, help="관측소 ID (예: HC, PC)")
    
    # 실행 모드
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--all", action="store_true", help="전체 파이프라인 실행")
    mode_group.add_argument("--steps", nargs="+", 
                           choices=['preprocessing', 'calibration', 'soil_moisture', 'validation', 'visualization'],
                           help="특정 단계들만 실행")
    mode_group.add_argument("--status", action="store_true", help="파이프라인 상태만 확인")
    
    # 옵션
    parser.add_argument("--cal-period", nargs=2, metavar=('START', 'END'),
                       help="캘리브레이션 기간 (YYYY-MM-DD YYYY-MM-DD)")
    parser.add_argument("--calc-period", nargs=2, metavar=('START', 'END'),
                       help="계산 기간 (YYYY-MM-DD YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true", help="기존 결과 무시하고 재실행")
    parser.add_argument("--skip-validation", action="store_true", help="검증 단계 건너뛰기")
    
    args = parser.parse_args()
    
    # 기본값 설정
    if not args.all and not args.steps and not args.status:
        args.all = True
        
    try:
        # 파이프라인 러너 초기화
        runner = CRNPPipelineRunner(args.station)
        
        if args.status:
            # 상태 확인만
            status = runner.get_pipeline_status()
            
            print(f"🔍 {args.station} 관측소 파이프라인 상태")
            print("=" * 50)
            print(f"전체 완성도: {status.get('completion_percentage', 0):.1f}%")
            print(f"다음 단계: {status.get('ready_for_next_step', 'unknown')}")
            
            steps_status = status.get('steps', {})
            step_names = {
                'preprocessing': '전처리',
                'calibration': '캘리브레이션',
                'soil_moisture': '토양수분 계산', 
                'validation': '검증',
                'visualization': '시각화'
            }
            
            print(f"\n📊 단계별 상태:")
            for step_key, step_name in step_names.items():
                if step_key in steps_status:
                    step_status = steps_status[step_key]
                    
                    # 단계별로 다른 키 확인
                    available = (step_status.get('preprocessing_available') or
                               step_status.get('calibration_available') or  
                               step_status.get('calculation_available') or
                               step_status.get('plots_available') or
                               step_status.get('validation_available'))
                    
                    status_icon = "✅" if available else "❌"
                    print(f"  {step_name}: {status_icon}")
                    
        elif args.all:
            # 전체 파이프라인 실행
            calibration_period = tuple(args.cal_period) if args.cal_period else None
            calculation_period = tuple(args.calc_period) if args.calc_period else None
            
            result = runner.run_full_pipeline(
                calibration_period=calibration_period,
                calculation_period=calculation_period,
                skip_validation=args.skip_validation,
                force_recalculation=args.force
            )
            
        elif args.steps:
            # 특정 단계들만 실행
            kwargs = {}
            if args.cal_period:
                kwargs['calibration_period'] = tuple(args.cal_period)
            if args.calc_period:
                kwargs['calculation_period'] = tuple(args.calc_period)
            if args.force:
                kwargs['force_recalculation'] = True
                
            result = runner.run_specific_steps(args.steps, **kwargs)
            
            print(f"\n✅ 선택된 단계 완료: {', '.join(args.steps)}")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ 파이프라인 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())