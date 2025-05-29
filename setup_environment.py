# setup_environment.py

"""
CRNP 데이터 처리 시스템 환경 설정 스크립트
필요한 라이브러리를 자동으로 설치합니다.

사용법:
    python setup_environment.py
    python setup_environment.py --minimal  # 최소한의 라이브러리만 설치
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """파이썬 버전 확인"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        return False
    else:
        print(f"✅ Python 버전 확인: {sys.version}")
        return True

def install_package(package, description=""):
    """개별 패키지 설치"""
    try:
        print(f"📦 설치 중: {package} {description}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ 설치 완료: {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 설치 실패: {package} - {str(e)}")
        return False

def install_requirements(minimal=False):
    """필요한 라이브러리 설치"""
    
    # 필수 라이브러리 (최소 설치)
    essential_packages = [
        ("pandas>=1.5.0", "데이터 처리"),
        ("numpy>=1.20.0", "수치 계산"),
        ("openpyxl>=3.0.0", "Excel 파일 처리"),
        ("PyYAML>=6.0", "YAML 설정 파일"),
        ("matplotlib>=3.5.0", "기본 시각화"),
        ("scipy>=1.7.0", "과학 계산")
    ]
    
    # 추가 라이브러리 (전체 설치)
    additional_packages = [
        ("chardet>=4.0.0", "파일 인코딩 감지"),
        ("seaborn>=0.11.0", "고급 시각화"),
        ("scikit-learn>=1.0.0", "머신러닝/이상값 탐지"),
        ("psutil>=5.8.0", "시스템 모니터링"),
        ("joblib>=1.1.0", "병렬 처리")
    ]
    
    packages_to_install = essential_packages
    if not minimal:
        packages_to_install.extend(additional_packages)
    
    print(f"📚 설치할 패키지 수: {len(packages_to_install)}")
    print("=" * 50)
    
    success_count = 0
    failed_packages = []
    
    for package, description in packages_to_install:
        if install_package(package, f"({description})"):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    print(f"📊 설치 결과: {success_count}/{len(packages_to_install)} 성공")
    
    if failed_packages:
        print("❌ 설치 실패한 패키지:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\n💡 실패한 패키지는 수동으로 설치해주세요:")
        print("   pip install " + " ".join([pkg.split(">=")[0] for pkg in failed_packages]))
    else:
        print("✅ 모든 패키지가 성공적으로 설치되었습니다!")
    
    return len(failed_packages) == 0

def create_directories():
    """필요한 디렉토리 구조 생성"""
    directories = [
        "config/stations",
        "data/input/HC/fdr",
        "data/input/HC/crnp",
        "data/input/PC/fdr", 
        "data/input/PC/crnp",
        "data/output/HC",
        "data/output/PC",
        "logs",
        "src/core",
        "src/preprocessing",
        "src/utils",
        "scripts"
    ]
    
    print("\n📁 디렉토리 구조 생성 중...")
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print("✅ 디렉토리 구조 생성 완료")

def create_init_files():
    """__init__.py 파일 생성"""
    init_dirs = [
        "src",
        "src/core", 
        "src/preprocessing",
        "src/utils",
        "src/calibration",
        "src/validation",
        "src/visualization"
    ]
    
    print("\n📄 __init__.py 파일 생성 중...")
    
    for init_dir in init_dirs:
        init_file = Path(init_dir) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
            
    print("✅ __init__.py 파일 생성 완료")

def check_installation():
    """설치 확인"""
    print("\n🧪 설치 확인 중...")
    
    test_imports = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("yaml", None),
        ("matplotlib.pyplot", "plt"),
        ("scipy", None)
    ]
    
    failed_imports = []
    
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    # chardet는 선택적 확인
    try:
        import chardet
        print("✅ chardet (선택사항)")
    except ImportError:
        print("⚠️  chardet (선택사항) - 없어도 작동함")
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)}개 모듈 import 실패")
        return False
    else:
        print("\n✅ 모든 필수 모듈 import 성공!")
        return True

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRNP 환경 설정")
    parser.add_argument("--minimal", action="store_true", 
                       help="최소한의 라이브러리만 설치")
    parser.add_argument("--skip-install", action="store_true",
                       help="라이브러리 설치 건너뛰기")
    parser.add_argument("--check-only", action="store_true",
                       help="설치 확인만 실행")
    
    args = parser.parse_args()
    
    print("🚀 CRNP 데이터 처리 시스템 환경 설정")
    print("=" * 50)
    
    # 파이썬 버전 확인
    if not check_python_version():
        return False
    
    if args.check_only:
        return check_installation()
    
    # 디렉토리 구조 생성
    create_directories()
    
    # __init__.py 파일 생성
    create_init_files()
    
    # 라이브러리 설치
    if not args.skip_install:
        install_success = install_requirements(args.minimal)
        
        # 설치 확인
        if install_success:
            check_installation()
        
        print("\n🎯 다음 단계:")
        print("   1. YAML 설정 파일 생성: python scripts/create_yaml_configs.py")
        print("   2. 데이터 파일 배치 (data/input/{station_id}/)")
        print("   3. 전처리 실행: python scripts/run_preprocessing.py --station HC --check-only")
        
        return install_success
    else:
        print("⏭️  라이브러리 설치를 건너뛰었습니다.")
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        sys.exit(1)