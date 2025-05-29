import matplotlib.pyplot as plt
import seaborn as sns

class ModernPlotStyle:
    def __init__(self):
        # 모던한 컬러 팔레트
        self.colors = {
            'primary': '#2E86AB',      # 블루
            'secondary': '#A23B72',    # 마젠타  
            'accent': '#F18F01',       # 오렌지
            'success': '#C73E1D',      # 레드
            'neutral': '#6C757D',      # 그레이
            'background': '#F8F9FA'    # 라이트 그레이
        }
        
        self.setup_style()
    
    def setup_style(self):
        """모던한 플롯 스타일 설정"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 폰트 설정
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            
            # 선 스타일
            'lines.linewidth': 2,
            'lines.markersize': 6,
            
            # 축 스타일  
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            
            # 색상
            'axes.prop_cycle': plt.cycler('color', list(self.colors.values())),
        })