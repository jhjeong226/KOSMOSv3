class ValidationMetrics:
    @staticmethod
    def calculate_r2(observed, predicted):
        """결정계수 계산"""
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        obs_clean = observed[mask]
        pred_clean = predicted[mask]
        
        ss_res = np.sum((obs_clean - pred_clean) ** 2)
        ss_tot = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod  
    def calculate_rmse(observed, predicted):
        """평균제곱근오차 계산"""
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        return np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2))
    
    @staticmethod
    def calculate_mae(observed, predicted):
        """평균절대오차 계산"""
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        return np.mean(np.abs(observed[mask] - predicted[mask]))
    
    @staticmethod
    def calculate_nse(observed, predicted):
        """Nash-Sutcliffe 효율성 계산"""
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        obs_clean = observed[mask]
        pred_clean = predicted[mask]
        
        numerator = np.sum((obs_clean - pred_clean) ** 2)
        denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
        return 1 - (numerator / denominator)