class CRNPPipeline:
    def __init__(self, station_config, processing_config):
        self.station_config = station_config
        self.processing_config = processing_config
        self.logger = setup_logger(station_config['station_info']['id'])
        
    def run_full_pipeline(self, start_date=None, end_date=None):
        """전체 파이프라인 실행"""
        try:
            # 1. 전처리
            self.logger.info("Starting data preprocessing...")
            preprocessed_data = self.preprocess_data()
            
            # 2. 캘리브레이션
            self.logger.info("Starting calibration...")
            calibration_results = self.calibrate(preprocessed_data)
            
            # 3. 토양수분 계산
            self.logger.info("Calculating soil moisture...")
            soil_moisture_data = self.calculate_soil_moisture(calibration_results)
            
            # 4. 검증
            self.logger.info("Running validation...")
            validation_results = self.validate_results(soil_moisture_data)
            
            # 5. 시각화
            self.logger.info("Generating visualizations...")
            self.generate_plots(soil_moisture_data, validation_results)
            
            return {
                'soil_moisture': soil_moisture_data,
                'validation': validation_results,
                'calibration': calibration_results
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise