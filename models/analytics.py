"""
LifeLine Pro - Analytics Module
Advanced analytics and trend analysis for medical monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
import joblib
from pathlib import Path
import json
import logging

from .data_models import AnalysisResult, AlertSeverity, PatientProfile

logger = logging.getLogger(__name__)

class HealthAnalytics:
    """Advanced health analytics system for trend analysis and predictions"""
    
    def __init__(self, data_dir: str = "snapshots"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize analytics components
        self._init_analytics()
        
        # Load or create analytics models
        self._load_models()
    
    def _init_analytics(self):
        """Initialize analytics components and thresholds"""
        # Use scaler if available, otherwise use a simple placeholder
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.analysis_window = {
            'short_term': timedelta(minutes=15),
            'medium_term': timedelta(hours=1),
            'long_term': timedelta(hours=24)
        }
        
        # Define trend thresholds
        self.trend_thresholds = {
            'critical_change': 0.25,  # 25% change
            'significant_change': 0.15,  # 15% change
            'moderate_change': 0.10,  # 10% change
        }
    
    def _load_models(self):
        """Load or initialize machine learning models"""
        model_path = self.data_dir / 'anomaly_detector.joblib'
        
        if not SKLEARN_AVAILABLE:
            self.anomaly_detector = None
            logger.warning("scikit-learn not available: anomaly detection disabled")
            return

        if model_path.exists():
            try:
                self.anomaly_detector = joblib.load(model_path)
                logger.info("Loaded existing anomaly detection model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._init_models()
        else:
            self._init_models()
    
    def _init_models(self):
        """Initialize new machine learning models"""
        if not SKLEARN_AVAILABLE:
            self.anomaly_detector = None
            return

        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        logger.info("Initialized new anomaly detection model")
    
    def analyze_trends(self, 
                      patient: PatientProfile,
                      recent_data: List[AnalysisResult],
                      timeframe: str = 'medium_term') -> Dict:
        """
        Analyze trends in patient monitoring data
        Returns trend analysis and predictions
        """
        if not recent_data:
            return {}
            
        # Convert to dataframe for analysis
        df = pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'risk_score': r.risk_score,
                'pain_score': r.pain_score,
                'has_alert': len(r.alerts) > 0,
                'alert_severity': max([a['severity'] for a in r.alerts]) if r.alerts else 'none'
            }
            for r in recent_data
        ])
        
        # Calculate trends
        trends = self._calculate_trends(df, timeframe)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df)
        
        # Generate predictions
        predictions = self._generate_predictions(df, patient)
        
        # Calculate risk trajectory
        risk_trajectory = self._analyze_risk_trajectory(df)
        
        # Generate insights
        insights = self._generate_trend_insights(
            trends, anomalies, predictions, risk_trajectory
        )
        
        return {
            'trends': trends,
            'anomalies': anomalies,
            'predictions': predictions,
            'risk_trajectory': risk_trajectory,
            'insights': insights
        }
    
    def _calculate_trends(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate trends in different metrics"""
        window = self.analysis_window[timeframe]
        recent_cutoff = datetime.now() - window
        
        df_window = df[df['timestamp'] > recent_cutoff]
        if len(df_window) < 2:
            return {}
            
        trends = {}
        
        # Risk score trend
        risk_start = df_window['risk_score'].iloc[0]
        risk_end = df_window['risk_score'].iloc[-1]
        risk_change = ((risk_end - risk_start) / risk_start) if risk_start > 0 else 0
        
        trends['risk_score'] = {
            'change': risk_change,
            'direction': 'increasing' if risk_change > 0 else 'decreasing',
            'significance': self._get_trend_significance(abs(risk_change))
        }
        
        # Pain score trend
        pain_scores = df_window['pain_score'].dropna()
        if len(pain_scores) >= 2:
            pain_change = (pain_scores.iloc[-1] - pain_scores.iloc[0]) / max(pain_scores.iloc[0], 1)
            trends['pain_score'] = {
                'change': pain_change,
                'direction': 'increasing' if pain_change > 0 else 'decreasing',
                'significance': self._get_trend_significance(abs(pain_change))
            }
        
        # Alert frequency
        alert_freq_start = df_window.head(len(df_window)//2)['has_alert'].mean()
        alert_freq_end = df_window.tail(len(df_window)//2)['has_alert'].mean()
        alert_change = alert_freq_end - alert_freq_start
        
        trends['alert_frequency'] = {
            'change': alert_change,
            'direction': 'increasing' if alert_change > 0 else 'decreasing',
            'significance': self._get_trend_significance(abs(alert_change))
        }
        
        return trends
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in monitoring data"""
        if len(df) < 10:  # Need minimum data points
            return []
        
        if not SKLEARN_AVAILABLE or self.anomaly_detector is None:
            logger.debug("Anomaly detection skipped because scikit-learn is not available or model is not initialized")
            return []
        try:
            # Prepare features
            features = df[['risk_score', 'pain_score']].fillna(0)
            
            # Fit and predict
            self.anomaly_detector.fit(features)
            anomaly_labels = self.anomaly_detector.predict(features)
            
            # Find anomalous points
            anomalies = []
            for idx, is_anomaly in enumerate(anomaly_labels == -1):
                if is_anomaly:
                    anomalies.append({
                        'timestamp': df['timestamp'].iloc[idx],
                        'risk_score': df['risk_score'].iloc[idx],
                        'pain_score': df['pain_score'].iloc[idx],
                        'type': 'Critical' if df['risk_score'].iloc[idx] > 80 else 'Significant'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def _generate_predictions(self, df: pd.DataFrame, patient: PatientProfile) -> Dict:
        """Generate short-term predictions"""
        if len(df) < 10:
            return {}
            
        try:
            # Simple trend-based predictions
            risk_trend = np.polyfit(range(len(df)), df['risk_score'], 1)[0]
            current_risk = df['risk_score'].iloc[-1]
            
            predictions = {
                'risk_score_15min': min(100, max(0, current_risk + risk_trend * 15)),
                'risk_score_30min': min(100, max(0, current_risk + risk_trend * 30)),
                'risk_score_1hr': min(100, max(0, current_risk + risk_trend * 60)),
            }
            
            # Predict alert likelihood
            recent_alerts = df.tail(20)['has_alert'].mean()
            alert_trend = recent_alerts > 0.3
            
            predictions['alert_likelihood'] = {
                'next_15min': 'high' if alert_trend and current_risk > 70 else 'medium' if alert_trend or current_risk > 60 else 'low',
                'next_hour': 'high' if risk_trend > 0 and current_risk > 60 else 'medium' if risk_trend > 0 or current_risk > 50 else 'low'
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {}
    
    def _analyze_risk_trajectory(self, df: pd.DataFrame) -> Dict:
        """Analyze patient's risk trajectory"""
        if len(df) < 5:
            return {}
            
        try:
            # Calculate risk velocity (rate of change)
            risk_velocity = np.diff(df['risk_score']).mean()
            
            # Calculate risk acceleration (change in velocity)
            risk_accel = np.diff(np.diff(df['risk_score'])).mean()
            
            # Determine trajectory
            if abs(risk_velocity) < 0.1:
                trajectory = 'stable'
            elif risk_velocity > 0:
                trajectory = 'deteriorating' if risk_accel >= 0 else 'stabilizing'
            else:
                trajectory = 'improving' if risk_accel <= 0 else 'uncertain'
            
            return {
                'trajectory': trajectory,
                'velocity': risk_velocity,
                'acceleration': risk_accel
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk trajectory: {e}")
            return {}
    
    def _generate_trend_insights(self, trends: Dict, anomalies: List,
                               predictions: Dict, trajectory: Dict) -> List[Dict]:
        """Generate actionable insights from trends"""
        insights = []
        
        # Risk trajectory insights
        if trajectory:
            if trajectory['trajectory'] == 'deteriorating':
                insights.append({
                    'priority': 'HIGH',
                    'category': 'Risk Trajectory',
                    'title': 'Deteriorating Condition',
                    'message': 'Patient condition is actively worsening',
                    'recommendation': 'Consider immediate intervention'
                })
            elif trajectory['trajectory'] == 'improving':
                insights.append({
                    'priority': 'LOW',
                    'category': 'Risk Trajectory',
                    'title': 'Improving Condition',
                    'message': 'Patient showing signs of improvement',
                    'recommendation': 'Continue current treatment plan'
                })
        
        # Trend-based insights
        for metric, trend in trends.items():
            if trend['significance'] == 'critical':
                insights.append({
                    'priority': 'CRITICAL',
                    'category': f'{metric.replace("_", " ").title()} Trend',
                    'title': f'Critical {trend["direction"].title()} Trend',
                    'message': f'Significant {trend["direction"]} trend in {metric.replace("_", " ")}',
                    'recommendation': 'Immediate assessment required'
                })
        
        # Anomaly insights
        if anomalies:
            critical_anomalies = [a for a in anomalies if a['type'] == 'Critical']
            if critical_anomalies:
                insights.append({
                    'priority': 'CRITICAL',
                    'category': 'Anomaly Detection',
                    'title': 'Critical Anomalies Detected',
                    'message': f'Detected {len(critical_anomalies)} critical anomalies',
                    'recommendation': 'Review patient status immediately'
                })
        
        # Prediction insights
        if predictions.get('alert_likelihood'):
            if predictions['alert_likelihood']['next_15min'] == 'high':
                insights.append({
                    'priority': 'HIGH',
                    'category': 'Predictions',
                    'title': 'High Alert Risk',
                    'message': 'High likelihood of alerts in next 15 minutes',
                    'recommendation': 'Prepare for potential intervention'
                })
        
        return sorted(insights, 
                     key=lambda x: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x['priority']))
    
    def _get_trend_significance(self, change: float) -> str:
        """Determine the significance of a trend change"""
        if change >= self.trend_thresholds['critical_change']:
            return 'critical'
        elif change >= self.trend_thresholds['significant_change']:
            return 'significant'
        elif change >= self.trend_thresholds['moderate_change']:
            return 'moderate'
        return 'minor'
    
    def save_analysis_result(self, result: AnalysisResult):
        """Save analysis result for historical tracking"""
        try:
            # Convert to dict for storage
            result_dict = {
                'timestamp': result.timestamp.isoformat(),
                'risk_score': result.risk_score,
                'pain_score': result.pain_score,
                'alerts': result.alerts,
                'insights': result.insights
            }
            
            # Save to CSV for time series data
            df = pd.DataFrame([result_dict])
            csv_path = self.data_dir / 'analysis_history.csv'
            df.to_csv(csv_path, mode='a', header=not csv_path.exists())
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
    
    def get_historical_data(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve historical analysis data"""
        try:
            csv_path = self.data_dir / 'analysis_history.csv'
            if not csv_path.exists():
                return pd.DataFrame()
                
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
                
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()