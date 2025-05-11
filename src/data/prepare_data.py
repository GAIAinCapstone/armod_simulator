import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 상위 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.database import get_database_connection, fetch_air_quality_data

def preprocess_data(df):
    """
    대기질 데이터를 전처리합니다.
    
    Args:
        df (DataFrame): 원본 데이터프레임
    
    Returns:
        DataFrame: 전처리된 데이터프레임
    """
    # measure_date를 datetime 형식으로 변환
    df['measure_date'] = pd.to_datetime(df['measure_date'])
    
    # 결측치 처리
    numeric_columns = [col for col in df.columns if 'measure' in col or 'stdr' in col]
    df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
    
    # 이상치 처리 (IQR 방식)
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def prepare_time_series_data(df, target_columns=['nox_measure', 'sox_measure', 'tsp_measure']):
    """
    시계열 예측을 위한 데이터를 준비합니다.
    
    Args:
        df (DataFrame): 전처리된 데이터프레임
        target_columns (list): 예측할 대상 컬럼 목록
    
    Returns:
        tuple: (X, y) 학습 데이터와 타겟 데이터
    """
    # 시계열 특성 생성
    df['hour'] = df['measure_date'].dt.hour
    df['day_of_week'] = df['measure_date'].dt.dayofweek
    df['month'] = df['measure_date'].dt.month
    
    # 시계열 데이터 준비
    X = df[['hour', 'day_of_week', 'month'] + [col for col in df.columns if 'stdr' in col]]
    y = df[target_columns]
    
    return X, y

def main():
    """
    메인 실행 함수
    """
    # 데이터베이스 연결
    connection = get_database_connection()
    if connection is None:
        print("데이터베이스 연결 실패")
        return
    
    try:
        # 데이터 가져오기
        df = fetch_air_quality_data(connection)
        if df is None:
            print("데이터 조회 실패")
            return
        
        # 데이터 전처리
        df_processed = preprocess_data(df)
        
        # 전처리된 데이터 저장
        df_processed.to_csv('data/processed/air_quality_processed.csv', index=False)
        print("데이터 전처리 완료")
        
        # 시계열 데이터 준비
        X, y = prepare_time_series_data(df_processed)
        
        # 시계열 데이터 저장
        X.to_csv('data/processed/features.csv', index=False)
        y.to_csv('data/processed/targets.csv', index=False)
        print("시계열 데이터 준비 완료")
        
    finally:
        connection.close()

if __name__ == "__main__":
    main() 