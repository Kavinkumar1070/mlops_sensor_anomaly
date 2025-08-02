import pandas as pd
from exp_log.exception import CustomException
from exp_log.logger import logging
import sys
# --- Constants ---
FAULTY_VALUES = [500, 1000, 9999, 9998, 600]

# --- Wind speed bin mapping ---
def get_ws_sector_df():
    return pd.DataFrame({
    'wsfrom': [
        -0.250, 0.251, 0.751, 1.251, 1.751, 2.251, 2.751, 3.251, 3.751, 4.251,
        4.751, 5.251, 5.751, 6.251, 6.751, 7.251, 7.751, 8.251, 8.751, 9.251,
        9.751, 10.251, 10.751, 11.251, 11.751, 12.251, 12.751, 13.251, 13.751, 14.251,
        14.751, 15.251, 15.751, 16.251, 16.751, 17.251, 17.751, 18.251, 18.751, 19.251,
        19.751, 20.251, 20.751, 21.251, 21.751, 22.251, 22.751, 23.251, 23.751, 24.251,
        24.751, 25.251, 25.751, 26.251, 26.751, 27.251, 27.751, 28.251, 28.751, 29.251,
        29.751, 30.251, 30.751, 31.251, 31.751, 32.251, 32.751, 33.251, 33.751, 34.251,
        34.751, 35.251, 35.751, 36.251, 36.751, 37.251, 37.751, 38.251, 38.751, 39.251,
        39.751
    ],
    'wsto': [
        0.250, 0.750, 1.250, 1.750, 2.250, 2.750, 3.250, 3.750, 4.250, 4.750,
        5.250, 5.750, 6.250, 6.750, 7.250, 7.750, 8.250, 8.750, 9.250, 9.750,
        10.250, 10.750, 11.250, 11.750, 12.250, 12.750, 13.250, 13.750, 14.250, 14.750,
        15.250, 15.750, 16.250, 16.750, 17.250, 17.750, 18.250, 18.750, 19.250, 19.750,
        20.250, 20.750, 21.250, 21.750, 22.250, 22.750, 23.250, 23.750, 24.250, 24.750,
        25.250, 25.750, 26.250, 26.750, 27.250, 27.750, 28.250, 28.750, 29.250, 29.750,
        30.250, 30.750, 31.250, 31.750, 32.250, 32.750, 33.250, 33.750, 34.250, 34.750,
        35.250, 35.750, 36.250, 36.750, 37.250, 37.750, 38.250, 38.750, 39.250, 39.750,
        40.250
    ],
    'wsname': [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
        10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
        15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
        20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
        25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5,
        30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5,
        35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5,
        40.0
    ]
})
# --- Wind direction sector mapping ---
def get_wd_sector_df():
    return pd.DataFrame({
        'wdfrom': [15.001, 45.001, 75.001, 105.001, 135.001, 165.001, 195.001, 225.001,
                   255.001, 285.001, 315.001, 0.000, 345.001],
        'wdto': [45.000, 75.000, 105.000, 135.000, 165.000, 195.000, 225.000, 255.000,
                 285.000, 315.000, 345.000, 15.000, 360.000],
        'wdirection': ['NNE', 'NE', 'E', 'SE', 'SSE', 'S', 'SSW', 'SW', 'W', 'NW', 'NNW', 'N', 'N']
    })

# --- Binning Functions ---
def map_wind_speed(col):
    ws_sector_df = get_ws_sector_df()
    result = col.copy()
    valid_mask = ~col.isin(FAULTY_VALUES)
    col_rounded = col[valid_mask].round(3)
    for i in range(len(ws_sector_df)):
        mask = (col_rounded >= ws_sector_df.loc[i, 'wsfrom']) & (col_rounded <= ws_sector_df.loc[i, 'wsto'])
        result.loc[valid_mask & mask] = ws_sector_df.loc[i, 'wsname']
    return result

def map_wind_direction(col):
    sector_df = get_wd_sector_df()
    result = col.copy()
    for i in range(len(sector_df)):
        wdfrom = sector_df.loc[i, 'wdfrom']
        wdto = sector_df.loc[i, 'wdto']
        direction = sector_df.loc[i, 'wdirection']
        if wdfrom <= wdto:
            mask = (col >= wdfrom) & (col <= wdto)
        else:
            mask = (col >= wdfrom) | (col <= wdto)
        mask &= ~col.isin(FAULTY_VALUES)
        result.loc[mask] = direction
    return result

# --- Apply mappings ---
def process_wind_columns(df):
    if 'wind_direction' in df.columns:
        df['wind_direction'] = df['wind_direction'].astype('float32')
        df['wind_direction'] = map_wind_direction(df['wind_direction'])

    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'].astype('float32')
        df['wind_speed'] = map_wind_speed(df['wind_speed'])

    return df

COLUMNS_TO_DROP = [
    'sno', 'flidarid', 'group', 'bearing', 'statusflag', 'infoflag', 'axysflag', 'algorithm', 'battery',
    'generator', 'utemp', 'ltemp', 'podhumid', 'metcmpbearing', 'mettilt', 'pcktsrain', 'pcktsfog',
    'rangeflag', 'constantflag', 'rocflag', 'spikeflag'
]
FAILURE_MAPPING = {
    1000: 'maintenance',
    9999: 'sensor_fail1',
    9998: 'sensor_fail2',
    500: 'Wind_dir_fail',
    600: 'wind_spd_fail'
}

# --- Shared logic for preprocessing (flattening, handling flags, etc.) ---
def common_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unwanted columns
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')

    # Flatten by height
    all_records = []
    for _, row in df.iterrows():
        for i in range(1, 12):
            all_records.append({
                'windkey': row.get('windkey'),
                'datetimekey': row.get('datetimekey'),
                'height': i,
                'wind_direction': row.get(f"wdirtnavg{i}"),
                'wind_speed': row.get(f"hwsavg{i}"),
                'internal_flag': row.get(f"intflag{i}"),
                'external_flag': row.get(f"extflag{i}"),
                'lidar_enabled': row.get(f"enabled{i}"),
            })

    df = pd.DataFrame(all_records)

    # Handle nulls
    df.loc[df['windkey'].isna() & df['wind_direction'].isna(), 'wind_direction'] = 1000
    df.loc[df['windkey'].isna() & df['wind_speed'].isna(), 'wind_speed'] = 1000
    df.loc[df['wind_direction'].isna() & df['wind_speed'].notna(), 'wind_direction'] = 500
    df.loc[df['wind_direction'].notna() & df['wind_speed'].isna(), 'wind_speed'] = 600
    df.loc[(df['internal_flag'] == 77) & (df['wind_direction'].isna()), 'wind_direction'] = 9998.0
    df.loc[(df['internal_flag'] == 77) & (df['wind_speed'].isna()), 'wind_speed'] = 9998.0

    # Failure type
    df['type_failure'] = df['wind_direction'].map(FAILURE_MAPPING)
    df['type_failure'] = df['type_failure'].fillna(df['wind_speed'].map(FAILURE_MAPPING))

    # Time-based feature
    df['datetimekey'] = pd.to_datetime(df['datetimekey'])
    df['hour'] = df['datetimekey'].dt.hour
    df = df.drop(columns=['datetimekey', 'windkey'], errors='ignore')

    # Normalize direction
    df['wind_direction'] = df['wind_direction'].apply(lambda x: x % 360 if x not in FAULTY_VALUES else x)

    # Dedup
    df = df.drop_duplicates()

    return df
# --- Training Preprocessing using in-memory DataFrame only ---
def preprocess_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info('Whole data basic preprocess start')
        df = common_preprocessing(df)
        logging.info('Whole data basic preprocess Ends')
        return df
    except Exception as e:
        raise CustomException(e, sys)


def run_pipeline_from_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Pipeline processing started...")

        df['type_failure'] = df['type_failure'].fillna('normal')
        df['wind_direction'] = df['wind_direction'].round(0)
        df['wind_speed'] = df['wind_speed'].round(3)

        df = process_wind_columns(df)
        df = df.drop_duplicates()

        logging.info("✅ Pipeline complete.")
        return df

    except Exception as e:
        raise CustomException(e, sys)


def run_full_preprocessing_pipeline_from_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("🔥 Full preprocessing + wind binning pipeline started")

        df_preprocessed = preprocess_raw_df(raw_df)
        df_final = run_pipeline_from_df(df_preprocessed)

        logging.info("🔥 Full preprocessing pipeline completed successfully")
        return df_final

    except Exception as e:
        raise CustomException(e, sys)


