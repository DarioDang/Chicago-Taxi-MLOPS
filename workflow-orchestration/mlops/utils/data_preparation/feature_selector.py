import pandas as pd
from typing import List, Optional

CATEGORICAL_FEATURES = ['PU_DO']
NUMERICAL_FEATURES = [
    'trip_miles', 'is_weekend', 'fare_per_mile', 'hour', 'day_of_week'
]

def select_features(df: pd.DataFrame, features: Optional[List[str]] = None, top_pudo_limit: int = 1000) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    if features:
        columns += features

    columns = list(dict.fromkeys(columns))
    df = df[[col for col in columns if col in df.columns]]

    if 'PU_DO' in df.columns:
        top_pudo = df['PU_DO'].value_counts().nlargest(top_pudo_limit).index
        df['PU_DO'] = df['PU_DO'].where(df['PU_DO'].isin(top_pudo), "Other")

    return df
