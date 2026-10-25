"""
Data Preprocessing Pipeline for Zero-Day GraphIDS.
Handles schema detection, data sanitation, feature normalization, and leakage prevention.
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Common label column names across benchmark network datasets
LABEL_COLUMN_CANDIDATES = [
    'label', 'labels', 'attack', 'attack_type', 'attack_cat', 'class', 'target'
]

# Identity and timestamp columns that risk target leakage or ungeneralizable bias
LEAKAGE_COLUMNS = [
    'flow_id', 'flow id', 'id', 'timestamp', 'time', 'date', 'simillarhttp',
    'fwd_header_length.1'
]

def clean_column_name(col: str) -> str:
    """Normalize column header: strip whitespace and replace unusual characters."""
    col = str(col).strip()
    return re.sub(r'\s+', ' ', col)

def detect_schema_type(columns: List[str]) -> str:
    """Detect whether dataset follows CIC-IDS2017, CTU-13, USTC-TFC2016, or Generic NetFlow."""
    cols_lower = [c.lower() for c in columns]
    if any('total fwd packets' in c or 'flow duration' in c for c in cols_lower):
        return 'CIC-IDS2017'
    elif any('totpkts' in c or 'srcaddr' in c for c in cols_lower):
        return 'CTU-13'
    elif any('packet_length' in c or 'bytes' in c for c in cols_lower):
        return 'USTC-TFC2016'
    return 'Generic-NetFlow'

def identify_label_column(columns: List[str]) -> Optional[str]:
    """Find the column most likely representing ground-truth traffic class."""
    for col in columns:
        if col.strip().lower() in LABEL_COLUMN_CANDIDATES:
            return col
    return None

def clean_and_sanitize_flow_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """
    Cleans raw DataFrame:
    1. Trims and standardizes column names.
    2. Identifies and isolates label column.
    3. Excludes metadata/leakage columns.
    4. Replaces infinite values with NaN and imputes median.
    5. Deduplicates records.
    """
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    label_col = identify_label_column(list(df.columns))

    # Identify leakage columns to remove
    cols_to_drop = []
    for c in df.columns:
        if c != label_col and c.lower() in LEAKAGE_COLUMNS:
            cols_to_drop.append(c)

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Convert numeric features, coerce errors to NaN
    feature_cols = [c for c in df.columns if c != label_col]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Check if column is categorical or can be numeric
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notnull().sum() > 0.5 * len(df):
                df[col] = converted
            else:
                # One-hot encode or factorize low cardinality categorical columns
                if 1 < df[col].nunique() < 20:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                else:
                    df.drop(columns=[col], inplace=True)

    # Re-evaluate feature columns after conversions: keep only numeric features
    feature_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]

    if feature_cols:
        # Replace inf and -inf with NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaNs with column median (or 0 if all NaN)
        for col in feature_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        # Drop zero variance columns to prevent division by zero
        non_zero_var_cols = []
        for col in feature_cols:
            if df[col].std() > 1e-7:
                non_zero_var_cols.append(col)
            elif len(df) == 1:
                non_zero_var_cols.append(col)

        if non_zero_var_cols:
            feature_cols = non_zero_var_cols

    if label_col:
        df = df[feature_cols + [label_col]]
    else:
        df = df[feature_cols]

    return df, label_col, feature_cols

def split_and_normalize_dataset(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits dataset into Train, Validation, and Test splits.
    Scales features using StandardScaler fitted ONLY on training set to prevent data leakage.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values if label_col else np.zeros(len(df))

    test_ratio = 1.0 - (train_ratio + val_ratio)
    stratify = y if label_col and len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None

    # First split: Train vs Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify
    )

    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_temp = y_temp if stratify is not None and min(pd.Series(y_temp).value_counts()) >= 2 else None

    # Second split: Val vs Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=temp_val_ratio,
        random_state=random_state,
        stratify=stratify_temp
    )

    # Fit scaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'label_col': label_col,
        'train': {'X': X_train_scaled, 'y': y_train, 'count': len(X_train)},
        'val': {'X': X_val_scaled, 'y': y_val, 'count': len(X_val)},
        'test': {'X': X_test_scaled, 'y': y_test, 'count': len(X_test)},
    }
