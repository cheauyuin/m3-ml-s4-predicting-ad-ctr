"""Reusable feature-engineering helpers for the CTR project.

Import these in notebooks with:
    import sys; sys.path.append('../src')
    from features import add_time_features
"""
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the Avazu 'hour' column (format YYMMDDHH) into useful parts."""
    df = df.copy()
    df["hour_of_day"] = (df["hour"] % 100).astype(int)
    df["day"] = ((df["hour"] // 100) % 100).astype(int)
    return df
