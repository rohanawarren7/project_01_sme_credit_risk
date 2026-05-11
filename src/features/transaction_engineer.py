"""
Transaction Feature Engineer.

Transforms 18 months of transaction history into stability ratios.
All methods accept an 'as_of_date' to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')
from src.entity_resolution.merchant_resolver import MerchantResolver


class TransactionFeatureEngineer:
    """Transform transaction history into stable, predictive risk ratios."""
    
    def __init__(self, transactions_df: pd.DataFrame,
                 merchant_mapping_df: Optional[pd.DataFrame] = None):
        self.df = transactions_df.copy()
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        
        if merchant_mapping_df is not None:
            self.df = self.df.merge(
                merchant_mapping_df[['raw', 'canonical']],
                left_on='merchant_name', right_on='raw', how='left'
            )
            self.df['canonical'] = self.df['canonical'].fillna(self.df['merchant_name'])
        else:
            self.df['canonical'] = self.df['merchant_name']
    
    def _filter_as_of(self, entity_id: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """Filter transactions up to as_of_date for a specific entity."""
        df = self.df[
            (self.df['company_id'] == entity_id) &
            (self.df['transaction_date'] <= as_of_date)
        ].copy()
        return df
    
    def stability_ratios(self, entity_id: str, as_of_date: pd.Timestamp) -> Dict:
        """Compute core stability features."""
        df = self._filter_as_of(entity_id, as_of_date)
        
        if len(df) < 10:
            return {f: np.nan for f in self._stability_feature_names()}
        
        # Monthly aggregation
        df['month'] = df['transaction_date'].dt.to_period('M')
        monthly = df.groupby('month')['amount'].agg(['sum', 'mean', 'count'])
        
        # Inflow/outflow split
        inflows = df[df['amount'] > 0]
        outflows = df[df['amount'] < 0]
        
        monthly_in = inflows.groupby(inflows['transaction_date'].dt.to_period('M'))['amount'].sum()
        monthly_out = outflows.groupby(outflows['transaction_date'].dt.to_period('M'))['amount'].sum().abs()
        
        features = {}
        
        # 1. Coefficient of variation of monthly net
        monthly_net = monthly_in.add(monthly_out, fill_value=0)
        features['cv_monthly_net'] = monthly_net.std() / abs(monthly_net.mean()) if monthly_net.mean() != 0 else np.nan
        
        # 2. Net operating flow ratio
        features['net_operating_ratio'] = monthly_in.sum() / monthly_out.sum() if monthly_out.sum() > 0 else np.nan
        
        # 3. Transaction frequency trend
        if len(monthly) >= 3:
            x = np.arange(len(monthly))
            y = monthly['count'].values
            slope = np.polyfit(x, y, 1)[0]
            features['txn_frequency_trend'] = slope / monthly['count'].mean() if monthly['count'].mean() > 0 else 0
        else:
            features['txn_frequency_trend'] = np.nan
        
        # 4. Days-without-transaction streak (last 90 days)
        last_90 = df[df['transaction_date'] >= as_of_date - pd.Timedelta(days=90)]
        if len(last_90) > 0:
            daily = last_90.groupby('transaction_date')['amount'].sum()
            date_range = pd.date_range(as_of_date - pd.Timedelta(days=89), as_of_date, freq='D')
            daily = daily.reindex(date_range, fill_value=0)
            is_zero = daily == 0
            if is_zero.any():
                # Group consecutive zeros
                streaks = is_zero.astype(int).groupby((~is_zero).cumsum()).sum()
                features['max_zero_day_streak_90d'] = streaks.max()
            else:
                features['max_zero_day_streak_90d'] = 0
        else:
            features['max_zero_day_streak_90d'] = 90
        
        # 5. Recurring revenue ratio
        if len(inflows) > 0:
            inflow_merchants = inflows.groupby('canonical')['amount'].sum().sort_values(ascending=False)
            features['recurring_revenue_ratio'] = inflow_merchants.head(3).sum() / inflow_merchants.sum() if inflow_merchants.sum() > 0 else np.nan
        else:
            features['recurring_revenue_ratio'] = np.nan
        
        # 6. Months runway estimate
        df_sorted = df.sort_values('transaction_date')
        df_sorted['cumulative'] = df_sorted['amount'].cumsum()
        rolling_max = df_sorted.set_index('transaction_date')['cumulative'].rolling('90D').max()
        cash_buffer = rolling_max.iloc[-1] if len(rolling_max) > 0 else 0
        monthly_burn = monthly_out.mean() if len(monthly_out) > 0 and monthly_out.mean() != 0 else 0
        features['months_runway'] = cash_buffer / monthly_burn if monthly_burn > 0 else np.nan
        
        # 7. Supplier concentration (Herfindahl index)
        if len(outflows) > 0:
            outflow_merchants = outflows.groupby('canonical')['amount'].sum().abs()
            shares = outflow_merchants / outflow_merchants.sum()
            features['supplier_hhi'] = (shares ** 2).sum() if outflow_merchants.sum() > 0 else np.nan
        else:
            features['supplier_hhi'] = np.nan
        
        # 8. Velocity: 90d vs 180d trend ratio
        d90 = df[df['transaction_date'] >= as_of_date - pd.Timedelta(days=90)]['amount'].sum()
        d180 = df[df['transaction_date'] >= as_of_date - pd.Timedelta(days=180)]['amount'].sum()
        features['velocity_90_180'] = d90 / (d180 - d90) if (d180 - d90) != 0 else np.nan
        
        # Additional features
        features['txn_count_90d'] = len(df[df['transaction_date'] >= as_of_date - pd.Timedelta(days=90)])
        features['txn_count_180d'] = len(df[df['transaction_date'] >= as_of_date - pd.Timedelta(days=180)])
        features['mean_txn_amount'] = df['amount'].abs().mean()
        features['median_txn_amount'] = df['amount'].abs().median()
        features['std_txn_amount'] = df['amount'].abs().std()
        features['total_inflow'] = inflows['amount'].sum() if len(inflows) > 0 else 0
        features['total_outflow'] = abs(outflows['amount'].sum()) if len(outflows) > 0 else 0
        date_span = (df['transaction_date'].max() - df['transaction_date'].min()).days
        features['active_days_ratio'] = df['transaction_date'].nunique() / date_span if date_span > 0 else 1.0
        
        return features
    
    def _stability_feature_names(self) -> list:
        return [
            'cv_monthly_net', 'net_operating_ratio', 'txn_frequency_trend',
            'max_zero_day_streak_90d', 'recurring_revenue_ratio', 'months_runway',
            'supplier_hhi', 'velocity_90_180', 'txn_count_90d', 'txn_count_180d',
            'mean_txn_amount', 'median_txn_amount', 'std_txn_amount',
            'total_inflow', 'total_outflow', 'active_days_ratio'
        ]
