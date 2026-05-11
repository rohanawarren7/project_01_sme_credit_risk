"""
Alternative Data Feature Engineer.

Aggregates synthetic alternative signals into feature vectors.
"""

import pandas as pd
import numpy as np
from typing import Dict


class AlternativeFeatureEngineer:
    """Transform enrichment data into model features."""
    
    def __init__(self,
                 google_reviews_df: pd.DataFrame,
                 trustpilot_df: pd.DataFrame,
                 linkedin_df: pd.DataFrame,
                 property_vacancy_df: pd.DataFrame,
                 companies_house_df: pd.DataFrame):
        self.google = google_reviews_df.copy()
        self.trustpilot = trustpilot_df.copy()
        self.linkedin = linkedin_df.copy()
        self.property = property_vacancy_df.copy()
        self.companies_house = companies_house_df.copy()
        
        # Parse dates
        for df, col in [(self.google, 'review_date'), (self.trustpilot, 'review_date'),
                        (self.linkedin, 'scrape_date'), (self.property, 'date')]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    
    def build_features(self, company_id: str, as_of_date: pd.Timestamp,
                       postcode: str = None) -> Dict:
        """Build alternative feature vector for an entity as of a date."""
        features = {}
        
        # 1. Google Reviews sentiment velocity
        google_reviews = self.google[
            (self.google['company_id'] == company_id) &
            (self.google['review_date'] <= as_of_date)
        ]
        if len(google_reviews) >= 3:
            google_reviews['days_ago'] = (as_of_date - google_reviews['review_date']).dt.days
            google_reviews['weight'] = np.exp(-google_reviews['days_ago'] / 30)
            features['google_sentiment_mean'] = np.average(
                google_reviews['sentiment_score'], weights=google_reviews['weight']
            )
            features['google_sentiment_trend'] = np.polyfit(
                np.arange(len(google_reviews)), google_reviews['sentiment_score'], 1
            )[0] if len(google_reviews) >= 6 else 0
            features['google_review_velocity'] = len(google_reviews[
                google_reviews['days_ago'] <= 90
            ]) / 3.0
        else:
            features['google_sentiment_mean'] = np.nan
            features['google_sentiment_trend'] = np.nan
            features['google_review_velocity'] = np.nan
        
        # 2. Trustpilot velocity
        trustpilot = self.trustpilot[
            (self.trustpilot['company_id'] == company_id) &
            (self.trustpilot['review_date'] <= as_of_date)
        ]
        if len(trustpilot) > 0:
            trustpilot['days_ago'] = (as_of_date - trustpilot['review_date']).dt.days
            features['trustpilot_90d_count'] = (trustpilot['days_ago'] <= 90).sum()
            features['trustpilot_rating_mean'] = trustpilot['rating'].mean()
        else:
            features['trustpilot_90d_count'] = np.nan
            features['trustpilot_rating_mean'] = np.nan
        
        # 3. LinkedIn headcount growth
        linkedin = self.linkedin[
            (self.linkedin['company_id'] == company_id) &
            (self.linkedin['scrape_date'] <= as_of_date)
        ]
        if len(linkedin) >= 2:
            headcount_series = linkedin.sort_values('scrape_date')['headcount_estimate'].values
            if len(headcount_series) >= 4:
                features['headcount_growth_rate'] = np.polyfit(
                    np.arange(len(headcount_series)), headcount_series, 1
                )[0] / headcount_series.mean() if headcount_series.mean() > 0 else 0
            else:
                features['headcount_growth_rate'] = np.nan
            features['headcount_latest'] = headcount_series[-1]
        else:
            features['headcount_growth_rate'] = np.nan
            features['headcount_latest'] = np.nan
        
        # 4. Local commercial vacancy
        if postcode:
            postcode_area = postcode[:2]
            property_df = self.property[
                (self.property['postcode_area'] == postcode_area) &
                (self.property['date'] <= as_of_date)
            ].sort_values('date')
            if len(property_df) > 0:
                latest = property_df.iloc[-1]
                features['local_vacancy_rate'] = latest['vacancy_rate']
                features['local_vacancy_trend'] = np.polyfit(
                    np.arange(len(property_df)), property_df['vacancy_rate'], 1
                )[0] if len(property_df) >= 6 else 0
            else:
                features['local_vacancy_rate'] = np.nan
                features['local_vacancy_trend'] = np.nan
        else:
            features['local_vacancy_rate'] = np.nan
            features['local_vacancy_trend'] = np.nan
        
        # 5. Companies House features
        ch = self.companies_house[self.companies_house['company_id'] == company_id]
        if len(ch) > 0:
            features['num_officers'] = ch['number_of_officers'].iloc[0]
            features['num_shareholders'] = ch['number_of_shareholders'].iloc[0]
            features['share_capital'] = ch['share_capital'].iloc[0]
            features['mortgages_count'] = ch['mortgages_count'].iloc[0]
            features['charges_count'] = ch['charges_count'].iloc[0]
        else:
            features['num_officers'] = np.nan
            features['num_shareholders'] = np.nan
            features['share_capital'] = np.nan
            features['mortgages_count'] = np.nan
            features['charges_count'] = np.nan
        
        return features
