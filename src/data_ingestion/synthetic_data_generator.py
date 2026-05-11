"""
Synthetic Data Generator for SME Credit Risk Engine.

Generates realistic Open Banking transactions, enrichment data, and default labels
for 2,000+ synthetic SMEs over an 18-month window.

Design principles:
- No real PII: all data is algorithmically generated
- Realistic noise: inconsistent merchant names, missing categories, seasonality
- Embedded default signal: ~15% default rate with leading indicators
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import json
import os

# Seed for reproducibility
RNG = np.random.default_rng(42)

# Merchant name taxonomy for realistic inconsistency
MERCHANT_TEMPLATES = {
    "Starbucks": [
        "Starbucks", "Starbucks Ltd", "STARBUCKS LIMITED", "Starbucks #084381",
        "Starbucks Store 84381", "SBUK RETAIL", "SQ *STARBUCKS"
    ],
    "Amazon": [
        "Amazon", "AMAZON.CO.UK", "Amazon EU SARL", "Amazon Marketplace",
        "AMZN MKTP", "SQ *AMAZON"
    ],
    "Tesco": [
        "Tesco", "Tesco PLC", "TESCO STORES LTD", "Tesco Metro #4412",
        "Tesco Express", "TESCO RETAIL"
    ],
    "Sainsburys": [
        "Sainsbury's", "J Sainsbury PLC", "SAINSBURYS LTD", "Sainsburys Local",
        "Sainsburys Supermarket"
    ],
    "Uber": [
        "Uber", "UBER TRIP", "UBER EATS", "Uber BV", "UBER *TRIP HELP.UBER.COM"
    ],
    "Shell": [
        "Shell", "Shell UK Ltd", "SHELL PETROL", "Shell Station #2291",
        "SHELL RETAIL"
    ],
    "BP": [
        "BP", "BP Connect", "BP OIL UK", "BP Station 8821", "BP RETAIL LTD"
    ],
    "QuickBooks": [
        "QuickBooks", "INTUIT QUICKBOOKS", "QB Online", "QuickBooks Subscription"
    ],
    "Xero": [
        "Xero", "XERO SOFTWARE", "Xero Ltd", "Xero Monthly"
    ],
    "Slack": [
        "Slack", "SLACK TECHNOLOGIES", "Slack Technologies Ltd", "Slack Subscription"
    ],
    "Microsoft": [
        "Microsoft", "MICROSOFT*STORE", "MSFT", "Microsoft 365",
        "Microsoft Corporation"
    ],
    "Google": [
        "Google", "GOOGLE *SERVICES", "Google Ireland", "Google Cloud",
        "GOOGLE *ADS"
    ],
    "AWS": [
        "Amazon Web Services", "AWS", "AMAZON WEB SERV", "AWS EMEA"
    ],
    "DigitalOcean": [
        "DigitalOcean", "DIGITALOCEAN.COM", "Digital Ocean LLC"
    ],
    "Stripe": [
        "Stripe", "STRIPE PAYMENTS", "Stripe Technology Europe Ltd"
    ],
    "GoCardless": [
        "GoCardless", "GOCARDLESS LTD", "GoCardless Payment"
    ],
    "Rent": [
        "ABC Property Mgmt", "Westminster Lettings", "City Centre Offices Ltd",
        "Regus Business Centre", "WeWork", "SPACES"
    ],
    "Utilities": [
        "British Gas", "British Gas Business", "EDF Energy", "E.ON UK",
        "SSE Business Energy", "Thames Water", "ScottishPower"
    ],
    "Insurance": [
        "Aviva", "AXA Insurance", "Direct Line", "Hiscox", "Zurich Insurance",
        "Allianz UK"
    ],
    "Accountant": [
        "Smith & Co Accountants", "BDO LLP", "Grant Thornton", "Mazars",
        "PKF Littlejohn", "Moore Kingston Smith"
    ],
    "Legal": [
        "Shoosmiths", "Trowers & Hamlins", "DWF Law LLP", "Mills & Reeve",
        "Hill Dickinson"
    ],
    "Courier": [
        "DHL", "DHL Express", "FedEx", "UPS", "Royal Mail", "Parcelforce",
        "Hermes", "DPD"
    ],
    "Catering": [
        "Deliveroo", "DELIVEROO", "Just Eat", "JUST EAT TAKEAWAY",
        "Uber Eats"
    ],
    "Stationery": [
        "Staples", "Ryman", "WHSmith", "Viking Direct", "Office Depot"
    ],
    "Travel": [
        "Booking.com", "EXPEDIA", "Trainline", "National Rail", "EasyJet",
        "British Airways", "Ryanair"
    ],
}

# Industry profiles with different spending patterns
INDUSTRY_PROFILES = {
    "Technology": {
        "avg_monthly_txns": 80,
        "avg_txn_amount": 450,
        "recurring_merchants": ["QuickBooks", "Xero", "Slack", "AWS", "DigitalOcean"],
        "supplier_concentration": 0.6,
        "seasonality": False,
        "default_base_rate": 0.12,
    },
    "Retail": {
        "avg_monthly_txns": 150,
        "avg_txn_amount": 280,
        "recurring_merchants": ["Tesco", "Sainsburys", "Courier", "Rent", "Utilities"],
        "supplier_concentration": 0.7,
        "seasonality": True,
        "seasonal_peak_months": [11, 12],
        "seasonal_amplitude": 2.5,
        "default_base_rate": 0.18,
    },
    "Professional Services": {
        "avg_monthly_txns": 45,
        "avg_txn_amount": 1200,
        "recurring_merchants": ["Rent", "Accountant", "Insurance", "Utilities", "Legal"],
        "supplier_concentration": 0.5,
        "seasonality": False,
        "default_base_rate": 0.10,
    },
    "Hospitality": {
        "avg_monthly_txns": 200,
        "avg_txn_amount": 180,
        "recurring_merchants": ["Tesco", "Catering", "Utilities", "Rent", "Insurance"],
        "supplier_concentration": 0.75,
        "seasonality": True,
        "seasonal_peak_months": [6, 7, 8, 12],
        "seasonal_amplitude": 2.0,
        "default_base_rate": 0.22,
    },
    "Construction": {
        "avg_monthly_txns": 60,
        "avg_txn_amount": 850,
        "recurring_merchants": ["Utilities", "Rent", "Insurance", "Shell", "BP"],
        "supplier_concentration": 0.55,
        "seasonality": True,
        "seasonal_peak_months": [4, 5, 6, 7, 8, 9],
        "seasonal_amplitude": 1.8,
        "default_base_rate": 0.16,
    },
    "Healthcare": {
        "avg_monthly_txns": 70,
        "avg_txn_amount": 380,
        "recurring_merchants": ["Utilities", "Insurance", "Rent", "Stationery", "Courier"],
        "supplier_concentration": 0.5,
        "seasonality": False,
        "default_base_rate": 0.08,
    },
}

CATEGORY_MAP = {
    "Starbucks": "Food & Drink", "Amazon": "Supplies", "Tesco": "Food & Drink",
    "Sainsburys": "Food & Drink", "Uber": "Travel", "Shell": "Fuel",
    "BP": "Fuel", "QuickBooks": "Software", "Xero": "Software", "Slack": "Software",
    "Microsoft": "Software", "Google": "Software", "AWS": "Software",
    "DigitalOcean": "Software", "Stripe": "Payment Processing",
    "GoCardless": "Payment Processing", "Rent": "Rent", "Utilities": "Utilities",
    "Insurance": "Insurance", "Accountant": "Professional Services",
    "Legal": "Professional Services", "Courier": "Logistics",
    "Catering": "Food & Drink", "Stationery": "Supplies", "Travel": "Travel",
}


def generate_sme_profiles(n_smes: int = 2000) -> pd.DataFrame:
    """Generate synthetic SME company profiles."""
    industries = list(INDUSTRY_PROFILES.keys())
    industry_weights = [len(p["recurring_merchants"]) for p in INDUSTRY_PROFILES.values()]
    
    smes = []
    for i in range(n_smes):
        industry = RNG.choice(industries, p=np.array(industry_weights) / sum(industry_weights))
        profile = INDUSTRY_PROFILES[industry]
        
        company_age_months = int(RNG.exponential(36)) + 1
        loan_amount = round(RNG.lognormal(10, 1.2), 2)
        
        sme = {
            "company_id": f"SME-{i+1:05d}",
            "company_name": f"Company {i+1}",
            "industry": industry,
            "incorporation_date": (datetime(2020, 1, 1) - timedelta(days=company_age_months * 30)).strftime("%Y-%m-%d"),
            "company_age_months": company_age_months,
            "postcode": f"{RNG.integers(1, 100):02d}{RNG.choice(['A', 'B', 'C', 'D', 'E'])}{RNG.integers(1, 10)} {RNG.integers(1, 10)}{RNG.choice(['A', 'B', 'C', 'D', 'E'])}{RNG.choice(['A', 'B', 'C', 'D', 'E'])}",
            "loan_amount": loan_amount,
            "num_employees": max(1, int(RNG.lognormal(2, 0.8))),
            "default_base_rate": profile["default_base_rate"],
        }
        smes.append(sme)
    
    return pd.DataFrame(smes)


def generate_transactions(
    sme_profile: pd.Series,
    start_date: datetime = datetime(2022, 1, 1),
    end_date: datetime = datetime(2023, 6, 30),
    is_defaulter: bool = False,
    default_date: datetime = None,
) -> pd.DataFrame:
    """Generate 18 months of transactions for a single SME - OPTIMIZED VERSION."""
    profile = INDUSTRY_PROFILES[sme_profile["industry"]]
    
    # Generate all dates first
    all_dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(all_dates)
    
    # Base transaction count
    base_daily_txns = profile["avg_monthly_txns"] / 30.0
    
    # Seasonality adjustment
    seasonal_multipliers = np.ones(n_days)
    if profile.get("seasonality", False):
        for i, d in enumerate(all_dates):
            if d.month in profile.get("seasonal_peak_months", []):
                seasonal_multipliers[i] = profile.get("seasonal_amplitude", 1.0)
    
    # Distress adjustment for defaulters
    distress_multipliers = np.ones(n_days)
    if is_defaulter and default_date:
        for i, d in enumerate(all_dates):
            days_to_default = (default_date - d).days
            if 0 < days_to_default <= 180:
                distress_multipliers[i] = 0.5 + 0.5 * (days_to_default / 180.0)
    
    daily_txn_counts = RNG.poisson(base_daily_txns * seasonal_multipliers * distress_multipliers)
    total_txns = daily_txn_counts.sum()
    
    if total_txns == 0:
        total_txns = 1
        daily_txn_counts[0] = 1
    
    # Build transaction lists
    txn_dates = np.repeat(all_dates, daily_txn_counts)
    
    # Inflow/outflow
    is_inflow = RNG.random(total_txns) < 0.35
    
    # Amounts
    base_amount = profile["avg_txn_amount"]
    amounts = np.zeros(total_txns)
    amounts[is_inflow] = abs(RNG.lognormal(np.log(base_amount * 1.5), 0.6, is_inflow.sum()))
    amounts[~is_inflow] = -abs(RNG.lognormal(np.log(base_amount), 0.8, (~is_inflow).sum()))
    
    # Distress amount volatility
    if is_defaulter and default_date:
        days_to_default_arr = np.array([(default_date - pd.Timestamp(d)).days for d in txn_dates])
        distress_mask = (days_to_default_arr > 0) & (days_to_default_arr <= 180)
        if distress_mask.any():
            distress_multipliers_amount = np.where(
                RNG.random(distress_mask.sum()) < 0.4, 0.3, 3.0
            )
            amounts[distress_mask] *= distress_multipliers_amount
    
    amounts = np.round(amounts, 2)
    
    # Merchant selection
    all_merchant_keys = list(MERCHANT_TEMPLATES.keys())
    recurring = profile["recurring_merchants"]
    weights = [3 if k in recurring else 1 for k in all_merchant_keys]
    weights = np.array(weights) / sum(weights)
    
    selected_keys = RNG.choice(all_merchant_keys, size=total_txns, p=weights)
    merchant_names = [RNG.choice(MERCHANT_TEMPLATES[k]) for k in selected_keys]
    categories = [CATEGORY_MAP.get(k, "Other") for k in selected_keys]
    
    # Missing categories (3%)
    missing_mask = RNG.random(total_txns) < 0.03
    for i in np.where(missing_mask)[0]:
        categories[i] = None
    
    payment_methods = RNG.choice(
        ["Direct Debit", "Card", "Bank Transfer", "Standing Order"], total_txns
    )
    references = [f"REF-{RNG.integers(100000, 999999)}" for _ in range(total_txns)]
    
    df = pd.DataFrame({
        "company_id": sme_profile["company_id"],
        "transaction_date": [d.strftime("%Y-%m-%d") for d in txn_dates],
        "amount": amounts,
        "merchant_name": merchant_names,
        "category": categories,
        "payment_method": payment_methods,
        "reference": references,
    })
    
    return df


def generate_enrichment_data(sme_profiles: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate synthetic enrichment data sources."""
    
    companies_house = []
    for _, sme in sme_profiles.iterrows():
        ch = {
            "company_id": sme["company_id"],
            "company_name": sme["company_name"],
            "company_number": f"{RNG.integers(10000000, 99999999)}",
            "incorporation_date": sme["incorporation_date"],
            "company_status": "Active",
            "company_type": RNG.choice(["Private limited company", "Limited liability partnership"]),
            "registered_office_postcode": sme["postcode"],
            "sic_code": RNG.choice(["62012", "47110", "69201", "41201", "41100", "86900"]),
            "accounts_next_due": (datetime(2023, 12, 31) + timedelta(days=int(RNG.integers(0, 365)))).strftime("%Y-%m-%d"),
            "confirmation_statement_next_due": (datetime(2023, 12, 31) + timedelta(days=int(RNG.integers(0, 365)))).strftime("%Y-%m-%d"),
            "filing_frequency": "Annual",
            "number_of_officers": max(1, RNG.poisson(2)),
            "number_of_shareholders": max(1, RNG.poisson(3)),
            "share_capital": round(RNG.lognormal(8, 1.5), 2),
            "mortgages_count": RNG.poisson(0.3),
            "charges_count": RNG.poisson(0.5),
        }
        companies_house.append(ch)
    
    # Google Reviews
    google_reviews = []
    for _, sme in sme_profiles.iterrows():
        n_reviews = RNG.poisson(12)
        for _ in range(n_reviews):
            review_date = datetime(2022, 1, 1) + timedelta(days=int(RNG.integers(0, 545)))
            google_reviews.append({
                "company_id": sme["company_id"],
                "review_date": review_date.strftime("%Y-%m-%d"),
                "rating": max(1, min(5, int(RNG.normal(3.8, 0.8)))),
                "sentiment_score": max(-1, min(1, RNG.normal(0.3, 0.4))),
            })
    
    # Trustpilot
    trustpilot = []
    for _, sme in sme_profiles.iterrows():
        if RNG.random() < 0.6:
            n_reviews = RNG.poisson(8)
            for _ in range(n_reviews):
                review_date = datetime(2022, 1, 1) + timedelta(days=int(RNG.integers(0, 545)))
                trustpilot.append({
                    "company_id": sme["company_id"],
                    "review_date": review_date.strftime("%Y-%m-%d"),
                    "rating": max(1, min(5, int(RNG.normal(3.5, 0.9)))),
                })
    
    # LinkedIn
    linkedin = []
    for _, sme in sme_profiles.iterrows():
        base_headcount = sme["num_employees"]
        for month in pd.date_range("2022-01-01", "2023-06-30", freq="MS"):
            noise = RNG.normal(0, base_headcount * 0.1)
            linkedin.append({
                "company_id": sme["company_id"],
                "scrape_date": month.strftime("%Y-%m-%d"),
                "headcount_estimate": max(1, int(base_headcount + noise)),
            })
    
    # Property vacancy
    postcode_areas = sme_profiles["postcode"].str[:2].unique()
    property_vacancy = []
    for area in postcode_areas:
        base_rate = RNG.uniform(0.05, 0.25)
        for month in pd.date_range("2022-01-01", "2023-06-30", freq="MS"):
            trend = (month - datetime(2022, 1, 1)).days / 545 * 0.05
            property_vacancy.append({
                "postcode_area": area,
                "date": month.strftime("%Y-%m-%d"),
                "vacancy_rate": round(base_rate + trend + RNG.normal(0, 0.02), 4),
            })
    
    return {
        "companies_house": pd.DataFrame(companies_house),
        "google_reviews": pd.DataFrame(google_reviews),
        "trustpilot": pd.DataFrame(trustpilot),
        "linkedin": pd.DataFrame(linkedin),
        "property_vacancy": pd.DataFrame(property_vacancy),
    }


def generate_default_labels(sme_profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate default labels with realistic signal structure."""
    defaults = []
    
    for _, sme in sme_profiles.iterrows():
        base_rate = sme["default_base_rate"]
        
        if sme["company_age_months"] < 12:
            base_rate += 0.08
        elif sme["company_age_months"] < 24:
            base_rate += 0.04
        
        if sme["loan_amount"] > 50000:
            base_rate += 0.02
        
        defaulted = RNG.random() < base_rate
        
        if defaulted:
            default_date = datetime(2022, 6, 1) + timedelta(days=int(RNG.integers(0, 365)))
            default_date_str = default_date.strftime("%Y-%m-%d")
        else:
            default_date_str = None
        
        defaults.append({
            "company_id": sme["company_id"],
            "defaulted": int(defaulted),
            "default_date": default_date_str,
            "application_date": "2022-01-01",
        })
    
    return pd.DataFrame(defaults)


def generate_all_data(n_smes: int = 2000, output_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """Generate complete synthetic dataset and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/open_banking", exist_ok=True)
    
    print(f"Generating {n_smes} SME profiles...")
    sme_profiles = generate_sme_profiles(n_smes)
    sme_profiles.to_csv(f"{output_dir}/sme_profiles.csv", index=False)
    
    print("Generating default labels...")
    default_labels = generate_default_labels(sme_profiles)
    default_labels.to_csv(f"{output_dir}/default_labels.csv", index=False)
    
    print("Generating transactions (this may take a moment)...")
    all_transactions = []
    for idx, sme in sme_profiles.iterrows():
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{n_smes} SMEs...")
        
        default_info = default_labels[default_labels["company_id"] == sme["company_id"]].iloc[0]
        is_defaulter = bool(default_info["defaulted"])
        dd = default_info["default_date"]
        default_date = datetime.strptime(dd, "%Y-%m-%d") if pd.notna(dd) and dd else None
        
        txns = generate_transactions(sme, is_defaulter=is_defaulter, default_date=default_date)
        all_transactions.append(txns)
    
    transactions_df = pd.concat(all_transactions, ignore_index=True)
    transactions_df.to_csv(f"{output_dir}/open_banking/transactions.csv", index=False)
    
    print("Generating enrichment data...")
    enrichment = generate_enrichment_data(sme_profiles)
    enrichment["companies_house"].to_csv(f"{output_dir}/companies_house/companies_house.csv", index=False)
    enrichment["google_reviews"].to_csv(f"{output_dir}/google_reviews/google_reviews.csv", index=False)
    enrichment["trustpilot"].to_csv(f"{output_dir}/trustpilot/trustpilot.csv", index=False)
    enrichment["linkedin"].to_csv(f"{output_dir}/linkedin/linkedin.csv", index=False)
    enrichment["property_vacancy"].to_csv(f"{output_dir}/property_vacancy/property_vacancy.csv", index=False)
    
    print(f"\nDataset generation complete!")
    print(f"  SMEs: {len(sme_profiles)}")
    print(f"  Transactions: {len(transactions_df):,}")
    print(f"  Default rate: {default_labels['defaulted'].mean():.1%}")
    print(f"  Output directory: {output_dir}")
    
    return {
        "sme_profiles": sme_profiles,
        "default_labels": default_labels,
        "transactions": transactions_df,
        **enrichment,
    }


if __name__ == "__main__":
    generate_all_data(n_smes=2000, output_dir="data/raw")
