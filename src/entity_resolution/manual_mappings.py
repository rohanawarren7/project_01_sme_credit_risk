"""
Manual merchant name mappings for known problematic cases.

These supplement the automated pipeline for high-frequency merchants
where normalisation alone is insufficient.
"""

MANUAL_MAPPINGS = {
    # DigitalOcean variants
    "DIGITAL OCEAN LLC": "DIGITALOCEAN",
    "DIGITALOCEANCOM": "DIGITALOCEAN",
    "DIGITAL OCEAN": "DIGITALOCEAN",
    
    # AWS variants
    "AWS": "AMAZON WEB SERVICES",
    "AWS EMEA": "AMAZON WEB SERVICES",
    "AMAZON WEB SERV": "AMAZON WEB SERVICES",
    
    # QuickBooks variants
    "QB": "QUICKBOOKS",
    "QB ONLINE": "QUICKBOOKS",
    "INTUIT QUICKBOOKS": "QUICKBOOKS",
    
    # Xero variants
    "XERO SOFTWARE": "XERO",
    
    # Stripe variants
    "STRIPE TECHNOLOGY EUROPE": "STRIPE",
    
    # Shell variants
    "SHELL PETROL": "SHELL",
    
    # BP variants
    "BP CONNECT": "BP",
    "BP OIL UK": "BP",
    "BP STATION": "BP",
    
    # Uber variants
    "UBER TRIP": "UBER",
    "UBER EATS": "UBER",
    "UBER BV": "UBER",
    "UBER TRIP HELPUBERCOM": "UBER",
    
    # Tesco variants
    "TESCO METRO": "TESCO",
    "TESCO EXPRESS": "TESCO",
    "TESCO PLC": "TESCO",
    "TESCO STORES": "TESCO",
    "TESCO RETAIL": "TESCO",
    
    # Sainsburys variants
    "SAINSBURYS LTD": "SAINSBURYS",
    "SAINSBURYS LOCAL": "SAINSBURYS",
    "SAINSBURYS SUPERMARKET": "SAINSBURYS",
    "J SAINSBURY PLC": "SAINSBURYS",
    
    # Starbucks variants
    "STARBUCKS LTD": "STARBUCKS",
    "STARBUCKS LIMITED": "STARBUCKS",
    "STARBUCKS STORE": "STARBUCKS",
    "SBUK RETAIL": "STARBUCKS",
    "SQ STARBUCKS": "STARBUCKS",
    
    # Amazon variants
    "AMAZON EU SARL": "AMAZON",
    "AMAZON MARKETPLACE": "AMAZON",
    "AMZN MKTP": "AMAZON",
    "SQ AMAZON": "AMAZON",
    
    # Slack variants
    "SLACK TECHNOLOGIES": "SLACK",
    "SLACK SUBSCRIPTION": "SLACK",
    "SLACK TECHNOLOGIES LTD": "SLACK",
    
    # GoCardless variants
    "GOCARDLESS LTD": "GOCARDLESS",
    "GOCARDLESS PAYMENT": "GOCARDLESS",
    
    # Microsoft variants
    "MICROSOFT STORE": "MICROSOFT",
    "MSFT": "MICROSOFT",
    "MICROSOFT 365": "MICROSOFT",
    
    # Google variants
    "GOOGLE SERVICES": "GOOGLE",
    "GOOGLE IRELAND": "GOOGLE",
    "GOOGLE CLOUD": "GOOGLE",
    "GOOGLE ADS": "GOOGLE",
    "ADS": "GOOGLE",
    
    # BP variants
    "BP OIL": "BP",
    "BP OIL UK": "BP",
    "BP CONNECT": "BP",
    
    # Uber variants
    "TRIP": "UBER",
    
    # WHSmith / stationery
    "WHSMITH": "STATIONERY",
    "RYMAN": "STATIONERY",
    "VIKING DIRECT": "STATIONERY",
    "OFFICE DEPOT": "STATIONERY",
    
    # Legal variants
    "DWF LAW": "LEGAL",
    "HILL DICKINSON": "LEGAL",
    "TROWERS HAMLINS": "LEGAL",
    "MILLS REEVE": "LEGAL",
    "SHOOSMITHS": "LEGAL",
    
    # Accountant variants
    "SMITH CO ACCOUNTANTS": "ACCOUNTANT",
    "PKF LITTLEJOHN": "ACCOUNTANT",
    "MOORE KINGSTON SMITH": "ACCOUNTANT",
    
    # Courier variants
    "PARCELFORCE": "COURIER",
    "HERMES": "COURIER",
    "DPD": "COURIER",
    
    # Travel variants
    "BOOKINGCOM": "TRAVEL",
    "EXPEDIA": "TRAVEL",
    "TRAINLINE": "TRAVEL",
    "NATIONAL RAIL": "TRAVEL",
    "EASYJET": "TRAVEL",
    "BRITISH AIRWAYS": "TRAVEL",
    "RYANAIR": "TRAVEL",
    
    # Just Eat / Deliveroo
    "JUST EAT TAKEAWAY": "CATERING",
    "JUST EAT": "CATERING",
}
