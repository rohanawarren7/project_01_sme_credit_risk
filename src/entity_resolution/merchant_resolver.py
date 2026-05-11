"""
Entity Resolution Pipeline for Open Banking Merchant Names.

3-pass pipeline:
1. Deterministic normalisation (regex stripping)
2. Fuzzy match against canonical dictionary (rapidfuzz)
3. DBSCAN clustering on TF-IDF character n-grams for unmapped merchants
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from typing import List, Optional, Dict, Tuple
from src.entity_resolution.manual_mappings import MANUAL_MAPPINGS


class MerchantResolver:
    """Multi-pass entity resolution for Open Banking merchant names."""
    
    # Strip these exact suffixes at end of string
    LEGAL_SUFFIXES = [
        r'\bLTD\.?\s*$', r'\bLIMITED\.?\s*$', r'\bPLC\.?\s*$', r'\bLLP\.?\s*$',
        r'\bINC\.?\s*$', r'\bCORP\.?\s*$', r'\bCO\.?\s*$', r'\bUK\.?\s*$',
        r'\bBV\.?\s*$', r'\bSARL\.?\s*$', r'\bGMBH\.?\s*$'
    ]
    
    # Strip these prefixes from start of string (must be followed by * or space)
    PROCESSOR_PREFIXES = [
        r'^SQ\s*\*+\s*', r'^PAYPAL\s*\*+\s*', r'^STRIPE\s*\*+\s*',
        r'^SUMUP\s*\*+\s*', r'^ZETTLE\s*\*+\s*', r'^GOCARDLESS\s*\*+\s*',
        r'^ADYEN\s*\*+\s*', r'^AMZN\s*\*+\s*', r'^GOOGLE\s*\*+\s*',
        r'^MICROSOFT\s*\*+\s*', r'^UBER\s*\*+\s*', r'^MSFT\s*\*+\s*'
    ]
    
    # Remove these descriptors anywhere in string
    DESCRIPTORS = [
        r'\bRETAIL\b', r'\bSUPERMARKET\b', r'\bEXPRESS\b', r'\bMETRO\b',
        r'\bLOCAL\b', r'\bONLINE\b', r'\bSUBSCRIPTION\b', r'\bMONTHLY\b',
        r'\bSTORE\s*\d+\b', r'\bSTATION\s*\d+\b', r'#\d+',
        r'\bSERVICES\b', r'\bTECHNOLOGIES\b', r'\bPAYMENTS\b',
        r'\bPAYMENT\b', r'\bBUSINESS\b', r'\bCENTRE\b',
        r'\bMANAGEMENT\b', r'\bMGMT\b', r'\bOFFICES\b'
    ]
    
    def __init__(self, canonical_dictionary: Optional[List[str]] = None):
        self.canonical = [c for c in (canonical_dictionary or []) if pd.notna(c) and isinstance(c, str) and len(c.strip()) > 0]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        if self.canonical:
            self._fit_canonical()
    
    def _fit_canonical(self):
        if len(self.canonical) > 0:
            self.vectorizer.fit(self.canonical)
    
    def normalise(self, raw_name: str) -> str:
        """Pass 1: Deterministic cleaning."""
        if not raw_name or not isinstance(raw_name, str):
            return ""
        
        name = raw_name.upper().strip()
        
        # Strip processor prefixes first
        for prefix in self.PROCESSOR_PREFIXES:
            name = re.sub(prefix, '', name, flags=re.IGNORECASE)
        
        # Remove store/location codes
        name = re.sub(r'\s*#\d+\s*', ' ', name)
        name = re.sub(r'\s*STORE\s*\d+\s*', ' ', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*STATION\s*\d+\s*', ' ', name, flags=re.IGNORECASE)
        
        # Remove URLs and email fragments
        name = re.sub(r'HELP\.\w+\.COM', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\*\w+\.COM', '', name, flags=re.IGNORECASE)
        
        # Remove common descriptors
        for desc in self.DESCRIPTORS:
            name = re.sub(desc, '', name, flags=re.IGNORECASE)
        
        # Strip legal suffixes at end
        for suffix in self.LEGAL_SUFFIXES:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
        
        # Clean up punctuation and whitespace
        name = re.sub(r'[^A-Z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def fuzzy_resolve(self, normalised_name: str, threshold: int = 70) -> Tuple[str, int]:
        """Pass 2: Fuzzy match against canonical dictionary."""
        if not normalised_name or len(normalised_name) < 2:
            return (normalised_name, 0)
        
        if not self.canonical:
            return (normalised_name, 0)
        
        match, score, _ = process.extractOne(
            normalised_name, self.canonical, scorer=fuzz.token_sort_ratio
        )
        
        if score >= threshold:
            return (match, score)
        return (normalised_name, score)
    
    def cluster_unknowns(self, unknown_names: List[str], eps: float = 0.3) -> Dict[str, str]:
        """Pass 3: DBSCAN clustering on TF-IDF character n-grams for unmapped merchants."""
        unknown_names = [name for name in unknown_names if name and len(name.strip()) > 1]
        if len(unknown_names) < 2:
            return {name: name for name in unknown_names}
        
        X = self.vectorizer.fit_transform(unknown_names)
        
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(X)
        labels = clustering.labels_
        
        clusters = {}
        for label in set(labels):
            if label == -1:
                continue
            members = [unknown_names[i] for i in range(len(unknown_names)) if labels[i] == label]
            exemplar = min(members, key=len)
            for m in members:
                clusters[m] = exemplar
        
        return clusters
    
    def resolve(self, raw_names: List[str], fuzzy_threshold: int = 70, 
                cluster_eps: float = 0.3) -> pd.DataFrame:
        """Full pipeline: normalise → manual mapping → fuzzy match → cluster residuals."""
        results = []
        unknowns = []
        
        for raw in raw_names:
            norm = self.normalise(raw)
            
            # Check manual mappings first
            if norm in MANUAL_MAPPINGS:
                results.append({
                    'raw': raw, 'normalised': norm,
                    'canonical': MANUAL_MAPPINGS[norm], 'method': 'manual', 'score': 100
                })
                continue
            
            resolved, score = self.fuzzy_resolve(norm, threshold=fuzzy_threshold)
            
            if score >= fuzzy_threshold:
                results.append({
                    'raw': raw, 'normalised': norm,
                    'canonical': resolved, 'method': 'fuzzy', 'score': score
                })
            else:
                unknowns.append({'raw': raw, 'normalised': norm})
        
        # Cluster unknowns
        if unknowns:
            unknown_names = [u['normalised'] for u in unknowns]
            clusters = self.cluster_unknowns(unknown_names, eps=cluster_eps)
            
            for u in unknowns:
                canonical = clusters.get(u['normalised'], u['normalised'])
                results.append({
                    'raw': u['raw'], 'normalised': u['normalised'],
                    'canonical': canonical, 'method': 'clustered' if u['normalised'] in clusters else 'unknown',
                    'score': None
                })
        
        return pd.DataFrame(results)
    
    def build_canonical_dictionary(self, raw_names: List[str], 
                                   min_frequency: int = 3) -> List[str]:
        """Build canonical dictionary from raw names via normalisation + frequency filter."""
        normalised = [self.normalise(name) for name in raw_names]
        freq = pd.Series(normalised).value_counts()
        canonical = freq[freq >= min_frequency].index.tolist()
        # Remove empty strings
        canonical = [c for c in canonical if c and len(c.strip()) > 0]
        return canonical
