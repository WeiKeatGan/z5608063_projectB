"""Finance lexicon extension for VADER.

Plain VADER is built for general sentiment and is weak on finance language:
about half of finance headlines score neutral, and many are false neutrals
("beats expectations", "downgraded to sell", "FDA approval"). This lexicon adds
single-token finance terms with scores on VADER's -4 (extremely negative) to
+4 (extremely positive) scale. VADER tokenises on whitespace and punctuation,
so the entries are single words (multiword phrases such as "falls short of"
cannot match a lexicon key and are not included).

These scores are domain judgment, not machine-learned. REVIEW THEM before you
submit: they are the part of the sentiment model that is your own contribution,
and every value should be something you can defend in the report.
"""
FINANCE_LEXICON: dict[str, float] = {
    # --- performance surprises ---
    "beat": 2.2, "beats": 2.2, "beating": 2.2,
    "exceeded": 2.5, "exceeds": 2.0, "topped": 2.0,
    "miss": -2.2, "misses": -2.2, "missed": -2.2, "missing": -1.8,
    "outperform": 2.0, "outperformed": 2.0, "outperforming": 2.0, "outperformance": 2.5,
    "underperform": -2.0, "underperformed": -2.0, "underperforming": -2.0,
    # --- analyst actions ---
    "upgrade": 2.5, "upgrades": 2.5, "upgraded": 2.5, "upgrading": 2.0,
    "downgrade": -2.5, "downgrades": -2.5, "downgraded": -2.5, "downgrading": -2.0,
    "raised": 1.5, "raises": 1.2, "lowered": -1.5, "lowers": -1.2, "cutting": -1.0,
    "neutral": 0.0, "overweight": 1.0, "underweight": -1.0,
    "buyback": 1.8, "buybacks": 1.8,
    # --- price action ---
    "surge": 3.0, "surges": 3.0, "surged": 3.0, "soar": 3.0, "soars": 3.0, "soared": 3.0,
    "rally": 2.5, "rallies": 2.5, "rallied": 2.5, "rebound": 2.0, "rebounds": 2.0,
    "rebounded": 2.0, "recovered": 1.8, "recovery": 1.5, "climb": 1.2, "climbed": 1.2,
    "jump": 1.5, "jumps": 1.5, "jumped": 1.5, "gains": 1.5, "gained": 1.5,
    "plunge": -3.0, "plunged": -3.0, "plunges": -3.0, "crash": -3.0, "crashed": -3.0,
    "crashes": -3.0, "tank": -3.0, "tanked": -3.0, "tanks": -3.0, "tumble": -2.5,
    "tumbled": -2.5, "slump": -2.5, "slumped": -2.5, "slide": -1.5, "slid": -1.5,
    "selloff": -2.0, "dive": -2.0, "dived": -2.0, "drop": -1.5, "drops": -1.5,
    "dropped": -1.5, "decline": -1.5, "declines": -1.5, "declined": -1.5,
    "declining": -1.5, "fell": -1.5, "fall": -1.2, "falls": -1.5, "fallen": -1.8,
    # --- company fundamentals ---
    "profit": 1.8, "profits": 1.8, "profitable": 2.0, "profitability": 1.8,
    "loss": -1.8, "losses": -1.8, "lost": -1.5, "debt": -1.0, "deficit": -1.5,
    "dividend": 1.2, "dividends": 1.2, "growth": 1.2, "record": 1.2,
    "strong": 1.5, "stronger": 1.5, "strongest": 1.8, "robust": 1.5, "solid": 1.5,
    "weak": -1.5, "weaker": -1.8, "weakest": -1.8, "weakness": -1.5,
    "deteriorate": -2.0, "deteriorating": -2.0, "deterioration": -2.0,
    "sluggish": -1.8, "stagnant": -1.5, "stagnation": -1.5, "tepid": -1.5,
    "layoffs": -2.5, "layoff": -2.5, "firing": -2.0, "fired": -2.0,
    "resign": -1.2, "resigns": -1.2, "resigned": -1.2, "resignation": -1.2,
    "ousted": -2.5, "oust": -2.0, "restructuring": -1.0, "restructures": -1.0,
    # --- regulatory / legal / conduct ---
    "lawsuit": -2.0, "lawsuits": -2.0, "sued": -2.5, "sues": -2.5,
    "fraud": -3.5, "scandal": -3.0, "probe": -1.5, "probes": -1.5,
    "investigation": -1.5, "investigated": -1.5, "subpoena": -2.5, "subpoenas": -2.5,
    "bankrupt": -4.0, "bankruptcy": -4.0, "insolvent": -3.5, "insolvency": -3.5,
    "default": -3.0, "defaults": -3.0, "defaulted": -3.0,
    "recall": -2.0, "recalled": -2.0, "recalls": -2.0, "defect": -2.0,
    "settlement": -1.0, "fine": -1.0, "fines": -1.2, "fined": -1.2,
    "warning": -1.5, "warns": -1.5, "warned": -1.5, "cautions": -1.2,
    "sanction": -2.0, "sanctions": -2.5,
    # --- outlook / macro tone ---
    "bullish": 2.0, "bearish": -2.0, "optimistic": 1.8, "optimism": 1.8,
    "pessimistic": -1.8, "pessimism": -1.8, "confident": 1.5, "confidence": 1.2,
    "upbeat": 1.8, "momentum": 1.0, "tailwind": 2.0, "headwind": -2.0,
    "turmoil": -2.5, "volatility": -0.8, "uncertainty": -1.2, "uncertain": -1.2,
    "risk": -1.2, "risks": -1.2, "risky": -1.5, "concern": -1.0, "concerns": -1.0,
    "pressure": -0.8, "pressured": -1.2, "pressures": -1.0,
    "overvalued": -1.5, "undervalued": 1.5, "bubble": -2.0,
    "breakthrough": 2.5, "milestone": 1.5, "approval": 2.0, "approved": 2.0,
    "approves": 2.0, "launch": 1.0, "launched": 1.0, "launches": 1.0,
    "expansion": 1.2, "expand": 1.2, "expands": 1.2, "expanded": 1.2,
    "partnership": 1.2, "partner": 1.0, "partners": 1.0, "collaboration": 1.0,
}

