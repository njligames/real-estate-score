import sys
import requests
import pandas as pd
import numpy as np
import os
import json
import time
from typing import Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI

# =========================================================
# FRED CLIENT WITH RETRIES
# =========================================================
class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str, retries: int = 3, timeout: int = 30):
        if not api_key:
            raise ValueError("FRED API key required")
        self.api_key = api_key
        self.timeout = timeout

        # Configure retry logic: 429=Too Many Requests, 5xx=Server Errors
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        ))
        self.session.mount("https://", adapter)

    def get_series(self, series_id: str, start: str = "2000-01-01") -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
        }
        r = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()["observations"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].dropna()

# =========================================================
# AIRDNA CLIENT
# =========================================================
class AirDNAClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, url: str, params: dict):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_occupancy(self, market: str) -> pd.Series:
        data = self._get("https://api.airdna.co/v1/occupancy", {"market": market})
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["occupancy"].astype(float)

    def get_revenue(self, market: str) -> pd.Series:
        data = self._get("https://api.airdna.co/v1/revenue", {"market": market})
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["revenue"].astype(float)

    def get_listings(self, market: str) -> pd.Series:
        data = self._get("https://api.airdna.co/v1/listings", {"market": market})
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["listings"].astype(float)

# =========================================================
# SCORING LOGIC
# =========================================================
def rolling_trend(series: pd.Series, window: int = 6): return series.diff(window)
def zscore(series: pd.Series, window: int = 12): return (series - series.rolling(window).mean()) / series.rolling(window).std()
def normalize_tanh(x: pd.Series): return np.tanh(x)

class MacroScorer:
    @staticmethod
    def rate_score(mortgage_rates: pd.Series): return normalize_tanh(-rolling_trend(mortgage_rates, 6))
    @staticmethod
    def supply_score(housing_supply: pd.Series): return normalize_tanh(rolling_trend(housing_supply, 6))
    @staticmethod
    def recession_score(spread: pd.Series, recession: pd.Series):
        common_idx = spread.index.intersection(recession.index)
        score = pd.Series(index=common_idx, dtype=float)
        score[spread.loc[common_idx] < 0] = 0.6
        score[recession.loc[common_idx] == 1] = 1.0
        score[spread.loc[common_idx] >= 0] = -0.4
        return score.fillna(0.0)
    @staticmethod
    def consumer_score(sentiment: pd.Series): return normalize_tanh(-zscore(sentiment, 12))

class STRScorer:
    @staticmethod
    def compute(occ, rev, listings):
        df = pd.concat([occ, rev, listings], axis=1).dropna()
        score = (np.tanh((df.iloc[:,0] - 0.6) * 3) * 0.4 +
                 np.tanh(df.iloc[:,1].pct_change(12) * 5) * 0.4 +
                 (-np.tanh(df.iloc[:,2].pct_change(12) * 5)) * 0.2)
        return score.clip(-1, 1)

# =========================================================
# ENGINE
# =========================================================
class RealEstateEngine:
    def __init__(self, fred_api_key: Optional[str] = None, airdna_api_key: Optional[str] = None, retries: int = 3, timeout: int = 30):
        self.fred = FredClient(fred_api_key, retries=retries, timeout=timeout) if fred_api_key else None
        self.airdna = AirDNAClient(airdna_api_key) if airdna_api_key else None

    def _as_of(self, series: pd.Series):
        return float(series.dropna().iloc[-1])

    def run_fred(self):
        mortgage = self.fred.get_series("MORTGAGE30US")
        permits = self.fred.get_series("PERMIT")
        spread = self.fred.get_series("T10Y2Y")
        recession = self.fred.get_series("USREC")
        sentiment = self.fred.get_series("UMCSENT")

        # Get the latest mortgage rate
        latest_mortgage_rate = float(mortgage.dropna().iloc[-1])

        df = pd.concat([mortgage, permits, spread, recession, sentiment], axis=1, sort=False).dropna()
        df.columns = ["mortgage", "permits", "spread", "recession", "sentiment"]

        score = (MacroScorer.rate_score(df["mortgage"]) * 0.3 +
                 MacroScorer.supply_score(df["permits"]) * 0.25 +
                 MacroScorer.recession_score(df["spread"], df["recession"]) * 0.25 +
                 MacroScorer.consumer_score(df["sentiment"]) * 0.2).clip(-1, 1)

        # Return both the score and the rate
        return self._as_of(score), latest_mortgage_rate

# =========================================================
# TELEGRAM & MAIN
# =========================================================
def send_telegram_message(message_body):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message_body, "parse_mode": "HTML"})

if __name__ == "__main__":
    load_dotenv()

    engine = RealEstateEngine(
        fred_api_key=os.getenv("FRED_API_KEY"),
        retries=5,
        timeout=45
    )

    try:
        # Unpack the two return values
        score_val, current_rate = engine.run_fred()

        msg = f"""
<b>Daily Real Estate Update</b>
<b>Score:</b> {round(score_val, 4)}
<b>30-Year Mortgage Rate:</b> {current_rate}%

Meaning: <a href="https://github.com/njligames/mortgage_dashboard/tree/main/real_estate_score#interpretation-of-the-score">Score Interpretation</a>
Dashboard: <a href="https://mortgage-dashboard-p3xc.onrender.com/">View Dashboard</a>
        """
        send_telegram_message(msg)
        print(f"Success! Score: {score_val}, Rate: {current_rate}")

    except Exception as e:
        print(f"Workflow failed after retries: {e}")
        sys.exit(1)
