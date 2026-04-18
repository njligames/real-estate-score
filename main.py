import requests
import pandas as pd
import numpy as np
from typing import Optional
import os
import requests
from twilio.rest import Client
from dotenv import load_dotenv
import json
from openai import OpenAI


# =========================================================
# FRED CLIENT
# =========================================================

class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("FRED API key required")
        self.api_key = api_key

    def get_series(self, series_id: str, start: str = "2000-01-01") -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
        }

        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()

        data = r.json()["observations"]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df.set_index("date")["value"].dropna()


# =========================================================
# AIRDNA CLIENT (PLACEHOLDER IMPLEMENTATION)
# =========================================================

class AirDNAClient:
    """
    Requires valid AirDNA API access.
    Endpoints below are placeholders and must be mapped
    to your actual AirDNA contract.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AirDNA API key required")
        self.api_key = api_key

    def _get(self, url: str, params: dict):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_occupancy(self, market: str) -> pd.Series:
        data = self._get(
            "https://api.airdna.co/v1/occupancy",
            {"market": market},
        )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["occupancy"].astype(float)

    def get_revenue(self, market: str) -> pd.Series:
        data = self._get(
            "https://api.airdna.co/v1/revenue",
            {"market": market},
        )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["revenue"].astype(float)

    def get_listings(self, market: str) -> pd.Series:
        data = self._get(
            "https://api.airdna.co/v1/listings",
            {"market": market},
        )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["listings"].astype(float)


# =========================================================
# SIGNAL UTILITIES
# =========================================================

def rolling_trend(series: pd.Series, window: int = 6):
    return series.diff(window)


def zscore(series: pd.Series, window: int = 12):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def normalize_tanh(x: pd.Series):
    return np.tanh(x)


# =========================================================
# FRED-BASED MACRO SCORING
# =========================================================

class MacroScorer:
    @staticmethod
    def rate_score(mortgage_rates: pd.Series):
        trend = rolling_trend(mortgage_rates, 6)
        return normalize_tanh(-trend)

    @staticmethod
    def supply_score(housing_supply: pd.Series):
        trend = rolling_trend(housing_supply, 6)
        return normalize_tanh(trend)

    @staticmethod
    def recession_score(spread: pd.Series, recession: pd.Series):
        # Align the two series by their index intersection
        common_idx = spread.index.intersection(recession.index)
        spread = spread.loc[common_idx]
        recession = recession.loc[common_idx]

        score = pd.Series(index=common_idx, dtype=float)

        score[spread < 0] = 0.6
        score[recession == 1] = 1.0
        score[spread >= 0] = -0.4

        return score.fillna(0.0)

    @staticmethod
    def consumer_score(sentiment: pd.Series):
        z = zscore(sentiment, 12)
        return normalize_tanh(-z)


# =========================================================
# STR SCORING (AIRDNA)
# =========================================================

class STRScorer:
    @staticmethod
    def occupancy_score(x: pd.Series):
        return np.tanh((x - 0.6) * 3)

    @staticmethod
    def revenue_score(x: pd.Series):
        return np.tanh(x.pct_change(12) * 5)

    @staticmethod
    def supply_score(x: pd.Series):
        return -np.tanh(x.pct_change(12) * 5)

    @staticmethod
    def compute(occ, rev, listings):
        df = pd.concat([occ, rev, listings], axis=1).dropna()
        df.columns = ["occ", "rev", "list"]

        score = (
            STRScorer.occupancy_score(df["occ"]) * 0.4 +
            STRScorer.revenue_score(df["rev"]) * 0.4 +
            STRScorer.supply_score(df["list"]) * 0.2
        )

        return score.clip(-1, 1)


# =========================================================
# MAIN ENGINE
# =========================================================

class RealEstateEngine:
    """
    Three modes:
    - run_fred()
    - run_airdna()
    - run_combined()
    """

    def __init__(self, fred_api_key: Optional[str] = None, airdna_api_key: Optional[str] = None):
        self.fred = FredClient(fred_api_key) if fred_api_key else None
        self.airdna = AirDNAClient(airdna_api_key) if airdna_api_key else None

    # -----------------------------------------------------
    # helper: align to as_of date
    # -----------------------------------------------------
    def _as_of(self, series: pd.Series, as_of: Optional[str]):
        series = series.dropna()

        if as_of is None:
            return float(series.iloc[-1])

        as_of = pd.to_datetime(as_of)
        valid = series.index[series.index <= as_of]

        if len(valid) == 0:
            raise ValueError("No data available before as_of date")

        return float(series.loc[valid[-1]])

    # =====================================================
    # 1. FRED ONLY
    # =====================================================
    def run_fred(self, as_of: Optional[str] = None):
        # ... inside run_fred ...
        mortgage = self.fred.get_series("MORTGAGE30US")
        permits = self.fred.get_series("PERMIT")
        spread = self.fred.get_series("T10Y2Y")
        recession = self.fred.get_series("USREC")
        sentiment = self.fred.get_series("UMCSENT")

        # Join them all first to ensure perfect index alignment
        df_raw = pd.concat(
            [mortgage, permits, spread, recession, sentiment],
            axis=1,
            sort=False
        ).sort_index()
        df_raw.columns = ["mortgage", "permits", "spread", "recession", "sentiment"]
        df_raw = df_raw.dropna()

        # Pass the aligned columns to your scorers
        rate = MacroScorer.rate_score(df_raw["mortgage"])
        supply = MacroScorer.supply_score(df_raw["permits"])
        rec = MacroScorer.recession_score(df_raw["spread"], df_raw["recession"])
        cons = MacroScorer.consumer_score(df_raw["sentiment"])

        df = pd.concat([rate, supply, rec, cons], axis=1).dropna()
        df.columns = ["rate", "supply", "recession", "consumer"]

        score = (
            df["rate"] * 0.3 +
            df["supply"] * 0.25 +
            df["recession"] * 0.25 +
            df["consumer"] * 0.2
        ).clip(-1, 1)

        return self._as_of(score, as_of)

    # =====================================================
    # 2. AIRDNA ONLY
    # =====================================================
    def run_airdna(self, market: str, as_of: Optional[str] = None):
        if not self.airdna:
            raise ValueError("AirDNA API key not provided")

        occ = self.airdna.get_occupancy(market)
        rev = self.airdna.get_revenue(market)
        lst = self.airdna.get_listings(market)

        score = STRScorer.compute(occ, rev, lst)

        return self._as_of(score, as_of)

    # =====================================================
    # 3. COMBINED
    # =====================================================
    def run_combined(self, market: str, as_of: Optional[str] = None):
        if not self.fred or not self.airdna:
            raise ValueError("Both FRED and AirDNA API keys are required")

        fred_score = self.run_fred(as_of=None)
        occ = self.airdna.get_occupancy(market)
        rev = self.airdna.get_revenue(market)
        lst = self.airdna.get_listings(market)

        str_score_series = STRScorer.compute(occ, rev, lst)

        # align scalar fred score across STR index for combination
        fred_series = pd.Series(fred_score, index=str_score_series.index)

        combined = (fred_series * 0.6 + str_score_series * 0.4).clip(-1, 1)

        return self._as_of(combined, as_of)


def interpret_score(score: float) -> dict:
    """
    Converts normalized buy score [-1, 1] into actionable regime + explicit definitions.
    """

    if score is None:
        raise ValueError("Score cannot be None")

    if score < -1 or score > 1:
        raise ValueError("Score must be in [-1, 1]")

    # -------------------------
    # Regime mapping
    # -------------------------
    if score >= 0.6:
        regime = "AGGRESSIVE_BUY"
        action = "Strong macro + STR alignment"
    elif score >= 0.2:
        regime = "SELECTIVE_BUY"
        action = "Positive conditions, require discipline"
    elif score >= -0.2:
        regime = "NEUTRAL"
        action = "No clear edge"
    elif score >= -0.6:
        regime = "DEFENSIVE"
        action = "Weak conditions, reduce risk"
    else:
        regime = "AVOID"
        action = "Adverse conditions"

    # -------------------------
    # Confidence (explicit meaning included)
    # -------------------------
    confidence = round(abs(score), 3)

    if confidence >= 0.6:
        confidence_level = "HIGH"
        confidence_meaning = "Strong deviation from neutral conditions"
    elif confidence >= 0.3:
        confidence_level = "MEDIUM"
        confidence_meaning = "Moderate directional signal"
    else:
        confidence_level = "LOW"
        confidence_meaning = "Weak or unclear signal (near neutral)"

    # -------------------------
    # Leverage posture (explicit meaning included)
    # -------------------------
    if score >= 0.6:
        leverage_posture = "NORMAL_TO_EXPANSIVE"
        leverage_meaning = "Capital can be deployed more aggressively due to favorable conditions"
    elif score >= 0.2:
        leverage_posture = "NORMAL"
        leverage_meaning = "Standard leverage assumptions appropriate"
    elif score >= -0.2:
        leverage_posture = "REDUCED"
        leverage_meaning = "Reduce leverage and tighten underwriting"
    else:
        leverage_posture = "MINIMAL"
        leverage_meaning = "Preserve capital; avoid leverage-heavy positions"

    return {
        "score": round(score, 4),
        "regime": regime,
        "action": action,

        "confidence": confidence,
        "confidence_level": confidence_level,
        "confidence_meaning": confidence_meaning,

        "leverage_posture": leverage_posture,
        "leverage_meaning": leverage_meaning,
    }

def send_sms(msg):
    # Retrieve credentials from Environment Variables (set these in Render)
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    from_number = os.environ['TWILIO_PHONE_NUMBER']
    to_number = os.environ['MY_PERSONAL_NUMBER'] # Your phone number

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=msg,
        from_=from_number,
        to=to_number
    )

    print(f"Message sent successfully! SID: {message.sid}")


def send_telegram_message(message_body):
    # Retrieve these from your Render Environment Variables
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_body,
        "parse_mode": "HTML" # Allows basic formatting like bold/italics
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        print("Telegram message sent!")
    else:
        print(f"Failed to send: {response.text}")


def create_openai_message(score):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    prompt = f"""
Please interpret the score based on how the algorithm works and compose a message. PLease a TLDO, first.

this is the score:

    {score}

Here is the definition:

How the Algorithm Works
1. Macro Model (FRED)
The macro score is composed of four signals:

A. Interest Rate Trend (30%)
Uses mortgage rate trend (6-month change)
Falling rates → positive signal
Rising rates → negative signal
Why it matters: Rates drive affordability → affordability drives demand → demand drives price.

B. Housing Supply (25%)
Uses housing permits (proxy for future supply)
Increasing supply → buyer advantage
Tight supply → seller advantage
Why it matters: Supply determines negotiation leverage and future price pressure.

C. Recession Signal (25%)
Based on:

Yield curve inversion
Official recession indicator
Why it matters: The best buying windows historically occur:

Early recession
Peak uncertainty
D. Consumer Sentiment (20%)
Based on sentiment z-score
Low sentiment → good buying conditions
Why it matters: Real estate is driven by psychology:

Fear → fewer buyers → better deals
Confidence → bidding wars → overpaying
Macro Score Formula
Macro Score =
  (Rates * 0.3) +
  (Supply * 0.25) +
  (Recession * 0.25) +
  (Consumer * 0.2)
2. STR Market Model (AirDNA)
A. Occupancy (40%)
Measures demand consistency
Higher occupancy → stronger cash flow stability
B. Revenue Growth (40%)
Year-over-year growth
Captures demand + pricing power
C. Supply Growth (20%)
Growth in listings
Higher supply → saturation risk
STR Score Formula
STR Score =
  (Occupancy * 0.4) +
  (Revenue Growth * 0.4) +
  (Supply Pressure * 0.2)
3. Combined Model
Final Score =
  (Macro Score * 0.6) +
  (STR Score * 0.4)
Why These Weights Were Chosen
Macro Weights
Component	Weight	Rationale
Rates	30%	Strongest direct driver of housing demand
Supply	25%	Determines pricing pressure and negotiation
Recession	25%	Captures timing asymmetry (best entry windows)
Consumer	20%	Behavioral confirmation signal
Key idea: Rates are slightly dominant because they affect all buyers immediately.

STR Weights
Component	Weight	Rationale
Occupancy	40%	Stability of income
Revenue Growth	40%	Upside potential
Supply	20%	Risk control (saturation)
Key idea: Revenue + occupancy drive returns; supply acts as a constraint.

Combined Weights
Layer	Weight	Rationale
Macro	60%	Timing errors are more expensive than location errors
STR	40%	Market selection still critical
Key idea: A bad macro environment can break even a strong STR market.

How Changing Weights Affects Outcomes
Increasing Rate Weight (e.g., 30% → 50%)
Model becomes more sensitive to interest rate changes
More reactive to Fed policy
May trigger earlier buy signals during rate drops
Tradeoff: Can overreact to short-term rate volatility

Increasing Recession Weight
More aggressive buying during downturns
Better at catching bottoms
Risk: entering too early in prolonged downturns
Increasing STR Weight (Combined Model)
More emphasis on local performance

Better for:

Experienced operators
Market specialists
Tradeoff: Can ignore macro risks (e.g., rising rates crushing returns)

Reducing Supply Weight
Model becomes less sensitive to saturation
May overestimate market strength in overheated areas
Increasing Supply Weight
More conservative
Avoids crowded STR markets
Tradeoff: May miss high-demand growth markets early

Interpretation of the Score
Score Range	Meaning
0.6 to 1.0	Strong buying conditions
0.2 to 0.6	Favorable
-0.2 to 0.2	Neutral
-0.6 to -0.2	Defensive
-1.0 to -0.6	Avoid
Confidence
Confidence is defined as:

confidence = abs(score)
Meaning:
Measures distance from neutrality
NOT statistical confidence
Indicates strength of signal, not accuracy
Leverage Posture
Derived from score:

Range	Posture	Meaning
High positive	Expansive	Can take more risk
Moderate	Normal	Standard underwriting
Neutral	Reduced	Tighten assumptions
Negative	Minimal	Preserve capital
Key Limitations
No property-level underwriting
No regulatory/zoning data
AirDNA API access required for STR layer
FRED data is lagging (not real-time)
Suggested Improvements
Backtest against:

Home Price Index
STR revenue trends
Add:

Seasonality modeling
Local regulation risk
Interest rate forecasts
Introduce:

Nonlinear scoring
Regime persistence (avoid rapid flipping)
Bottom Line
This system provides a structured way to answer:

When to invest (macro)
Where to invest (market)
How aggressively to invest (score interpretation)
It is not a replacement for deal analysis—it is a filter for when conditions are in your favor.
"""

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        return response.output[0].content[0].text
    except Exception as e:
        return score

# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    AIRDNA_API_KEY = os.getenv("AIRDNA_API_KEY")

    engine = RealEstateEngine(
        fred_api_key=FRED_API_KEY,
        airdna_api_key=AIRDNA_API_KEY
    )
    _s = engine.run_fred()
    score = interpret_score(_s)
    score['meaning'] = "https://github.com/njligames/mortgage_dashboard/tree/main/real_estate_score#interpretation-of-the-score"
    json_string = json.dumps(score, indent=4)

    # send_sms(json_string)
    # send_telegram_message(create_openai_message(json_string))
    # send_telegram_message(json_string)
    msg = f"""
Score: {_s}
Meaning: "https://github.com/njligames/mortgage_dashboard/tree/main/real_estate_score#interpretation-of-the-score"
    """
    send_telegram_message(msg)
    print(json_string)

    # print(engine.run_airdna("Miami"))
    # print(engine.run_combined("Miami"))
