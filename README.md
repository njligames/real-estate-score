## README — Real Estate Buy Signal Engine

## Environment Setup

### 1. Create a Virtual Environment

From the root of your project:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
venv\Scripts\activate
```

---

### 3. Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

---

### 4. Install Dependencies

Create a `requirements.txt` file (see below), then run:

```bash
pip install -r requirements.txt
```

---

### 5. Verify Installation

```bash
python -c "import pandas, numpy, requests; print('OK')"
```

---

### 6. Deactivate Environment (when done)

```bash
deactivate
```

---

## requirements.txt

```txt
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
```

---

## Notes

* No heavy dependencies are required; this keeps the environment lightweight and fast.
* If you later add:

  * backtesting → consider `scipy`, `statsmodels`
  * API server → add `fastapi`, `uvicorn`
  * storage → add `sqlalchemy`, `psycopg2`

---

This setup ensures:

* isolation from system Python
* reproducible builds
* compatibility across environments


### Overview

This project provides a **data-driven framework for timing short-term rental (STR) real estate investments** using two layers:

1. **Macro timing (FRED)** — determines *when* to deploy capital
2. **Market performance (AirDNA)** — determines *where* to deploy capital

The system outputs a normalized **Buy Score ∈ [-1, 1]** and an interpretation layer that translates the score into actionable investment regimes.

---

# Why This Matters

Most real estate investors fail in one of two ways:

* They buy **good properties at bad macro times** (e.g., peak rates, tightening liquidity)
* They buy **in bad markets during good macro conditions**

This model separates those concerns:

| Layer        | Question Answered                            |
| ------------ | -------------------------------------------- |
| Macro (FRED) | Is this a good time to buy?                  |
| STR (AirDNA) | Is this a good market to buy in?             |
| Combined     | Is this a high-probability investment setup? |

---

# How the Algorithm Works

## 1. Macro Model (FRED)

The macro score is composed of four signals:

### A. Interest Rate Trend (30%)

* Uses mortgage rate trend (6-month change)
* Falling rates → **positive signal**
* Rising rates → **negative signal**

**Why it matters:**
Rates drive affordability → affordability drives demand → demand drives price.

---

### B. Housing Supply (25%)

* Uses housing permits (proxy for future supply)
* Increasing supply → **buyer advantage**
* Tight supply → **seller advantage**

**Why it matters:**
Supply determines negotiation leverage and future price pressure.

---

### C. Recession Signal (25%)

* Based on:

  * Yield curve inversion
  * Official recession indicator

**Why it matters:**
The best buying windows historically occur:

* **Early recession**
* **Peak uncertainty**

---

### D. Consumer Sentiment (20%)

* Based on sentiment z-score
* Low sentiment → **good buying conditions**

**Why it matters:**
Real estate is driven by psychology:

* Fear → fewer buyers → better deals
* Confidence → bidding wars → overpaying

---

## Macro Score Formula

```
Macro Score =
  (Rates * 0.3) +
  (Supply * 0.25) +
  (Recession * 0.25) +
  (Consumer * 0.2)
```

---

## 2. STR Market Model (AirDNA)

### A. Occupancy (40%)

* Measures demand consistency
* Higher occupancy → stronger cash flow stability

---

### B. Revenue Growth (40%)

* Year-over-year growth
* Captures demand + pricing power

---

### C. Supply Growth (20%)

* Growth in listings
* Higher supply → saturation risk

---

## STR Score Formula

```
STR Score =
  (Occupancy * 0.4) +
  (Revenue Growth * 0.4) +
  (Supply Pressure * 0.2)
```

---

## 3. Combined Model

```
Final Score =
  (Macro Score * 0.6) +
  (STR Score * 0.4)
```

---

# Why These Weights Were Chosen

## Macro Weights

| Component | Weight | Rationale                                      |
| --------- | ------ | ---------------------------------------------- |
| Rates     | 30%    | Strongest direct driver of housing demand      |
| Supply    | 25%    | Determines pricing pressure and negotiation    |
| Recession | 25%    | Captures timing asymmetry (best entry windows) |
| Consumer  | 20%    | Behavioral confirmation signal                 |

**Key idea:**
Rates are slightly dominant because they affect *all buyers immediately*.

---

## STR Weights

| Component      | Weight | Rationale                 |
| -------------- | ------ | ------------------------- |
| Occupancy      | 40%    | Stability of income       |
| Revenue Growth | 40%    | Upside potential          |
| Supply         | 20%    | Risk control (saturation) |

**Key idea:**
Revenue + occupancy drive returns; supply acts as a constraint.

---

## Combined Weights

| Layer | Weight | Rationale                                             |
| ----- | ------ | ----------------------------------------------------- |
| Macro | 60%    | Timing errors are more expensive than location errors |
| STR   | 40%    | Market selection still critical                       |

**Key idea:**
A bad macro environment can break even a strong STR market.

---

# How Changing Weights Affects Outcomes

## Increasing Rate Weight (e.g., 30% → 50%)

* Model becomes **more sensitive to interest rate changes**
* More reactive to Fed policy
* May trigger earlier buy signals during rate drops

**Tradeoff:**
Can overreact to short-term rate volatility

---

## Increasing Recession Weight

* More aggressive buying during downturns
* Better at catching **bottoms**
* Risk: entering too early in prolonged downturns

---

## Increasing STR Weight (Combined Model)

* More emphasis on local performance
* Better for:

  * Experienced operators
  * Market specialists

**Tradeoff:**
Can ignore macro risks (e.g., rising rates crushing returns)

---

## Reducing Supply Weight

* Model becomes less sensitive to saturation
* May overestimate market strength in overheated areas

---

## Increasing Supply Weight

* More conservative
* Avoids crowded STR markets

**Tradeoff:**
May miss high-demand growth markets early

---

# Interpretation of the Score

| Score Range  | Meaning                  |
| ------------ | ------------------------ |
| 0.6 to 1.0   | Strong buying conditions |
| 0.2 to 0.6   | Favorable                |
| -0.2 to 0.2  | Neutral                  |
| -0.6 to -0.2 | Defensive                |
| -1.0 to -0.6 | Avoid                    |

---

# Confidence

Confidence is defined as:

```
confidence = abs(score)
```

### Meaning:

* Measures **distance from neutrality**
* NOT statistical confidence
* Indicates strength of signal, not accuracy

---

# Leverage Posture

Derived from score:

| Range         | Posture   | Meaning               |
| ------------- | --------- | --------------------- |
| High positive | Expansive | Can take more risk    |
| Moderate      | Normal    | Standard underwriting |
| Neutral       | Reduced   | Tighten assumptions   |
| Negative      | Minimal   | Preserve capital      |

---

# Key Limitations

* No property-level underwriting
* No regulatory/zoning data
* AirDNA API access required for STR layer
* FRED data is lagging (not real-time)

---

# Suggested Improvements

* Backtest against:

  * Home Price Index
  * STR revenue trends
* Add:

  * Seasonality modeling
  * Local regulation risk
  * Interest rate forecasts
* Introduce:

  * Nonlinear scoring
  * Regime persistence (avoid rapid flipping)

---

# Bottom Line

This system provides a structured way to answer:

* **When to invest (macro)**
* **Where to invest (market)**
* **How aggressively to invest (score interpretation)**

It is not a replacement for deal analysis—it is a **filter for when conditions are in your favor**.
