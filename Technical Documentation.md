# Technical Documentation

## Manufacturing Talent Supply Chain Risk Analysis

**Version**: 1.0  
**Last Updated**: February 2024  
**Author**: Sean Kim

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Pipeline](#2-data-pipeline)
3. [Layer 1: Aging Risk Model](#3-layer-1-aging-risk-model)
4. [Layer 2: Attrition Risk Model](#4-layer-2-attrition-risk-model)
5. [Layer 3: Hiring Capacity Model](#5-layer-3-hiring-capacity-model)
6. [Layer 4: Training Capacity Model](#6-layer-4-training-capacity-model)
7. [Layer 5: External Competition Model](#7-layer-5-external-competition-model)
8. [Statistical Methods](#8-statistical-methods)
9. [Code Reference](#9-code-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Architecture

### 1.1 Overview

The analysis system follows a modular pipeline architecture:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Data   │ -> │  Cleansing  │ -> │   Analysis  │ -> │   Output    │
│  Ingestion  │    │  & Filtering│    │   Layers    │    │   Reports   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ↑                   ↑                  ↑                  ↑
  Census J2J        Energy Belt       5-Layer Model      PNG, DOCX,
  IPEDS, BLS        5-State Filter    Statistical        Summary TXT
```

### 1.2 File Dependencies

```
final_portfolio.py
├── pandas (data manipulation)
├── numpy (numerical computation)
├── scipy.stats (statistical tests)
├── matplotlib (visualization)
└── pathlib (file management)
```

### 1.3 Configuration

All configuration is centralized at the top of `final_portfolio.py`:

```python
# File Paths
J2J_FILE   = "/path/to/j2j_census.csv"
IPEDS_FILE = "/path/to/ipeds_completions.csv"
OUTPUT_DIR = "/path/to/output"

# Color Palette (consistent across all layers)
COLORS = {
    'Manufacturing': '#d62728',  # Red (crisis indicator)
    'Logistics':     '#ff7f0e',  # Orange
    'Construction':  '#8c564b',  # Brown
    'Retail':        '#9467bd',  # Purple
    'Hospitality':   '#e377c2',  # Pink
    'Services':      '#bcbd22',  # Olive
    'Unemployment':  '#7f7f7f',  # Gray
    'Trend':         '#95a5a6',  # Light Gray
    'Demand':        '#f39c12',  # Orange
    'Supply':        '#bdc3c7'   # Silver
}
```

---

## 2. Data Pipeline

### 2.1 Census J2J (Job-to-Job Flows)

#### Source
- **Provider**: U.S. Census Bureau, LEHD Program
- **URL**: https://lehd.ces.census.gov/data/#j2j
- **Format**: CSV (gzip compressed)

#### Schema

| Column | Type | Description |
|--------|------|-------------|
| `geography` | str | State FIPS code (e.g., '13' = Georgia) |
| `industry` | str | NAICS sector (e.g., '31-33' = Manufacturing) |
| `agegrp` | str | Age group code (A00=All, A04=25-34, A07=55-64, A08=65+) |
| `sex` | str | Gender (0=All, 1=Male, 2=Female) |
| `year` | int | Calendar year |
| `quarter` | int | Calendar quarter (1-4) |
| `seasonadj` | str | Seasonal adjustment (U=Unadjusted, S=Adjusted) |
| `J2J` | float | Job-to-job flow count |

#### Filtering Logic

```python
# Energy Belt 5 States
ENERGY_BELT_FIPS = ['1', '13', '37', '45', '47']
# 01 = Alabama
# 13 = Georgia
# 37 = North Carolina
# 45 = South Carolina
# 47 = Tennessee

# Filter mask
mask = (
    (df['geography'].isin(ENERGY_BELT_FIPS)) &   # 5 states
    (df['seasonadj'] == 'U') &                    # Unadjusted
    (df['sex'] == '0') &                          # All genders
    (df['industry'].str.startswith(('31','32','33'), na=False)) &  # Manufacturing
    (df['agegrp'] != 'A00')                       # Detailed age groups
)
```

### 2.2 IPEDS (Education Statistics)

#### Source
- **Provider**: National Center for Education Statistics
- **URL**: https://nces.ed.gov/ipeds/datacenter
- **Format**: CSV

#### Schema

| Column | Type | Description |
|--------|------|-------------|
| `UNITID` | int | Institution identifier |
| `CIPCODE` | str | Classification of Instructional Programs code |
| `AWLEVEL` | int | Award level (3=Associate, 5=Bachelor, etc.) |
| `CTOTALT` | int | Total completions |
| `YEAR` | int | Academic year |

#### CIP Code Mapping

| CIP Prefix | Field | Example Occupations |
|------------|-------|---------------------|
| 14 | Engineering | Mechanical, Electrical, Industrial Engineers |
| 15 | Engineering Technology | Engineering Technicians |
| 47 | Mechanic & Repair | Industrial Mechanics, HVAC |
| 48 | Precision Production | Welders, Machinists, CNC Operators |

#### Filtering Logic

```python
ipeds = pd.read_csv(IPEDS_FILE, dtype={"CIPCODE": str})
ipeds["CIP2"] = ipeds["CIPCODE"].str[:2]
target = ipeds[ipeds["CIP2"].isin(["14", "15", "47", "48"])]
```

### 2.3 BLS (Labor Statistics)

#### Sources Used

| Table | Purpose | Key Metrics |
|-------|---------|-------------|
| Table 1.10 | Exit rates by age | λ = 4.4%/yr (manufacturing avg) |
| JOLTS | Job openings | 142,800 openings (5-state) |
| OES | Occupation distribution | Tech share = 25% |

---

## 3. Layer 1: Aging Risk Model

### 3.1 Theoretical Foundation

Workforce depletion follows **exponential decay**, analogous to radioactive half-life:

```
N(t) = N₀ × e^(-λ × t)
```

Where:
- `N(t)` = Workforce at time t
- `N₀` = Initial workforce (index = 100)
- `λ` = Annual exit rate (decay constant)
- `t` = Time in years

### 3.2 Half-Life Derivation

```
At half-life: N(t½) = N₀/2
N₀/2 = N₀ × e^(-λ × t½)
0.5 = e^(-λ × t½)
ln(0.5) = -λ × t½
t½ = ln(2) / λ = 0.693 / λ
```

### 3.3 Parameter Values

| Scenario | λ (exit rate) | Half-Life | Source |
|----------|---------------|-----------|--------|
| Baseline | 0.044 (4.4%) | 15.8 years | BLS Table 1.10 |
| Accelerated | 0.066 (6.6%) | 10.5 years | Baseline × 1.5 |

### 3.4 Accelerated Scenario Justification

The 1.5× multiplier reflects:
- Baby Boomers (born 1946-1964) reaching 65+ in 2024-2030
- BLS data shows 65+ exit rate is 3.5× higher than 55-64
- Manufacturing has higher physical labor demands

### 3.5 Implementation

```python
def plot_layer1_aging():
    years = np.arange(2024, 2036)
    t = years - 2024
    
    lambda_base = 0.044
    lambda_accel = 0.066
    
    retention_base = 100 * np.exp(-lambda_base * t)
    retention_accel = 100 * np.exp(-lambda_accel * t)
    
    # Critical point (50% remaining)
    t_half_accel = np.log(2) / lambda_accel  # ≈ 10.5 years
```

---

## 4. Layer 2: Attrition Risk Model

### 4.1 Concept: Structural Gap

**Structural Gap** separates preventable from inevitable exits:

```
Structural Gap = Total Separation Rate − Natural Exit Rate
```

- **Total Separation**: All workers leaving (quits + retirements + layoffs)
- **Natural Exit**: Unavoidable exits (retirement, health, death)

### 4.2 Data Sources

| Metric | Source | Manufacturing Value |
|--------|--------|---------------------|
| Total Separation | Census QWI | 11.0%/year |
| Natural Exit | BLS Table 1.10 | 4.4%/year |
| **Structural Gap** | Calculated | **6.6%p** |

### 4.3 Cross-Industry Comparison

Industry selection criteria:
1. J2J transition probability > 5%
2. Similar skill requirements
3. Wage competition overlap

```python
industries = ['Manufacturing', 'Hospitality', 'Retail', 'Logistics', 'Construction']
natural_rates = [4.4, 4.5, 4.2, 4.3, 4.1]
total_rates = [11.0, 10.5, 9.8, 8.5, 7.2]
gaps = [t - n for t, n in zip(total_rates, natural_rates)]
```

---

## 5. Layer 3: Hiring Capacity Model

### 5.1 Layer 3-A: Replacement Ratio

#### Definition

```
Replacement Ratio (RR) = Inflow (age 25-34) / Outflow (age 55+)
```

#### Interpretation

| RR Value | Status | Meaning |
|----------|--------|---------|
| RR > 1.0 | Healthy | More young inflows than senior outflows |
| RR = 1.0 | Threshold | Exact replacement (sustainable) |
| RR < 1.0 | Deficit | Not replacing departures |

#### J2J Age Group Codes

| Code | Age Range | Role in RR |
|------|-----------|------------|
| A04 | 25-34 | Numerator (inflow) |
| A07 | 55-64 | Denominator (outflow) |
| A08 | 65-99 | Denominator (outflow) |

#### Calculation

```python
inflow = df_mfg[df_mfg['agegrp'] == 'A04'].groupby('year')['J2J'].sum()
outflow = df_mfg[df_mfg['agegrp'].isin(['A07', 'A08'])].groupby('year')['J2J'].sum()
rr_df['RR'] = inflow / outflow
```

### 5.2 Layer 3-B: Supply Gap

#### Formula

```
Effective Supply = US Graduates × Energy Belt Share × Mfg Retention Rate
Supply Gap = Effective Supply − Labor Demand
```

#### Parameter Values

| Parameter | Value | Source |
|-----------|-------|--------|
| US Graduates (CIP 14/15/47/48) | 478,651 | IPEDS 2024 |
| Energy Belt Share | 13% | BLS QCEW |
| Mfg Retention Rate | 53.1% | J2J Markov |
| **Effective Supply** | **33,047** | Calculated |
| BLS Openings (5-state) | 142,800 | JOLTS |
| Tech Occupation Share | 25% | OES |
| **Labor Demand** | **35,700** | Calculated |
| **Annual Gap** | **-2,653** | Supply − Demand |

---

## 6. Layer 4: Training Capacity Model

### 6.1 Feedback Loop

```
Senior Exit → Mentor Loss → Poor Training → High Turnover → More Hiring Pressure
     ↑                                                              ↓
     └──────────────────────── Loop ←───────────────────────────────┘
```

### 6.2 Policy Simulation

#### Phased Retirement Program

- **Source**: AARP Study (2023)
- **Effect**: Average retirement age extends +2.5 years
- **Model Impact**: Effective λ reduced by 20%

```python
lambda_policy = 0.066 * 0.8  # 20% reduction
senior_stock_policy = 100 * np.exp(-lambda_policy * t)
```

#### Results

| Scenario | 2035 Mentor Stock | Delta |
|----------|-------------------|-------|
| Current Trend | 48.7% | - |
| Retention Policy | 56.0% | +7.3 pts |

---

## 7. Layer 5: External Competition Model

### 7.1 Markov Transition Matrix

The J2J data enables construction of a **Markov Chain** showing probability of workforce transitions between industries.

#### States

```python
sectors = ['Manufacturing', 'Retail', 'Logistics', 
           'Construction', 'Services', 'Unemployment']
```

#### Transition Data (2010-2024 Cumulative)

```python
# Inflow to Manufacturing (where workers came from)
origin = [35.2, 12.5, 6.2, 8.5, 9.8, 27.8]

# Outflow from Manufacturing (where workers went)
destination = [32.1, 14.2, 11.5, 10.5, 12.8, 18.9]
```

### 7.2 Net Flow Calculation

```
Net Flow = Outflow% − Inflow%
```

| Sector | Inflow | Outflow | Net Flow | Interpretation |
|--------|--------|---------|----------|----------------|
| Manufacturing | 35.2% | 32.1% | **-3.1%p** | Net loss |
| Logistics | 6.2% | 11.5% | **+5.3%p** | Primary competitor |
| Services | 9.8% | 12.8% | +3.0%p | Secondary competitor |
| Construction | 8.5% | 10.5% | +2.0%p | Moderate threat |

### 7.3 Data Limitation Note

Layer 5 uses **Georgia proxy** data for Energy Belt. Justification:
1. Georgia is largest manufacturing state in the region
2. Industry mix is similar across 5 states
3. Transition patterns stable over 15 years

---

## 8. Statistical Methods

### 8.1 Linear Regression

Used for **Replacement Ratio trend analysis**.

```python
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(
    rr_df['year'], rr_df['RR']
)
```

#### Interpretation

| Metric | Value | Meaning |
|--------|-------|---------|
| Slope (β) | -0.0064 | RR decreases 0.0064/year |
| p-value | 0.001 | < 0.05 → statistically significant |
| R² | 0.87 | 87% of variance explained by linear trend |

### 8.2 Confidence Interval

95% CI for regression line:

```python
from scipy.stats import t as t_dist

n = len(rr_df)
residuals = rr_df['RR'] - (slope * rr_df['year'] + intercept)
se = np.sqrt(np.sum(residuals**2) / (n - 2))
t_val = t_dist.ppf(0.975, n - 2)
year_mean = rr_df['year'].mean()
sxx = np.sum((rr_df['year'] - year_mean)**2)

# For each prediction point x:
margin = t_val * se * np.sqrt(1/n + (x - year_mean)**2 / sxx)
```

### 8.3 Exponential Decay

```python
# Half-life calculation
lambda_val = 0.044
half_life = np.log(2) / lambda_val  # = 15.75 years

# Retention at time t
retention = 100 * np.exp(-lambda_val * t)
```

---

## 9. Code Reference

### 9.1 Main Functions

| Function | Layer | Purpose |
|----------|-------|---------|
| `plot_layer1_aging()` | L1 | Exponential decay simulation |
| `plot_layer2_attrition()` | L2 | Cross-industry gap comparison |
| `process_layer3_real_data()` | L3 | RR calculation + IPEDS gap |
| `plot_layer4_training()` | L4 | Policy simulation |
| `plot_layer5_competition()` | L5 | Markov transition analysis |

### 9.2 Helper Functions

```python
def draw_covid_zone(ax, y_pos_adjust=0):
    """Adds COVID-19 anomaly zone (2019-2021) to chart"""
    ax.axvspan(2019, 2021, color='#e74c3c', alpha=0.1, zorder=0)
    ...

def save_chart(filename):
    """Standardized chart saving"""
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", 
                dpi=150, bbox_inches='tight', facecolor='white')
```

### 9.3 Error Handling Pattern

```python
try:
    df = pd.read_csv(J2J_FILE, usecols=use_cols, dtype=dtype_opts)
    # ... processing
except FileNotFoundError:
    print(f"❌ Error: File not found - {J2J_FILE}")
except Exception as e:
    print(f"⚠️ Layer error: {e}")
```

---

## 10. Troubleshooting

### 10.1 Common Errors

#### "No Manufacturing data found"

**Cause**: Geography or industry filter too strict

**Solution**:
```python
# Check available values
print(df['geography'].unique())
print(df['industry'].unique())

# Verify FIPS codes
# Alabama = '1' (not '01')
# Georgia = '13'
```

#### "RR contains inf values"

**Cause**: Division by zero (outflow = 0)

**Solution**:
```python
rr_df['RR'] = np.where(
    rr_df['outflow'] > 0,
    rr_df['inflow'] / rr_df['outflow'],
    np.nan
)
rr_df = rr_df.dropna(subset=['RR'])
```

#### Memory error with large J2J file

**Solution**: Use chunked reading

```python
chunk_size = 100000
chunks = []
for chunk in pd.read_csv(J2J_FILE, chunksize=chunk_size):
    filtered = chunk[mask]
    chunks.append(filtered)
df = pd.concat(chunks, ignore_index=True)
```

### 10.2 Data Validation Checklist

- [ ] J2J file contains 5 Energy Belt states
- [ ] Age groups A04, A07, A08 all present
- [ ] Years span 2010-2024
- [ ] IPEDS CIP codes 14, 15, 47, 48 present
- [ ] No extreme outliers in RR values

---

## Appendix A: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas==2.0.0
pip install numpy==1.24.0
pip install scipy==1.10.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
```

## Appendix B: Full Requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2024 | Initial release |

---

*End of Technical Documentation*
