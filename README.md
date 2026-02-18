<p align="center">
# Manufacturing-Talent-Supply-Chain-at-Risk
  </p>
Manufacturing Talent Supply Chain at Risk: A 15-Year Empirical Analysis of the Energy Belt (2010–2024)
<p align="center">
  <b>Quantitative Diagnosis of Workforce Attrition Using Multi-Layer Risk Framework</b><br>
  <i>Energy Belt Region: AL | GA | NC | SC | TN </i>
</p>

📋 Overview
This project analyzes the structural collapse of the manufacturing workforce pipeline in the U.S. Energy Belt region. Using federal government data (Census J2J, IPEDS, BLS), we decompose workforce attrition into five interconnected risk layers and provide actionable insights for HR practitioners.
Key Questions Addressed

How fast is the workforce depleting due to aging?
Why are workers leaving beyond natural retirement?
Can we replace those who leave?
Can we effectively train new hires?
Are other industries stealing our talent?


🔑 Key Findings
MetricCurrent Value2035 ForecastAssessmentWorkforce Half-Life15.8 years10.5 years (accelerated)🔴 CRITICALReplacement Ratio0.250.18🔴 CRITICALStructural Gap6.6%p-🟡 HIGHAnnual Supply Gap-2,653 workers-🟡 MODERATE

🏗️ Framework: 5-Layer Risk Model
┌─────────────────────────────────────────────────────────────┐
│           TALENT SUPPLY CHAIN RISK MODEL                    │
└─────────────────────────────────────────────────────────────┘

Layer 1: AGING RISK          → How fast is workforce depleting?
         ↓                       λ = 4.4%/yr, Half-Life = 15.8yr
Layer 2: ATTRITION RISK      → Why do people leave voluntarily?
         ↓                       Structural Gap = 6.6%p
Layer 3: HIRING CAPACITY     → Can we replace those who leave?
         ↓                       RR = 0.25 (trend: -0.0064/yr)
Layer 4: TRAINING CAPACITY   → Can we train new hires properly?
         ↓                       Mentor Stock declining
Layer 5: EXTERNAL COMPETITION → Are others stealing our talent?
                                 Net Flow: -3.1%p to other industries

📊 Data Sources
SourceProviderPeriodRecordsUsageJ2JCensus Bureau2010-2024~85MReplacement Ratio, Markov ChainIPEDSDept. of Education2010-2024~4.3MGraduate supply analysisBLSBureau of Labor Statistics2024-Exit rates, job openings

📁 Project Structure
manufacturing-talent-risk/
│
├── data/                       # Raw and processed data
│   ├── j2j_census.csv          # Census J2J flows (large file)
│   ├── ipeds_completions.csv   # IPEDS graduate data
│   └── bls_separations.xlsx    # BLS Table 1.10
│
├── src/                        # Source code
│   ├── final_portfolio.py      # Main analysis script
│   ├── j2j_preprocessing.py    # J2J data cleaning
│   └── utils.py                # Helper functions
│
├── output/                     # Generated outputs
│   ├── layer1_aging.png
│   ├── layer2_attrition.png
│   ├── layer3a_rr.png
│   ├── layer3b_gap.png
│   ├── layer4_training.png
│   ├── layer5_competition.png
│   └── analysis_summary.txt
│
├── docs/                       # Documentation
│   ├── TECHNICAL.md            # Technical documentation
│   └── portfolio.docx          # Executive report
│
├── README.md                   # This file
└── requirements.txt            # Python dependencies

🚀 Quick Start
Prerequisites
bash# Python 3.10+
python --version

# Required packages
pip install pandas numpy scipy matplotlib seaborn
Installation
bash# Clone repository
git clone https://github.com/yourusername/manufacturing-talent-risk.git
cd manufacturing-talent-risk

# Install dependencies
pip install -r requirements.txt
Running Analysis
bash# Execute main analysis
python src/final_portfolio.py

# Output saved to ./output/

📈 Layer Details
Layer 1: Aging Risk

Model: Exponential Decay (N(t) = N₀ × e^(-λt))
Baseline λ: 4.4%/year (BLS Table 1.10)
Accelerated λ: 6.6%/year (Baby Boomer peak scenario)
Result: Half-life reduces from 15.8 → 10.5 years under acceleration

Layer 2: Attrition Risk

Metric: Structural Gap = Total Separation - Natural Exit
Manufacturing Gap: 6.6%p (highest among peer industries)
Comparison: Construction (3.1%p), Logistics (4.2%p), Retail (5.6%p)

Layer 3: Hiring Capacity

3-A (J2J): Replacement Ratio = Young Inflow / Senior Outflow

2010: RR = 0.95 | 2024: RR = 0.25 | 2035 forecast: 0.18
Statistical significance: p = 0.001, R² = 0.87


3-B (IPEDS): Supply = 33,047 vs Demand = 35,700 → Gap = -2,653/year

Layer 4: Training Capacity

Concept: Senior depletion → Mentoring loss → Poor training → Higher turnover
Policy Impact: Phased retirement can preserve +7.3pts mentor stock by 2035

Layer 5: External Competition

Method: Markov transition matrix from J2J data
Result: Manufacturing net loss of -3.1%p to other industries
Primary Competitor: Logistics (+5.3%p gain)
