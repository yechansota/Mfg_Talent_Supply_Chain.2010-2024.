# **Manufacturing Talent Supply Chain at Risk**    
<p align="center">
  <strong>A 15-Year Empirical Analysis of the Energy Belt (2010–2024)</strong><br>
  <em>Quantitative Diagnosis of Workforce Attrition Using Multi-Layer Risk Framework</em><br>
  <small>(Region: AL · GA · NC · SC · TN)</small>
</p>

### **Project Motivation**
This project initially began in 2023 as an HR Case Competition focused on workforce aging. While my team's approach then was centered on identifying "best practices," I often wondered if we could go further by quantifying the actual velocity and structural mechanics of these shifts.

I have revisited this challenge from a Macro-HR perspective: Talent Supply Chain Management. By manually extracting and cleansing 15 years of administrative records (2010–2024)  from 'the U.S. Census Bureau (J2J, QWI)', 'the BLS', and 'IPEDS' , I built a 5-layer diagnostic framework. My goal was to move beyond anecdotal evidence and provide an early-warning system for the manufacturing sector in the Energy Belt —a region where operational stability is currently masking a compound talent crisis.

As a professional in the automotive industry, I recognized that the long-term viability of our operations is inextricably linked to the health of the regional labor market. The "Energy Belt"—comprising AL, GA, NC, SC, and TN—serves as the backbone of U.S. manufacturing. However, beneath the surface of today’s operational stability, 15 years of federal workforce data reveal a compounding talent crisis that can no longer be ignored.

I initiated this project to provide a data-driven "stress test(What-IF-Scenario)" of our talent supply chain. By synthesizing U.S. Census Bureau (J2J) flows, BLS industry tables, and IPEDS graduate data (2010–2024), I developed a five-layer quantitative risk model to assess the structural integrity of the region's workforce.

While each layer independently signals significant stress, their interaction describes a self-reinforcing collapse mechanism. Without immediate, coordinated intervention, this demographic and competitive shift will fundamentally reshape the region’s manufacturing capacity well before 2040. This analysis serves as a strategic roadmap to identify where our "talent reservoir" is leaking and how we can secure the skilled human capital necessary for the future of the automotive sector.

**Analytical Architecture**

To transform 15 years of fragmented federal data into a predictive model, I developed a Five-Layer Diagnostic Framework. This methodology moves beyond traditional static reporting by applying System Dynamics and Predictive Analytics to the regional talent supply chain.
**Data Synthesis**: Integrated 2010–2024 records from the U.S. Census Bureau (J2J/QWI), BLS, and IPEDS.
**Modeling Approach**: Combines Exponential Decay (Demographics), Supply-Demand Gap Analysis (Education), and Stock-Flow Simulation (Knowledge Transfer).

### The Core Finding: A Multi-Layered Talent Crisis 
First, according to (`Layer 1`) (BLS Table 1.10, 2024), the Energy Belt’s senior manufacturing workforce is exiting at an accelerated rate of 6.6% per year, primarily driven by Baby Boomer retirements, which will reduce the veteran talent pool by more than half within a decade.

**Layer 1: Aging Risk**
- **Model**: Exponential Decay **$$N(t) = N₀ × e^(-λt)$$**
- **Baseline λ**: 4.4%/year (BLS Table 1.10)
- **Accelerated λ**: 6.6%/year (Baby Boomer peak scenario)
- **Result**: Half-life reduces from 15.8 → 10.5 years under acceleration

Second, as shown in (`Layer 2`) (Census J2J 2010–2024), while young workers aged 25–34 are currently entering at a ratio of 2.57 replacements per departure, this ratio is declining at a statistically significant rate of 0.12 per year (p < 0.001), putting the industry on a path to fall below the critical sustainability threshold of 1.0 by 2037.

**Layer 2: Attrition Risk**
- **Metric**: Structural Gap = Total Separation - Natural Exit
- **Manufacturing Gap**: 6.6%p (highest among peer industries)
- **Comparison**: Construction (3.1%p), Logistics (4.2%p), Retail (5.6%p)

Third, the analysis in (`Layer 3`) reveals that manufacturing's 11% annual separation rate—the highest among comparable industries—is compounded by the cultural and economic friction points.

**Layer 3: Hiring Capacity**
- **3-A (J2J)**: Replacement Ratio = Young Inflow / Senior Outflow
- 2010: RR = 0.95 | 2024: RR = 0.25 | 2035 forecast: 0.18
- Statistical significance: p = 0.001, R² = 0.87
  **3-B (IPEDS)**: Supply = 33,047 vs Demand = 35,700 → Gap = -2,653/year

Layer 3 identified in our (`Layer 4`) Stock-Flow simulation. This simulation demonstrates that the resulting "Mentorship Vacuum" accelerates junior turnover, threatening to collapse the talent pipeline index from 258 to 9 by 2035, with a systemic "point of no return" arriving as early as 2028.

**Layer 4: The No-Intervention Scenario (The "Knowledge Vacuum")**
- **Concept**: Senior depletion → Mentoring loss → Poor training → Higher turnover
- **Policy Impact**: Phased retirement can preserve +7.3pts mentor stock by 2035

Finally, the external market pressure identified in (`Layer 5`) confirms that this is not just an internal depletion but a competitive loss, as the regional workforce is being actively redistributed toward the Logistics (+5.3%) and Service (+3.0%) sectors, leaving manufacturing at a structural disadvantage in the regional "war for talent."

**Layer 5: External Competition**
- **Method**: Markov transition matrix from J2J data
- **Result**: Manufacturing net loss of -3.1%p to other industries
- **Primary Competitor**: Logistics (+5.3%p gain)
  

---

## The 5-Layer Risk Framework: Deep Dive

### **Layer 1: Aging Risk — The Structural Certainty of Demographic Exit**
<img width="2084" height="884" alt="layer1_aging" src="https://github.com/user-attachments/assets/9b29b1b2-7cb3-48e5-93f3-1dbdefae6492" />
The Energy Belt’s senior manufacturing workforce is aging out faster than it can be replaced. According to Bureau of Labor Statistics (BLS) data, the historical "natural" separation rate for manufacturing—driven by unavoidable factors like retirement and health—has averaged 4.4% annually. This represents the baseline velocity of labor loss that no employer policy can fully prevent.

Current data for the Southeast manufacturing sector shows an effective annual separation rate approaching **6.6%**. While the difference between 4.4% and 6.6% may seem modest, the compounding effect over a decade is transformative. Under the baseline scenario, the industry retains **61%** of its current workforce by 2035; under the accelerated scenario, that figure drops to **48.7%**. For a region with 1.2 million manufacturing workers, this shift represents a loss of roughly 150,000 experienced employees beyond original projections. Historical data from the Census Bureau’s Job-to-Job Flows (J2J) validates these trends. Between 2010 and 2019, senior outflows rose steadily as early Boomers reached retirement age. A brief dip occurred during the 2020–2021 pandemic lockdowns due to economic uncertainty, but by 2022, the trend resumed with even greater intensity as pent-up retirements were released.


---

### **Layer 2: Youth Inflow — Currently Positive, But Rapidly Deteriorating**
<img width="2381" height="906" alt="layer2_youth_inflow" src="https://github.com/user-attachments/assets/5357e312-0bd9-4d4a-b11c-71e6b72a8cd6" />

Youth recruitment—specifically workers aged 25–34—has seen a massive surge. In 2010, roughly 67,000 young workers entered the industry; by 2024, that number doubled to approximately 130,000. This 100% increase reflects the success of regional workforce development and the economic appeal of the Southeast. However, absolute growth is only half the story. To understand long-term sustainability, we must look at the **Replacement Ratio (RR)**: the number of young workers entering for every senior worker (aged 55+) who leaves.

$$'2010 RR: ~4.3  |  2024 RR: 2.57 | Annual Decline: ~0.12 points  per  year'$$

Statistical analysis (R-squared of 0.89) confirms this is a highly consistent, near-deterministic trend. The industry is effectively losing ground in the race against time. While youth recruitment has doubled, senior exits have tripled over the same period. The "pipeline" isn't broken, but it is being vastly outpaced by the demographic exit of the Baby Boomer generation. If this linear decline continues, the Energy Belt will hit a critical breaking point by 2037. At that stage, the RR will drop below 1.0, meaning more experienced workers will leave than young workers arrive. Without a major shift in policy or recruitment intensity, mathematical headcount contraction becomes inevitable.

---

### **Layer 3: Retention Failure — Even New Hires Are Leaving**
<img width="1784" height="1184" alt="layer3_retention" src="https://github.com/user-attachments/assets/afbeb42c-8f31-40a5-9fee-036599408347" />

Layer 3 data reveals that the Energy Belt’s manufacturing sector suffers from an **11.0% annual separation rate**. This means approximately one in nine workers leaves the industry every year—the highest turnover rate among all major regional peer sectors.

**Decomposing Attrition: Natural vs. Structural**
To solve the problem, we must distinguish between what is inevitable and what is preventable:
**Natural Attrition (4.4%)**: This is the "unavoidable floor" caused by retirements, health issues, and lifecycle changes. It is consistent across most industrial sectors.
**Structural Attrition (+6.6%)**: This represents preventable exits driven by workplace factors like stagnant wages, rigid scheduling, poor culture, and lack of career growth.

---

### **Layer 4: Accelerated Collapse — The Negative Feedback Loop Simulation**
<img width="2384" height="907" alt="layer4_collapse" src="https://github.com/user-attachments/assets/b3ed8404-abaa-46c9-bb27-696b5cba8755" />

To capture the long-term impact of knowledge loss, Layer 4 employs a Stock-Flow Simulation that tracks the dynamic movement of talent over a 12-year horizon (2024-2035)."Layer 4 utilizes a Stock-Flow Simulation from 2024 to 2035. This model treats the workforce as a reservoir that is currently leaking faster than it is being refilled.

**The No-Intervention Scenario (Red Line)** :If current trends continue, the junior workforce index collapses from **258** to **9** by **2035** due to accelerated turnover driven by a "Mentorship Vacuum(Konwledge Transfer).". This represents a total breakdown of the talent pipeline. The "critical threshold"—where the junior workforce is cut in half—is crossed as early as 2028. The Policy case shows the tangible benefit of coordinated regional investment.The gap between these two lines—the "Prevented Collapse" zone—is the space where **policy, management, and community intervention can change the outcome**. The industry is currently on a trajectory toward 2028 being a point of no return for junior talent retention.

---

### **Layer 5: External Competition — Mapping the Sectoral Shift**
<img width="1785" height="1035" alt="layer5_competition" src="https://github.com/user-attachments/assets/2bc2317b-557f-4a77-aee1-3f3344326e30" />

Utilizing a Longitudinal Migration Analysis, Layer 5 investigates the shifting preferences of the modern workforce across the Energy Belt’s industrial landscape. We analyze the Origin-Destination (O-D) matrix to determine the structural drivers behind sectoral talent drain.

**The Sectoral Shift: Winners and Losers**
The analysis compares the "Origin" (2010 share of worker flows) to the "Destination" (2024 share). The results show a clear redistribution of the regional workforce:
**Manufacturing (-3.1%)**: The largest decline in the peer set. This represents a "missed growth" story; while the industry remained stable in headcount, it failed to grow at the pace of its neighbors, resulting in a loss of roughly 100,000 to 150,000 potential workers across the five-state region.
**Unemployment (-8.9%)**: While a drop in unemployment usually signals job growth, here it primarily reflects the retirement wave identified in Layer 1. Workers aren't just moving to new jobs; they are leaving the labor force entirely.
**Logistics (+5.3%)**: The undisputed winner. Driven by the e-commerce explosion, the Logistics sector (warehousing and distribution) has surged, capturing market share directly from manufacturing.
**Services (+3.0%)**: A broad category including healthcare support and food service management. These roles often have lower skill barriers and have grown alongside the region's aging population.

---

### **Considerations**
**Geographic and Sectoral Aggregation**
The use of state-level Job-to-Job Flows (J2J) data necessitates a degree of generalization. By treating the five-state Energy Belt as a single economic unit, the model overlooks localized variations. Significant differences exist between dense, high-tech automotive corridors and rural industrial counties, which may experience demographic pressures differently.

**The COVID-19 Distortion**
The pandemic period (2020–2021) created unique anomalies across all metrics. Initial lockdowns suppressed job-switching and delayed retirements, while 2022–2023 saw a "catch-up" surge in both areas. To maintain transparency, these years are highlighted in red across all visualizations. Because these years temporarily slowed the downward trend of the Replacement Ratio, the 2037 "sustainability threshold" is likely a conservative estimate; the actual breaking point could arrive sooner if the pandemic's stabilizing effect is removed from the regression.

**Simulation Parameters and Stress Testing**
The Layer 4 simulation is a worst-case stress test, not a definitive forecast. It uses specific coefficients—such as the 0.5 mentoring quality multiplier and a 35% cap on junior attrition—to model how the system behaves under extreme stress. These parameters are designed to identify where the "revolving door" effect becomes catastrophic. Actual outcomes will fluctuate based on the timing and intensity of regional policy responses.

### **Future Work: Expanding the Diagnostic Framework**
To build upon the foundational Five-Layer Risk Model and transition from a macro-level diagnosis to targeted, actionable interventions, future research will focus on the following key areas:

**Occupational Segmentation via SOC Codes:**
The current model evaluates the manufacturing sector at an aggregate industry level. Future iterations will disaggregate the data using Standard Occupational Classification (SOC) codes to isolate specific vulnerabilities. By comparing production-floor roles (e.g., SOC 51-xxxx) against administrative or technical roles (e.g., SOC 43-xxxx), we can identify occupation-specific attrition drivers and tailor intervention points with greater precision.

**Real-Time Dashboard with Early Warning System:**
To move beyond static historical analysis, we aim to develop a dynamic, real-time workforce analytics dashboard. This system will provide monthly updates on the Replacement Ratio (RR) and incorporate automated configurable alerts for exceptions or critical threshold breaches. By highlighting outliers and predictive signals (e.g., flight-risk forecasts), this early warning system will empower HR leaders to proactively address localized talent leaks before they compound into systemic failures.

**Causal Inference for Policy Evaluation:**
While the current Layer 4 simulation models the potential impact of interventions (e.g., phased retirement), future work will employ causal inference methods alongside A/B testing to empirically evaluate the effectiveness of newly implemented retention policies. This methodological shift will allow us to move beyond observing basic outcomes ("Did this work?") toward understanding the underlying mechanics ("What caused it?", "For whom did it work?", and "Why?").

**Representative Proxy Modeling**
The reliance on Georgia as a proxy for the entire Energy Belt in Layer 5 is a methodological choice based on its status as the region's largest manufacturing employer. While Georgia's diverse industrial mix (automotive, aerospace, food processing) makes it an excellent representative for Alabama or North Carolina, its specific status as a massive logistics hub (centered in Atlanta) may slightly overstate the competitive growth of the Logistics sector compared to more rural parts of the region.
