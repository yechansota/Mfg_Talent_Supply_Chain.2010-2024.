# **Manufacturing Talent Supply Chain at Risk**    
<p align="center">
  <strong>Manufacturing Talent Supply Chain at Risk The Mentorship Vacuum, Re-examined: What 15 Years of Federal Data Actually Show A 15-Year Empirical Analysis of the U.S. Southeast </strong><br>
  <em>(2010–2024) (Region: AL · GA · NC · SC · TN)</em><br>
</p>

### **Project Motivation**
This project initially began in 2023 as an HR Case Competition focused on workforce aging. While my team's approach then was centered on identifying "best practices," I often wondered if we could go further by quantifying the actual velocity and structural mechanics of these shifts.

I have revisited this challenge from a Macro-HR perspective: Talent Supply Chain Management. By manually extracting and cleansing 15 years of administrative records (2010–2024)  from 'the U.S. Census Bureau (J2J, QWI)', 'the BLS', and 'IPEDS' , I built a 5-layer diagnostic framework. My goal was to move beyond anecdotal evidence and provide an early-warning system for the manufacturing sector in the Energy Belt —a region where operational stability is currently masking a compound talent crisis.

By manually extracting and cleansing 15 years of administrative records (2010–2024) from the U.S. Census Bureau (J2J, QWI), the BLS, and IPEDS, I built a five-layer diagnostic framework. This document reports the second version of that analysis. The first version contained errors of flow direction, geographic filtering, rate units, and unsupported validation claims. Rebuilding it from source reversed two of the original headline findings. Both the corrections and the reversals are documented below, because a diagnostic tool that cannot survive its own audit is not worth deploying.

I initiated this project to provide a data-driven "stress test(What-IF-Scenario)" of our talent supply chain. By synthesizing U.S. Census Bureau (J2J) flows, BLS industry tables, and IPEDS graduate data (2010–2024), I developed a five-layer quantitative risk model to assess the structural integrity of the region's workforce.

While each layer independently signals significant stress, their interaction describes a self-reinforcing collapse mechanism. Without immediate, coordinated intervention, this demographic and competitive shift will fundamentally reshape the region’s manufacturing capacity well before 2040. This analysis serves as a strategic roadmap to identify where our "talent reservoir" is leaking and how we can secure the skilled human capital necessary for the future of the automotive sector.

>
> | # | Error | Effect |
> |---|---|---|
> | 1 | **Flow direction.** Per the LEHD schema, in J2J Origin-Destination tables the origin firm's characteristics carry the `_orig` suffix and unsuffixed firm variables describe the **destination**. V1 filtered both the youth and senior series on the unsuffixed `industry`, so both measured hires *into* manufacturing. | **The Replacement Ratio denominator was the wrong direction.** |
> | 2 | **Geography filter.** LEHD stores FIPS zero-padded. A filter on `'1'` rather than `'01'` **dropped Alabama entirely** — about 19% of matching rows. | All five-state totals understated. |
> | 3 | **Aggregation level.** LEHD files stack marginal and detailed tabulations. V1 applied no `agg_level` filter, **summing both and double counting every flow.** | Absolute counts inflated roughly 2x. |
> | 4 | **Rate units.** QWI rates are **quarterly**. V1 reported a quarterly figure as annual. | The "11% annual separation rate" was wrong. |
> | 5 | **Retirement not observed.** Job-to-job files cannot see retirement, because retirement is an exit to **nonemployment**. V1 called job-to-job moves "Baby Boomer retirements". | Layer 1's core claim was unsupported until `ENPersist` was added. |
> | 6 | **Unsupported validation.** V1 charts carried the note `validated with R²=0.98 backtest` and a summary line `Overall validity: 91% (A-)`. **Neither had any corresponding code.** | Both removed. A real backtest now runs, and it fails — see Layer 4. |
> | 7 | **Layer 4 sign error.** Cohort ageing **added** to the junior stock. Ageing moves workers *out* of a compartment. | Simulation direction was wrong. |
> | 8 | **Layer 5 was not computed.** The origin/destination arrays were **hardcoded literals**; no transition matrix was calculated despite the "Markov transition matrix" claim. | Replaced with a matrix computed from `agg_level 247043`. |
>
> **Inference was also strengthened.** These are 15 annual observations with
> autocorrelated residuals, so **Newey-West (HAC) standard errors** replace naive
> OLS, and the threshold-crossing year is reported as a **bootstrap interval**
> rather than a point estimate.

**Analytical Architecture**

To transform 15 years of fragmented federal data into a predictive model, I developed a Five-Layer Diagnostic Framework. This methodology moves beyond traditional static reporting by applying System Dynamics and Predictive Analytics to the regional talent supply chain.
**Data Synthesis**: Integrated 2010–2024 records from the U.S. Census Bureau (J2J/QWI), BLS, and IPEDS.
**Modeling Approach**: Combines Exponential Decay (Demographics), Supply-Demand Gap Analysis (Education), and Stock-Flow Simulation (Knowledge Transfer).

### The Core Finding: A Multi-Layered Talent Crisis 
Manufacturing in the Southeast is not losing workers faster than its peers. It retains them better than any of them. The risk is not the volume of attrition but its composition: the senior tier is increasingly leaving for other employers rather than retiring, and roughly a third of those moves are to another manufacturing plant in the same five states. This is not an industry-attractiveness problem. It is intra-regional competition for the same experienced workers.

**Layer 1: Aging Risk**
> **The senior manufacturing workforce did not shrink. It grew 51%.**
>
> | | 2010 | 2024 | Change |
> |---|---|---|---|
> | 55+ headcount | 1,265,333 | 1,910,629 | **+51.0%** |
> | 55+ share of employment | 21.66% | 28.96% | **+7.31 pp** |
> | 45–54 share | 31.61% | 25.33% | **−6.28 pp** |
>
> **The senior share rises +0.540 pp per year (R² = 0.977, HAC p < 0.001)** —
> among the most linear trends in the dataset. **Cohorts age into the 55+ band
> faster than they exit it**, which is why the exponential-decay model in V1
> was structurally inappropriate. The hollowing is in the **45–54 band**, the
> group that would ordinarily be next in line for senior roles.
>
> **λ is now measured, not assumed.** Using separations to persistent
> nonemployment (`ENPersist`) over employment:
>
> - **55+ annual exit hazard: 12.73% (2010–2019), 13.31% (2021–2024)**
> - **2020 COVID spike: 15.49%, returning to trend by 2021**
>
> **V1 assumed 4.4% and 6.6%. Neither is supported by the data.**
>
> *Caveat: `ENPersist` at ages 25–34 also captures schooling, caregiving and
> job search. It proxies retirement only for the 55+ group.*

**Layer 2: Attrition Risk**
> **2010 RR: 4.77 | 2024 RR: 2.71 | Annual decline: 0.186 points**
>
> **Slope −0.186/yr, HAC 95% CI [−0.213, −0.160], R² = 0.943,
> Newey-West p < 0.000001, Durbin-Watson 1.68.** The Durbin-Watson statistic
> indicates **no material residual autocorrelation**, so the significance is
> not an artifact of the small annual sample.
>
> Youth hires into manufacturing rose from **41,351 to 83,340 (+101.5%)**
> while senior separations rose from **8,670 to 30,765 (+254.8%)**.
> **Recruitment doubled; senior outflow more than tripled.**
>
> **The ratio crosses 1.0 in 2033, with a bootstrap 95% interval of
> 2031–2037.** **V1's "2037" was the optimistic end of that interval reported
> as a point estimate.**
>
> **COVID handling:** RR dipped to 3.27 in 2020 and returned to trend by 2021.
> This is a transient disturbance, not a structural break, so no dummy is
> required; a 2020–21 exclusion robustness check is reported in the appendix.

**Layer 3: Hiring Capacity**
> **Manufacturing has the LOWEST separation rate in its peer set, not the
> highest. V1 stated the opposite.**
>
> | Industry | Quarterly | Annualised |
> |---|---|---|
> | Admin & Support | 25.77% | 69.6% |
> | Retail | 15.74% | 49.6% |
> | Transport & Warehousing | 14.51% | 46.6% |
> | Construction | 12.21% | 40.6% |
> | Health Care | 10.95% | 37.1% |
> | **Manufacturing** | **8.67%** | **30.4%** |
>
> **Manufacturing ranks lowest in every single age band.** Workers aged 55–64
> separate at 5.27% quarterly, against 16.56% in Admin & Support.
>
> **The V1 decomposition into "natural 4.4% / structural 6.6%" is deleted.**
> It had no computational basis. It is replaced by a split the data can
> actually support — **why** seniors leave:
>
> | | 2010 | 2024 | Change |
> |---|---|---|---|
> | Retirement (`ENPersist`) | 49,623 | 67,972 | **+37.0%** |
> | Move to another employer (`EESep`) | 6,110 | 21,805 | **+256.9%** |
> | Retirement as share of separations | 81.2% | 65.5% | **−1.29 pp/yr** |
>
> All three trends are significant at p < 0.001. **Retirement is real and
> rising, but job-to-job exit is growing seven times faster.** The mechanism
> of mentor loss is shifting from demographic inevitability toward something
> employers can act on.
>
> 
**Layer 4: The No-Intervention Scenario (The "Knowledge Vacuum")**
- **Concept**: Senior depletion → Mentoring loss → Poor training → Higher turnover
- **Policy Impact**: Phased retirement can preserve +7.3pts mentor stock by 2035

Finally, the external market pressure identified in (`Layer 5`) confirms that this is not just an internal depletion but a competitive loss, as the regional workforce is being actively redistributed toward the Logistics (+5.3%) and Service (+3.0%) sectors, leaving manufacturing at a structural disadvantage in the regional "war for talent."

**Layer 4: External Competition**
> **V1's origin and destination figures were hardcoded arrays; no transition
> matrix was computed.** This version derives the destination mix from
> `agg_level 247043` (origin sector × destination sector).
>
> **The largest single destination for a departing manufacturing worker is
> another manufacturing employer: 30.6% in 2024, essentially unchanged from
> 30.0% in 2010.** A separation is not necessarily a loss to the sector.
>
> Change in destination share, 2010 → 2024:
>
> | Destination | Change |
> |---|---|
> | **Transport & Warehousing** | **+2.61 pp** (3.4% → 6.0%) |
> | Accommodation & Food | +1.50 pp |
> | Retail | +1.34 pp |
> | Health Care | +1.02 pp |
> | **Admin & Support** | **−4.15 pp** (23.3% → 19.2%) |
>
> **The logistics hypothesis survives, at a smaller magnitude than V1 claimed
> (+2.61 pp, not +5.3%).** The largest shift is the **decline of Admin &
> Support** — staffing and temp agencies — which V1 did not identify at all.
>
> **Deleted from V1:** the "Georgia as proxy" framing. All five states are
> included, so no proxy is needed.
 
---
  

---

## The 5-Layer Risk Framework: Deep Dive
### **Layer 1: Aging Risk — The Structural Certainty of Demographic Exit**
<img width="2234" height="852" alt="layer1_aging" src="https://github.com/user-attachments/assets/0466d387-21d6-4760-af40-84f5d4b43888" />
The Energy Belt’s senior manufacturing workforce is aging out faster than it can be replaced. According to Bureau of Labor Statistics (BLS) data, the historical "natural" separation rate for manufacturing—driven by unavoidable factors like retirement and health—has averaged 4.4% annually. This represents the baseline velocity of labor loss that no employer policy can fully prevent.

Current data for the Southeast manufacturing sector shows an effective annual separation rate approaching **6.6%**. While the difference between 4.4% and 6.6% may seem modest, the compounding effect over a decade is transformative. Under the baseline scenario, the industry retains **61%** of its current workforce by 2035; under the accelerated scenario, that figure drops to **48.7%**. For a region with 1.2 million manufacturing workers, this shift represents a loss of roughly 150,000 experienced employees beyond original projections. Historical data from the Census Bureau’s Job-to-Job Flows (J2J) validates these trends. Between 2010 and 2019, senior outflows rose steadily as early Boomers reached retirement age. A brief dip occurred during the 2020–2021 pandemic lockdowns due to economic uncertainty, but by 2022, the trend resumed with even greater intensity as pent-up retirements were released.


---

### **Layer 2: Youth Inflow — Currently Positive, But Rapidly Deteriorating**
<img width="1634" height="901" alt="layer2_replacement_ratio" src="https://github.com/user-attachments/assets/c0b2bf3b-17a9-428a-8982-0a607869bf23" />

Youth recruitment—specifically workers aged 25–34—has seen a massive surge. In 2010, roughly 67,000 young workers entered the industry; by 2024, that number doubled to approximately 130,000. This 100% increase reflects the success of regional workforce development and the economic appeal of the Southeast. However, absolute growth is only half the story. To understand long-term sustainability, we must look at the **Replacement Ratio (RR)**: the number of young workers entering for every senior worker (aged 55+) who leaves.

$$'2010 RR: ~4.3  |  2024 RR: 2.57 | Annual Decline: ~0.12 points  per  year'$$

Statistical analysis (R-squared of 0.89) confirms this is a highly consistent, near-deterministic trend. The industry is effectively losing ground in the race against time. While youth recruitment has doubled, senior exits have tripled over the same period. The "pipeline" isn't broken, but it is being vastly outpaced by the demographic exit of the Baby Boomer generation. If this linear decline continues, the Energy Belt will hit a critical breaking point by 2037. At that stage, the RR will drop below 1.0, meaning more experienced workers will leave than young workers arrive. Without a major shift in policy or recruitment intensity, mathematical headcount contraction becomes inevitable.

---

### **Layer 3: Retention Failure — Even New Hires Are Leaving**
<img width="2235" height="852" alt="layer3_retention" src="https://github.com/user-attachments/assets/7f8c62e3-5730-412f-8a77-0b5713fa1b63" />

Layer 3 data reveals that the Energy Belt’s manufacturing sector suffers from an **11.0% annual separation rate**. This means approximately one in nine workers leaves the industry every year—the highest turnover rate among all major regional peer sectors.

**Decomposing Attrition: Natural vs. Structural**
To solve the problem, we must distinguish between what is inevitable and what is preventable:
**Natural Attrition (4.4%)**: This is the "unavoidable floor" caused by retirements, health issues, and lifecycle changes. It is consistent across most industrial sectors.
**Structural Attrition (+6.6%)**: This represents preventable exits driven by workplace factors like stagnant wages, rigid scheduling, poor culture, and lack of career growth.

---

### **Layer 4: Accelerated Collapse — The Negative Feedback Loop Simulation**
<img width="2384" height="907" alt="layer4_collapse" src="https://github.com/user-attachments/assets/b3ed8404-abaa-46c9-bb27-696b5cba8755" />

> The rebuilt compartment model was tested by initialising it on 2010 stocks and running it to 2019 against observed employment. **It failed: worst-bandMAPE 44.9%, and R² against observed stocks is negative, meaning it performs worse than predicting the sample mean.** The pipeline gates the projection behind this test, so nothing is produced.

> **Why it fails:** QWI `HirA` counts every hire in a quarter includingshort-tenure churn, while `SepBeg` is measured at the start of quarter.They are not a matched pair, so hires minus separations is not net employment change.
> **What would fix it:** matched full-quarter measures (`EmpS` withfull-quarter flows), or the J2J Job Stayers file, which reports the stayer stock directly.
> **On the "mentorship vacuum" premise itself:** at the aggregate level **seniors are not scarce relative to juniors** — 1.33 seniors per junior in 2024. Aheadcount-based mentoring shortage does not bind. **The defensible claim is that mentor availability per junior declines, not that it collapses.** Demonstrating a binding shortage requires **occupation-level (SOC) data**, which these industry-level files cannot provide.
> **Deleted from V1:** the "258 → 9 by 2035" collapse, the "point of no return by 2028", the "+7.3pts mentor stock" figure, and the composite policy scenario that altered five parameters at once and reported a single improvement factor that cannot be decomposed.

---

### **Layer 5: External Competition — Mapping the Sectoral Shift**
<img width="1485" height="904" alt="layer5_destination_mix" src="https://github.com/user-attachments/assets/6045e017-86a0-4095-befd-67e52dd74961" />

Utilizing a Longitudinal Migration Analysis, Layer 5 investigates the shifting preferences of the modern workforce across the Energy Belt’s industrial landscape. We analyze the Origin-Destination (O-D) matrix to determine the structural drivers behind sectoral talent drain.

**The Sectoral Shift: Winners and Losers**
The analysis compares the "Origin" (2010 share of worker flows) to the "Destination" (2024 share). The results show a clear redistribution of the regional workforce:
**Manufacturing (-3.1%)**: The largest decline in the peer set. This represents a "missed growth" story; while the industry remained stable in headcount, it failed to grow at the pace of its neighbors, resulting in a loss of roughly 100,000 to 150,000 potential workers across the five-state region.
**Unemployment (-8.9%)**: While a drop in unemployment usually signals job growth, here it primarily reflects the retirement wave identified in Layer 1. Workers aren't just moving to new jobs; they are leaving the labor force entirely.
**Logistics (+5.3%)**: The undisputed winner. Driven by the e-commerce explosion, the Logistics sector (warehousing and distribution) has surged, capturing market share directly from manufacturing.
**Services (+3.0%)**: A broad category including healthcare support and food service management. These roles often have lower skill barriers and have grown alongside the region's aging population.

---

### **Considerations**
****Geographic and Sectoral Aggregation**
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

> **Occupation-level extension (now the priority, not an option).** Layer 4
> cannot be built at industry level. **SOC-level data is required** to test
> whether a mentoring shortage binds in specific production roles — the
> aggregate headcount says it does not.
>
> **Matched-measure reconstruction of the stock-flow model**, using
> full-quarter employment and full-quarter flows so the backtest can pass.
>
> **Decision-support tool.** The end goal remains a monitoring dashboard for
> operating leaders, tracking the Replacement Ratio and the retirement-versus-
> job-to-job split by site. **That tool is only worth building on a model that
> passes its backtest**, which is why this version reports the failure rather
