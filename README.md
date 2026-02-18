# **Manufacturing Talent Supply Chain at Risk**    
<p align="center">
  <strong>A 15-Year Empirical Analysis of the Energy Belt (2010–2024)</strong><br>
  <em>Quantitative Diagnosis of Workforce Attrition Using Multi-Layer Risk Framework</em><br>
  <small>(Region: Alabama · Georgia · North Carolina · South Carolina · Tennessee)</small>
</p>

## **Project Motivation**
This project is a personal journey to bridge HR strategy with rigorous data science. During my master's program at the University of Minnesota, I participated in an HR Case Competition focused on workforce aging. While our approach then was centered on identifying "best practices," I often wondered if we could go further by quantifying the actual velocity and structural mechanics of these shifts.

Today, I have revisited this challenge from a Macro-HR perspective: Talent Supply Chain Management. By manually extracting and cleansing 15 years of administrative records (2010–2024)  from the U.S. Census Bureau (J2J, QWI), the BLS, and IPEDS , I built a 5-layer diagnostic framework. My goal was to move beyond anecdotal evidence and provide an early-warning system for the manufacturing sector in the Energy Belt —a region where operational stability is currently masking a compound talent crisis.

## **The 5-Layer Risk Framework: Deep Dive**

<img width="1785" height="1035" alt="layer5_competition" src="https://github.com/user-attachments/assets/3a0935b5-11cc-4001-a831-9043dd0a0c56" />

### **Layer 1: Aging Risk — The Structural Certainty of Demographic Exit**
The most certain element of this analysis is the demographic "arithmetic" of the aging workforce. While the historical natural separation rate averages 4.4% per year , the Baby Boomer cohort crossing age 55 has accelerated this velocity to 6.6%. Compounded over a decade, this represents the difference between retaining 61% of the workforce versus only 48.7% by 2035. For a region employing 1.2 million manufacturing workers, this acceleration translates to roughly 150,000 experienced workers exiting sooner than baseline projections would suggest. The analysis identifies a 'critical mass' threshold where the senior workforce is reduced by half; under the accelerated scenario, the Energy Belt crosses this line in late 2034.


<img width="2381" height="906" alt="layer2_youth_inflow" src="https://github.com/user-attachments/assets/97ba710c-8235-45f4-8b5f-91740f689ca1" />

### **Layer 2: Youth Inflow — Currently Positive, But Rapidly Deteriorating**
Absolute youth inflow (ages 25–34) into manufacturing doubled from 45,000 in 2010 to 91,000 in 2024 , reflecting successful regional development. However, the Replacement Ratio (RR)—junior entrants per senior exit—is the true measure of sustainability. Linear regression on the 15-year series reveals a statistically powerful downward trend with a slope of −0.12 per year (p < 0.001, R² = 0.89). Although the 2024 RR stands at 2.58 , the ratio is being outpaced by the tripling of senior exits. If this trend holds, the industry will fall below the critical sustainability threshold of 1.0 by approximately 2037, making headcount contraction mathematically inevitable.

<img width="1484" height="1184" alt="layer3_retention" src="https://github.com/user-attachments/assets/39f018d2-9d9c-4ec2-9e94-9008e055a480" />

### **Layer 3: Retention Failure — Even New Hires Are Leaving**
Attracting talent is futile if workers do not stay to build institutional knowledge. Energy Belt manufacturing carries an annual separation rate of 11.0% , the highest among comparable sectors like Logistics (8.5%) or Construction (7.2%). By decomposing this attrition, we find that 6.6 percentage points of the loss are "structural"—separations driven by preventable factors such as compensation gaps, lack of shift predictability, and poor mentoring. This structural gap is the largest in the peer set, meaning roughly 60% of all manufacturing departures are theoretically preventable. While the high RR in Layer 2 looks strong, when paired with 11% churn, it creates a "revolving door" effect.

<img width="2384" height="907" alt="layer4_collapse" src="https://github.com/user-attachments/assets/be840b57-c210-495b-88aa-2e79eedbdd65" />

### **Layer 4: Accelerated Collapse — The Negative Feedback Loop Simulation**
Layers 1 through 3 interact to create a self-reinforcing collapse mechanism. This layer utilizes a discrete-time stock-flow simulation (2024–2035) to quantify the Interaction Effect: as senior workers exit , the mentoring quality degrades , which in turn causes junior attrition to spike further. In a "No Intervention" worst-case scenario, the junior workforce index could plummet by 96% by 2035. However, the simulation proves that stabilizer policies—such as phased retirement to slow senior exits by 20% and mentoring programs to cut junior attrition by 50%—can improve the 2035 outcome by 319%, effectively breaking the vicious cycle.

<img width="1785" height="1035" alt="layer5_competition" src="https://github.com/user-attachments/assets/6371dab0-ae39-4226-a1bd-6003b672a561" />

### **Layer 5: External Competition — Mapping the Sectoral Shift**
Manufacturing is losing a direct competitive battle for the regional labor pool. Using Markov transition analysis for Georgia as a representative proxy , we find that manufacturing's share of regional worker flows declined by 3.1 percentage points over 14 years. The primary victor is the Logistics sector, which gained 5.3 percentage points. Workers are migrating to Logistics not necessarily for higher wages, but for better non-wage employment experiences: predictable fixed shifts, heavy investment in ergonomic safety, and transparent career progression ladders.


**📂 Data Sources & Preparation**
The underlying analysis is built upon federal administrative records spanning 2010 to 2024:
U.S. Census Bureau J2J: Quarterly worker movements between industries and labor states.
U.S. Census Bureau QWI: Annualized separation and retention metrics.
Bureau of Labor Statistics (BLS): Table 1.10 industry separation benchmarks.
IPEDS: 15-year graduate completion data for modeling the education pipeline.
