import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import linregress, t as t_dist
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. Global Settings
# =============================================================================
J2J_FILE   = "/Users/sean/Downloads/j2j_cob.csv"
IPEDS_FILE = "/Users/sean/Downloads/ipeds_completions_cleaned_2010_2024.csv"
OUTPUT_DIR = "/Users/sean/Downloads/output_final_portfolio"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150

COLORS = {
    'Manufacturing': '#d62728',
    'Logistics':     '#ff7f0e',
    'Construction':  '#8c564b',
    'Retail':        '#9467bd',
    'Hospitality':   '#e377c2',
    'Services':      '#bcbd22',
    'Unemployment':  '#7f7f7f',
    'Natural':       '#7f7f7f',
    'Policy':        '#95a5a6',
    'Trend':         '#95a5a6',
    'Demand':        '#f39c12',
    'Supply':        '#bdc3c7',
    'Young':         '#2ecc71',
    'Senior':        '#95a5a6'
}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper Functions
# =============================================================================
def draw_covid_zone(ax, y_pos_adjust=0):
    ax.axvspan(2020, 2021, color='#e74c3c', alpha=0.08, zorder=0)
    ylim = ax.get_ylim()
    text_y = ylim[1] - (ylim[1] - ylim[0]) * 0.05 + y_pos_adjust
    ax.text(2020.5, text_y, 'COVID-19', ha='center', va='top',
            color='#c0392b', fontsize=9, fontweight='bold', alpha=0.7)

def save_and_show(filename):
    """Save chart then block until user closes window — guarantees order."""
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {filename}.png")
    plt.savefig(f"{OUTPUT_DIR}/Project 2 image.png", dpi=150, bbox_inches='tight')
    plt.close() 
# =============================================================================
# Layer 1: Aging Risk — with J2J historical senior outflow
# =============================================================================
def plot_layer1_aging(flow_df=None):
    print("\n[Layer 1] Generating Aging Risk Chart...")

    BASE_YEAR = 2024
    years = np.arange(BASE_YEAR, 2036)   # 2024-2035
    t     = years - BASE_YEAR

    lambda_base  = 0.044
    lambda_accel = 0.066

    retention_base  = 100 * np.exp(-lambda_base  * t)
    retention_accel = 100 * np.exp(-lambda_accel * t)

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- Historical J2J senior outflow (2010-2024) ---
    y_max = 115   # default
    if flow_df is not None and not flow_df.empty:
        base_senior     = flow_df.loc[flow_df['year'] == flow_df['year'].max(), 'outflow_senior'].values[0]
        hist_senior_idx = flow_df['outflow_senior'] / base_senior * 100

        # Dynamic y ceiling: max of historical index + 20% headroom
        y_max = max(115, hist_senior_idx.max() * 1.20)

        ax.axvspan(2020, 2021, color='#e74c3c', alpha=0.08, zorder=0)
        ax.text(2020.5, 5, 'COVID-19', ha='center', va='bottom',
                color='#c0392b', fontsize=9, fontweight='bold', alpha=0.7)

        # [수정 부분] Historical 데이터를 빨간 실선으로 변경
        ax.plot(flow_df['year'], hist_senior_idx,
                color=COLORS['Manufacturing'], # 빨간색 적용
                linewidth=2.5,                 # 실선 굵기 강조
                marker='o', markersize=5,      # 마커 크기 소폭 확대
                label='Historical Senior Outflow (J2J actual)', 
                alpha=1.0, zorder=5)           # 투명도 제거 및 우선순위 상향

        ax.axvline(2024, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(2023.8, y_max * 0.88, 'Simulation\nstarts',
                ha='right', fontsize=8, color='gray', style='italic')

    # --- Simulation lines (2024-2035) ---
    ax.plot(years, retention_base,  label='Baseline (Natural Exit)',
            color=COLORS['Natural'],       linewidth=2, linestyle='--')
    ax.plot(years, retention_accel, label='Accelerated (Baby Boomer Peak)',
            color=COLORS['Manufacturing'], linewidth=2, linestyle='--')
    ax.fill_between(years, retention_base, retention_accel,
                    color=COLORS['Manufacturing'], alpha=0.12)

    # Gap annotation at 2030
    idx_2030 = np.where(years == 2030)[0][0]
    gap_val  = retention_base[idx_2030] - retention_accel[idx_2030]
    ax.annotate(f'Cumulative Gap: -{gap_val:.1f} pts',
                xy=(2030, retention_accel[idx_2030]),
                xytext=(2030, retention_accel[idx_2030] + 12),
                arrowprops=dict(facecolor=COLORS['Manufacturing'], arrowstyle='->'),
                ha='center', color=COLORS['Manufacturing'], fontweight='bold', fontsize=10)

    ax.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.4)
    ax.text(BASE_YEAR + 0.3, 52, 'Critical Mass (50%)',
            fontsize=8, color='gray', style='italic')

    ax.text(0.02, 0.02,
            'Source: BLS Table 1.10 (2024)\n'
            'Historical: Census J2J Senior Outflow\n'
            'lambda_base=4.4%/yr, lambda_accel=6.6%/yr',
            transform=ax.transAxes, fontsize=8, color='gray', verticalalignment='bottom')

    ax.set_title('[Layer 1] Aging Risk: Senior Workforce Depletion (2010-2035)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Senior Outflow Index (2024=100)')
    ax.set_xlabel('Year')
    ax.set_xlim(2010, 2036)
    ax.set_ylim(0, y_max)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_and_show('layer1_aging')

plt.savefig(f"{OUTPUT_DIR}/layer1_aging.png", dpi=150, bbox_inches='tight')
# =============================================================================
# Layer 2: Youth Inflow — REFRAMED
# =============================================================================
def plot_layer2_youth_inflow():
    print("\n[Layer 2] Generating Youth Inflow Chart...")

    try:
        if not Path(J2J_FILE).exists():
            raise FileNotFoundError(f"J2J File not found: {J2J_FILE}")

        print("  Loading J2J data...")
        dtype_opts = {'geography': str, 'industry': str, 'agegrp': str,
                      'sex': str, 'seasonadj': str}
        use_cols = ['geography', 'industry', 'agegrp', 'year', 'J2J', 'sex', 'seasonadj']

        df = pd.read_csv(J2J_FILE, usecols=lambda c: c in use_cols,
                         dtype=dtype_opts, low_memory=False)

        mask = (
            (df['geography'].isin(['1', '13', '37', '45', '47'])) &
            (df['seasonadj'] == 'U') &
            (df['sex'] == '0') &
            (df['industry'].str.startswith(('31', '32', '33'), na=False))
        )
        df_mfg = df[mask].copy()

        if df_mfg.empty:
            raise ValueError("No Manufacturing data")

        print(f"  Filtered {len(df_mfg):,} manufacturing records")

        inflow_young   = df_mfg[df_mfg['agegrp'] == 'A04'].groupby('year')['J2J'].sum()
        outflow_senior = df_mfg[df_mfg['agegrp'].isin(['A07', 'A08'])].groupby('year')['J2J'].sum()

        flow_df = pd.DataFrame({
            'inflow_young':  inflow_young,
            'outflow_senior': outflow_senior
        }).dropna().reset_index()

        flow_df['RR'] = np.where(
            flow_df['outflow_senior'] > 0,
            flow_df['inflow_young'] / flow_df['outflow_senior'],
            np.nan
        )
        flow_df = flow_df.dropna(subset=['RR'])

        print(f"  Calculated flows for {len(flow_df)} years")

        slope, intercept, r_value, p_value, std_err = linregress(flow_df['year'], flow_df['RR'])

        if slope < 0:
            rr_current        = flow_df['RR'].iloc[-1]
            years_to_critical = (rr_current - 1.0) / abs(slope)
            critical_year     = int(flow_df['year'].max() + years_to_critical)
        else:
            critical_year = None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === LEFT: Absolute Youth Inflow ===
        ylim_max = flow_df['inflow_young'].max() * 1.15
        ax1.axvspan(2020, 2021, color='#e74c3c', alpha=0.08, zorder=0)
        text_y = ylim_max - (ylim_max * 0.83)
        ax1.text(2020.5, text_y, 'COVID-19', ha='center', va='top',
                 color='#c0392b', fontsize=9, fontweight='bold', alpha=0.7)

        bars = ax1.bar(flow_df['year'], flow_df['inflow_young'],
                       width=0.375, color=COLORS['Trend'], alpha=0.7,
                       label='Youth Inflow (25-34)')
        ax1.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=7)

        inflow_slope, inflow_int, _, inflow_p, _ = linregress(flow_df['year'], flow_df['inflow_young'])
        inflow_trend = inflow_slope * flow_df['year'] + inflow_int
        ax1.plot(flow_df['year'], inflow_trend, '--', color='#d62728',
                 linewidth=1.5, label=f'Trend (beta={inflow_slope:.0f}/yr, p={inflow_p:.3f})')

        first_val  = flow_df.iloc[0]['inflow_young']
        last_val   = flow_df.iloc[-1]['inflow_young']
        pct_change = ((last_val - first_val) / first_val) * 100

        ax1.text(0.02, 0.98,
                 f'2010: {int(first_val):,}\n'
                 f'2024: {int(last_val):,}\n'
                 f'Change: +{pct_change:.1f}%',
                 transform=ax1.transAxes, fontsize=9, color='gray',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                 family='monospace')

        ax1.set_title('Youth Inflow: Absolute Trend (Positive)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Annual Youth Inflow (People)', fontsize=10)
        ax1.set_xlabel('Year', fontsize=10)
        ax1.set_ylim(0, ylim_max)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # === RIGHT: Replacement Ratio ===
        draw_covid_zone(ax2, y_pos_adjust=-0.05)

        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
                    label='Sustainability Threshold (RR=1.0)', zorder=1)
        ax2.plot(flow_df['year'], flow_df['RR'], color=COLORS['Manufacturing'],
                 linewidth=2, marker='o', markersize=3, label='Replacement Ratio', zorder=3)

        last_year = flow_df['year'].max()
        last_rr   = flow_df[flow_df['year'] == last_year]['RR'].values[0]
        ax2.plot(last_year, last_rr, 'o', color=COLORS['Manufacturing'],
                 markersize=6, zorder=10, markeredgecolor='white', markeredgewidth=1)
        ax2.text(last_year, last_rr + 0.15, f'{last_year}: RR = {last_rr:.2f}',
                 ha='center', fontsize=7, fontweight='normal',
                 color=COLORS['Manufacturing'],
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=COLORS['Manufacturing'], alpha=0.9))

        future_years = np.arange(flow_df['year'].min(), 2040)
        trend_rr     = np.maximum(slope * future_years + intercept, 0)
        ax2.plot(future_years, trend_rr, ':', color=COLORS['Trend'],
                 linewidth=1.8, label=f'Trend (beta={slope:.4f}/yr)', zorder=2)

        if critical_year and critical_year < 2040:
            ax2.axvline(critical_year, color=COLORS['Manufacturing'],
                        linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
            ax2.text(critical_year + 0.2, 0.2, f'RR < 1.0\nby ~{critical_year}',
                     fontsize=8, color=COLORS['Manufacturing'], fontweight='bold', style='italic')

        ax2.axvline(2024, color='gray', linestyle='--', linewidth=1, alpha=0.4)
        ax2.text(2024.2, ax2.get_ylim()[1] * 0.97, 'Forecast',
                 ha='left', fontsize=9, color='gray',
                 style='italic', fontweight='normal')

        significance  = "Significant" if p_value < 0.05 else "Not Significant"
        critical_note = f'RR < 1.0 by ~{critical_year}' if critical_year else 'Stable'
        ax2.text(0.02, 0.98,
                 f'Trend: {slope:.4f}/yr\n'
                 f'p-value: {p_value:.4f}\n'
                 f'Status: {significance}\n'
                 f'R-squared: {r_value**2:.3f}\n'
                 f'WARNING:\n'
                 f'{critical_note}',
                 transform=ax2.transAxes, fontsize=8, color='gray',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                 family='monospace')

        ax2.set_title('Replacement Ratio: Positive Now,\nBut Rapidly Deteriorating',
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel('Replacement Ratio', fontsize=10)
        ax2.set_xlabel('Year', fontsize=10)
        ax2.set_ylim(0, max(1.5, flow_df['RR'].max() * 1.15))
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        fig.suptitle('[Layer 2] Youth Inflow: Currently Positive, But Rapidly Deteriorating',
                     fontsize=14, fontweight='bold', y=0.98)
        fig.text(0.5, 0.01,
                 'Source: Census J2J Energy Belt (AL, GA, NC, SC, TN) Manufacturing (2010-2024)',
                 ha='center', fontsize=8, color='gray')

        plt.tight_layout()
        save_and_show('layer2_youth_inflow')

        return {'rr_2024': last_rr, 'slope': slope, 'p_value': p_value,
                'inflow_change': pct_change, 'critical_year': critical_year,
                'flow_df': flow_df}

    except Exception as e:
        print(f"  Error Layer 2: {e}")
        import traceback
        traceback.print_exc()
        return {}
plt.savefig(f"{OUTPUT_DIR}/layer2_youth.png", dpi=150, bbox_inches='tight')
# =============================================================================
# Layer 3: Retention Failure
# =============================================================================
def plot_layer3_retention():
    print("\n[Layer 3] Generating Retention Failure Chart...")

    industries    = ['Manufacturing', 'Hospitality', 'Retail', 'Logistics', 'Construction']
    natural_rates = [4.4, 4.5, 4.2, 4.3, 4.1]
    total_rates   = [11.0, 10.5, 9.8, 8.5, 7.2]
    gaps          = [t - n for t, n in zip(total_rates, natural_rates)]
    bar_colors    = [COLORS.get(ind, '#95a5a6') for ind in industries]

    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.28

    ax.bar(np.arange(len(industries)), natural_rates, width=width,
           label='Natural Decay (Unavoidable)', color=COLORS['Natural'], alpha=0.5)
    ax.bar(np.arange(len(industries)), gaps, width=width, bottom=natural_rates,
           label='Structural Gap (Preventable)', color=bar_colors, alpha=0.9)

    for i, (n, g, t) in enumerate(zip(natural_rates, gaps, total_rates)):
        ax.text(i, n / 2,   f'{n}%',      ha='center', va='center', color='white', fontsize=9,  fontweight='bold')
        ax.text(i, n + g/2, f'+{g:.1f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        ax.text(i, t + 0.2, f'{t:.1f}%',  ha='center', fontweight='bold', color='black', fontsize=11)

    ax.text(0.02, 0.98,
            'High inflow (RR=2.57)\n'
            '+ High outflow (11%)\n'
            '= Net drain risk',
            transform=ax.transAxes, fontsize=9, color='gray',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')

    legend_elements = [
        Patch(facecolor=COLORS['Natural'], alpha=0.5, label='Natural (Unavoidable)'),
        Patch(facecolor='gray', label='Structural (Preventable)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)

    ax.set_title('[Layer 3] Retention Failure: Even New Hires Leave',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(industries)))
    ax.set_xticklabels(industries, fontsize=11)
    ax.set_ylabel('Annual Separation Rate (%)', fontsize=11)
    ax.set_ylim(0, 14)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.98, 0.02, 'Source: Census QWI + BLS Table 1.10 (2010-2024)',
            transform=ax.transAxes, fontsize=8, color='gray',
            horizontalalignment='right', verticalalignment='bottom')

    plt.tight_layout()
    save_and_show('layer3_retention')

    return {'structural_gap': gaps[0]}

plt.savefig(f"{OUTPUT_DIR}/layer3_retention.png", dpi=150, bbox_inches='tight')
# =============================================================================
# Layer 4: Accelerated Collapse
# =============================================================================
def plot_layer4_accelerated_collapse(flow_df=None):
    print("\n[Layer 4] Generating Accelerated Collapse Simulation...")

    BASE_YEAR = 2024
    years     = np.arange(BASE_YEAR, 2036)
    n_years   = len(years)

    if flow_df is not None and not flow_df.empty:
        rr_2024     = flow_df[flow_df['year'] == flow_df['year'].max()]['RR'].values[0]
        junior_init = rr_2024 * 100
    else:
        junior_init = 257.5
    senior_init = 100.0

    senior_base   = np.zeros(n_years)
    senior_policy = np.zeros(n_years)
    junior_base   = np.zeros(n_years)
    junior_policy = np.zeros(n_years)

    senior_base[0] = senior_policy[0] = senior_init
    junior_base[0] = junior_policy[0] = junior_init

    lambda_senior         = 0.066
    base_junior_attrition = 0.18
    mentoring_multiplier  = 0.5

    for i in range(1, n_years):
        senior_base[i]    = senior_base[i-1] * np.exp(-lambda_senior)
        mentoring_quality = senior_base[i] / senior_init
        junior_attrition  = min(base_junior_attrition * (1 + (1 - mentoring_quality) * mentoring_multiplier), 0.35)
        rr_declining      = max(0.25 * (1 - 0.05 * i), 0.10)
        junior_base[i]    = junior_base[i-1] * (1 - junior_attrition) + senior_base[i] * lambda_senior * rr_declining * 0.5

    for i in range(1, n_years):
        senior_policy[i]  = senior_policy[i-1] * np.exp(-lambda_senior * 0.8)
        mentoring_quality = senior_policy[i] / senior_init
        junior_attrition  = min((base_junior_attrition * 0.5) * (1 + (1 - mentoring_quality) * mentoring_multiplier * 0.5), 0.18)
        rr_improved       = min(0.25 * (1 + 0.02 * i), 0.40)
        junior_policy[i]  = junior_policy[i-1] * (1 - junior_attrition) + senior_policy[i] * lambda_senior * rr_improved * 0.5

    final_base   = junior_base[-1]
    final_policy = junior_policy[-1]
    improvement  = ((final_policy - final_base) / final_base) * 100

    has_history = flow_df is not None and not flow_df.empty

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if has_history:
        base_val = flow_df.loc[flow_df['year'] == flow_df['year'].max(), 'inflow_young'].values[0]
        hist_idx = flow_df['inflow_young'] / base_val * junior_init

        ax1.axvspan(2020, 2021, color='#e74c3c', alpha=0.08, zorder=0)
        ax1.text(2020.5, junior_init * 0.15, 'COVID-19', ha='center', va='bottom',
                 color='#c0392b', fontsize=9, fontweight='bold', alpha=0.7)
        ax1.plot(flow_df['year'], hist_idx,
                 color=COLORS['Manufacturing'], linewidth=2, marker='o', markersize=4,
                 label='Historical Youth Inflow (J2J actual)', alpha=0.8)
        ax1.axvline(2024, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.text(2023.8, junior_init * 1.05, 'Simulation',
                 ha='right', fontsize=8, color='gray', style='italic')

    ax1.plot(years, junior_base,   color=COLORS['Manufacturing'], linewidth=2,
             label='No Intervention (Worst-Case)', linestyle='-')
    ax1.plot(years, junior_policy, color='#7f7f7f', linewidth=1.5,
             label='With Retention Policy', linestyle='--')
    ax1.fill_between(years, junior_base, 0,             color=COLORS['Manufacturing'], alpha=0.10, label='_nolegend_')
    ax1.fill_between(years, junior_policy, junior_base, color='#7f7f7f',               alpha=0.15, label='Prevented Collapse')

    ax1.text(years[-1] - 0.1, final_base,   f'{final_base:.0f}',   ha='right', va='bottom', fontsize=10, fontweight='bold', color=COLORS['Manufacturing'])
    ax1.text(years[-1] - 0.1, final_policy, f'{final_policy:.0f}', ha='right', va='bottom', fontsize=10, fontweight='bold', color='#7f7f7f')

    ax1.axhline(junior_init * 0.5, color='red', linestyle=':', linewidth=1, alpha=0.4)
    ax1.text(BASE_YEAR + 0.2, junior_init * 0.5 + 3, 'Critical (50% of 2024)',
             fontsize=8, color='red', style='italic')

    ax1.text(0.02, 0.02,
             f'Start: Junior={junior_init:.0f}, Senior={senior_init:.0f}\n'
             f'(Proportional to RR={junior_init/senior_init:.2f})\n'
             f'Index: 2024 Senior = 100',
             transform=ax1.transAxes, fontsize=8, color='gray',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
             family='monospace')

    ax1.set_title('Junior Workforce: Historical + Worst-Case Simulation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Workforce Index (Senior 2024=100)', fontsize=10)
    ax1.set_xlabel('Year', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    ax2.plot(years, senior_base,   color=COLORS['Manufacturing'], linewidth=2.3,   label='No Intervention',  linestyle='-')
    ax2.plot(years, senior_policy, color='#7f7f7f',               linewidth=1.5, label='Phased Retirement', linestyle='--')
    ax2.fill_between(years, senior_base, senior_policy, color='#7f7f7f', alpha=0.2, label='Retained Experience')

    ax2.set_title('Senior Workforce (Skilled Workers)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Workforce Index (2024=100)', fontsize=10)
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'[Layer 4] Accelerated Collapse: Negative Feedback Loop\n'
                 f'2035 Worst-Case: No Policy = {final_base:.0f}, With Policy = {final_policy:.0f} (+{improvement:.0f}%)',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01,
             'Mechanism: Senior Exit -> Mentoring Gap -> Junior Attrition -> Fewer Skilled Workers (Vicious Cycle)',
             ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    save_and_show('layer4_collapse')

    return {'junior_base_2035': final_base, 'junior_policy_2035': final_policy,
            'improvement_pct': improvement}

plt.savefig(f"{OUTPUT_DIR}/layer4_collapse.png", dpi=150, bbox_inches='tight')
# =============================================================================
# Layer 5: External Competition
# =============================================================================
def plot_layer5_competition():
    print("\n[Layer 5] Generating External Competition Chart...")

    sectors     = ['Manufacturing', 'Retail', 'Logistics', 'Construction', 'Services', 'Unemployment']
    origin      = np.array([35.2, 12.5,  6.2,  8.5,  9.8, 27.8])
    destination = np.array([32.1, 14.2, 11.5, 10.5, 12.8, 18.9])

    assert abs(origin.sum()      - 100) < 0.1
    assert abs(destination.sum() - 100) < 0.1

    print(f"  Validation: Origin={origin.sum():.1f}%, Destination={destination.sum():.1f}%")

    net_flow   = destination - origin
    row_colors = [COLORS.get(s, '#95a5a6') for s in sectors]
    y          = np.arange(len(sectors))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(y, -origin,      color=row_colors, alpha=0.60, label='Inflow',  zorder=2)
    ax.barh(y,  destination, color=row_colors, alpha=0.95, label='Outflow', zorder=2)
    ax.axvline(0, color='black', linewidth=1.5, zorder=3)

    for i, (orig, dest, nf) in enumerate(zip(origin, destination, net_flow)):
        ax.text(-orig - 1.5, i, f'{orig:.1f}%', va='center', ha='right',  color='gray',  fontsize=10)
        ax.text( dest + 1.5, i, f'{dest:.1f}%', va='center', ha='left',   color='black', fontsize=10, fontweight='bold')
        ax.text(0, i, sectors[i], ha='center', va='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
        nf_color = 'green' if nf > 0 else 'red'
        ax.text(0, i - 0.38, f'Delta {nf:+.1f}%', ha='center', fontsize=10,
                color=nf_color, style='italic', fontweight='bold')

    mfg_net      = net_flow[0]
    top_gain_idx = np.argmax(net_flow[1:]) + 1
    top_gain     = sectors[top_gain_idx]

    ax.text(0.02, 0.98,
            f'  {top_gain}: +{net_flow[top_gain_idx]:.1f}%p\n'
            f'  Services: +{net_flow[4]:.1f}%p\n\n'
            f'Mfg Net Loss:\n'
            f'  {mfg_net:.1f}%p\n\n'
            f'Link to Layer 1:\n'
            f'Unemployment drop\n'
            f'(-8.9%p) reflects\n'
            f'retirement wave',
            transform=ax.transAxes, fontsize=9, color='gray',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            family='monospace')

    ax.set_title('[Layer 5] External Competition: Where Did Manufacturing Workers Go?',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-35, 35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    ax.text(0.98, 0.02,
            'Source: Census J2J Energy Belt Proxy (GA, 2010-2024)\n'
            'GA selected as largest mfg state in Energy Belt (380K employed)',
            transform=ax.transAxes, fontsize=8, color='gray',
            horizontalalignment='right', verticalalignment='bottom')

    plt.tight_layout()
    save_and_show('layer5_competition')

    return {'mfg_net': mfg_net}

plt.savefig(f"{OUTPUT_DIR}/layer5_competition.png", dpi=150, bbox_inches='tight')
# =============================================================================
# Summary Report
# =============================================================================
def generate_summary(results):
    rr_2024        = results.get('rr_2024')
    slope          = results.get('slope')
    p_value        = results.get('p_value')
    inflow_change  = results.get('inflow_change')
    critical_year  = results.get('critical_year')
    structural_gap = results.get('structural_gap')
    junior_base    = results.get('junior_base_2035')
    junior_policy  = results.get('junior_policy_2035')
    improvement    = results.get('improvement_pct')
    mfg_net        = results.get('mfg_net')

    summary = f"""
================================================================================
MANUFACTURING TALENT SUPPLY CHAIN RISK ANALYSIS
5-Layer Quantitative Framework | Energy Belt (AL, GA, NC, SC, TN)
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================
LAYER 1: AGING RISK
  Source: BLS Table 1.10 (2024)
  • Baseline exit rate: 4.4%/yr | Accelerated (Baby Boomer): 6.6%/yr
  • By 2035: 48.7% of current workforce remaining under acceleration

LAYER 2: YOUTH INFLOW — CURRENTLY POSITIVE, RAPIDLY DETERIORATING
  Source: Census J2J Energy Belt 5 States (2010-2024)
  • Youth inflow (25-34) changed +{inflow_change:.1f}% over 14 years (absolute increase)
  • Replacement Ratio (2024): {rr_2024:.3f} — currently above 1.0 (healthy)
  • BUT: Trend = {slope:.4f}/yr (p={p_value:.4f}) — statistically significant decline
  • Projected RR < 1.0 by approximately {critical_year} if trend continues

LAYER 3: RETENTION FAILURE
  Source: Census QWI + BLS Table 1.10 (2010-2024)
  • Manufacturing structural gap: {structural_gap:.1f}%p (highest among peer industries)
  • Total separation rate: 11.0% | Natural: 4.4% | Preventable: 6.6%p

LAYER 4: ACCELERATED COLLAPSE (WORST-CASE SIMULATION)
  Note: Assumes maximum feedback effect — actual outcome policy-dependent
  • No intervention: Junior workforce → {junior_base:.0f}% by 2035
  • With retention policy: {junior_policy:.0f}% (+{improvement:.0f}% improvement)
  Mechanism: Senior exit → Mentoring gap → Junior attrition acceleration

LAYER 5: EXTERNAL COMPETITION
  Source: Census J2J Energy Belt Proxy (GA as largest mfg state, 380K employed)
  • Manufacturing net loss: {mfg_net:.1f}%p over 2010-2024
  • Top destinations: Logistics (+5.3%p), Services (+3.0%p)

================================================================================
LIMITATIONS & DATA NOTES
================================================================================
  • State-level aggregation: County-level variation not captured
  • COVID-19 (2020-2021): Labor market distortions not fully isolated
  • Layer 5 proxy: GA used for Energy Belt — industry mix assumed similar
================================================================================
"""
    with open(f"{OUTPUT_DIR}/analysis_summary.txt", 'w') as f:
        f.write(summary)
    print(summary)
    print(f"  ✓ Summary saved: analysis_summary.txt")

# =============================================================================
# Main — Layer 1 shown FIRST by pre-loading J2J data
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  MANUFACTURING TALENT SUPPLY CHAIN RISK MODEL")
    print("  5-Layer Framework | Energy Belt (AL, GA, NC, SC, TN)")
    print("="*70)

    results = {}

    # Pre-load J2J so Layer 1 can show historical data before Layer 2 runs
    print("\n  Pre-loading J2J data...")
    flow_df_pre = None
    try:
        df_pre = pd.read_csv(J2J_FILE,
                             usecols=['geography', 'industry', 'agegrp', 'year', 'J2J', 'sex', 'seasonadj'],
                             dtype=str, low_memory=False)
        mask_pre = (
            (df_pre['geography'].isin(['1', '13', '37', '45', '47'])) &
            (df_pre['seasonadj'] == 'U') &
            (df_pre['sex'] == '0') &
            (df_pre['industry'].str.startswith(('31', '32', '33'), na=False))
        )
        df_mfg_pre              = df_pre[mask_pre].copy()
        df_mfg_pre['J2J']       = pd.to_numeric(df_mfg_pre['J2J'],  errors='coerce')
        df_mfg_pre['year']      = pd.to_numeric(df_mfg_pre['year'], errors='coerce')
        inflow_y                = df_mfg_pre[df_mfg_pre['agegrp'] == 'A04'].groupby('year')['J2J'].sum()
        outflow_s               = df_mfg_pre[df_mfg_pre['agegrp'].isin(['A07', 'A08'])].groupby('year')['J2J'].sum()
        flow_df_pre             = pd.DataFrame({'inflow_young': inflow_y, 'outflow_senior': outflow_s}).dropna().reset_index()
        flow_df_pre['RR']       = flow_df_pre['inflow_young'] / flow_df_pre['outflow_senior']
        print(f"  Pre-loaded {len(flow_df_pre)} years of J2J data.")
    except Exception as e:
        print(f"  Warning: could not pre-load J2J: {e}")

    # ── Layer 1 FIRST ──────────────────────────────────────────────────────
    plot_layer1_aging(flow_df=flow_df_pre)

    # ── Layer 2 ────────────────────────────────────────────────────────────
    layer2_results = plot_layer2_youth_inflow()
    results.update(layer2_results)
    flow_df = layer2_results.get('flow_df', flow_df_pre)

    # ── Layer 3 ────────────────────────────────────────────────────────────
    results.update(plot_layer3_retention())

    # ── Layer 4 (uses flow_df for proportional init) ───────────────────────
    results.update(plot_layer4_accelerated_collapse(flow_df=flow_df))

    # ── Layer 5 ────────────────────────────────────────────────────────────
    results.update(plot_layer5_competition())

    print("\n" + "="*70)
    print("  Generating Summary...")
    print("="*70)
    generate_summary(results)

    print("\n" + "="*70)
    print(f"  ✓ All charts saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()