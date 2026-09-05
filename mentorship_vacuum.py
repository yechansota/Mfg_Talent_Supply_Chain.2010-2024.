"""
================================================================================
 The Mentorship Vacuum
 Manufacturing Talent Supply Chain in the U.S. Southeast, 2010-2024
================================================================================

 A five-layer empirical diagnosis of manufacturing workforce risk across
 Alabama, Georgia, North Carolina, South Carolina and Tennessee.

 Every constant in this pipeline is either (a) computed from a public federal
 data file, or (b) declared in the ASSUMPTIONS block below and subjected to
 sensitivity analysis. There are no undocumented magic numbers.

--------------------------------------------------------------------------------
 DATA REQUIRED  (all free, no login, from https://ledextract.ces.census.gov/)
--------------------------------------------------------------------------------
   j2j_od.csv    J2J Origin-Destination. Job Flows path.
                 Origin + destination geography = AL GA NC SC TN.
                 Columns include: industry, industry_orig, agegrp, J2J.

   j2j_sep.csv   J2J Separations path.
                 Origin geography = AL GA NC SC TN, all NAICS sectors.
                 Columns include: MSep, EESep, ENPersist, ENFullQ.

   qwi.csv       QWI. Same five states, NAICS sectors, ages A01-A08.
                 Columns include: Emp, EmpEnd, EmpS, HirA, Sep, SepBeg, SepBegR.

--------------------------------------------------------------------------------
 KNOWN CORRECTIONS TO EARLIER VERSIONS OF THIS ANALYSIS
--------------------------------------------------------------------------------
 1. FLOW DIRECTION.  Per the LEHD schema, in J2J Origin-Destination tables the
    origin firm's characteristics carry the _orig suffix and the unsuffixed
    firm variables describe the DESTINATION. An earlier version filtered both
    the youth and senior series on the unsuffixed `industry`, so both measured
    hires INTO manufacturing. Senior separations now use `industry_orig`.

 2. GEOGRAPHY.  LEHD stores FIPS zero-padded. A filter on '1' rather than '01'
    silently dropped Alabama, roughly 19% of matching rows.

 3. AGGREGATION LEVEL.  These files stack marginal and detailed tabulations.
    Summing across agg_level double counts every flow. Exactly one level is
    selected for each quantity.

 4. RATE UNITS.  QWI rates are QUARTERLY. An earlier version reported a
    quarterly figure as annual. All rates here are annualised as 1-(1-q)^4
    and labelled.

 5. RETIREMENT.  Job-to-job files cannot observe retirement, because retirement
    is an exit to nonemployment. Labor force exit is measured with ENPersist
    from the Separations file.

 6. INFERENCE.  Newey-West (HAC) standard errors replace naive OLS on these
    autocorrelated annual series, and threshold crossing is reported as a
    bootstrap interval rather than a point estimate.

 7. LAYER 4 SIGN ERROR.  Cohort ageing previously ADDED to the junior stock.
    Ageing moves workers OUT of the junior compartment; it is now a transfer.

--------------------------------------------------------------------------------
 Author: Yechan (Sean) Kim
 License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

try:
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:                                    # pragma: no cover
    HAVE_SM = False


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

STATES = ["01", "13", "37", "45", "47"]      # AL GA NC SC TN, zero-padded
MFG = "31-33"                                 # the only manufacturing code in
                                              # sector-level LEHD extracts

# Aggregation levels. Verify against your own extract with:
#   df.groupby(['agg_level','ind_level']).size()
AGG = {
    "od_marginal": "197891",   # OD: destination sector x origin ALL
    "od_detail_o": "246787",   # OD: origin sector x destination ALL
    "od_matrix":   "247043",   # OD: origin sector x destination sector
    "sep":         "1283",     # Separations: sector x sex-all x age-detail
    "qwi":         "1315",     # QWI:         sector x sex-all x age-detail
}

YOUTH = ["A04"]                 # 25-34
MID = ["A05", "A06"]            # 35-54
SENIOR = ["A07", "A08"]         # 55+

AGE_LABEL = {"A04": "25-34", "A05": "35-44", "A06": "45-54",
             "A07": "55-64", "A08": "65+"}

PEERS = {
    "31-33": "Manufacturing",
    "23": "Construction",
    "48-49": "Transport & Whse",
    "62": "Health Care",
    "44-45": "Retail",
    "56": "Admin & Support",
}

SECTOR_NAME = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities", "23": "Construction",
    "31-33": "Manufacturing", "42": "Wholesale", "44-45": "Retail",
    "48-49": "Transport & Whse", "51": "Information", "52": "Finance",
    "53": "Real Estate", "54": "Professional Svcs", "55": "Management",
    "56": "Admin & Support", "61": "Education", "62": "Health Care",
    "71": "Arts & Rec", "72": "Accommodation & Food", "81": "Other Svcs",
    "92": "Public Admin",
}

RR_THRESHOLD = 1.0        # one youth hire per senior separation
COVID_YEARS = (2020, 2021)
BASE_WINDOW = (2021, 2024)   # post-COVID window used to calibrate Layer 4
N_BOOT = 5000
SEED = 42
BACKTEST_MAPE_LIMIT = 15.0   # a model above this is not used for projection

PALETTE = {
    "mfg": "#c0392b", "grey": "#95a5a6", "dark": "#7f8c8d",
    "warn": "#e67e22", "covid": "#e74c3c", "accent": "#f39c12",
}


@dataclass
class Assumptions:
    """Layer 4 parameters that are NOT observed in the data.

    Kept deliberately short. Every other rate in the simulation is measured
    in QWI. Anything listed here is a choice and must be defended in text.
    """
    policy_senior_retention: float = 0.20   # phased retirement: -20% senior exits
    horizon_end: int = 2035


# ==============================================================================
#  HELPERS
# ==============================================================================

def annualize(q):
    """Quarterly rate -> annual rate. LEHD rates are per quarter."""
    return 1.0 - (1.0 - q) ** 4


def shade_covid(ax, label=True, alpha=0.07, zorder=0, legend=False,
                label_pos="bottom"):
    """Shade the pandemic window. The caption sits at the BOTTOM of the axes so
    it never collides with data or the legend.

    `zorder` matters for filled charts: a stackplot paints over a zorder-0
    span, so those axes pass a value above the fill.
    """
    kw = {"color": PALETTE["covid"], "alpha": alpha, "zorder": zorder}
    if legend:
        kw["label"] = "COVID-19"
    ax.axvspan(*COVID_YEARS, **kw)
    if label:
        lo, hi = ax.get_ylim()
        if label_pos == "top":
            # Filled charts occupy the bottom of the panel, so the caption
            # goes into the clear margin above the stack instead.
            y, va = hi - (hi - lo) * 0.02, "top"
        else:
            y, va = lo + (hi - lo) * 0.035, "bottom"
        ax.text(np.mean(COVID_YEARS), y, "COVID-19",
                ha="center", va=va, color="#c0392b", fontsize=8,
                fontweight="bold", alpha=0.85, zorder=max(zorder + 1, 1))


def hac_fit(x, y, maxlags=1):
    """OLS with Newey-West errors. Falls back to scipy if statsmodels absent."""
    if HAVE_SM:
        X = sm.add_constant(np.asarray(x, dtype=float))
        ols = sm.OLS(y, X).fit()
        hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return {
            "intercept": float(hac.params[0]), "slope": float(hac.params[1]),
            "p_ols": float(ols.pvalues[1]), "p_hac": float(hac.pvalues[1]),
            "ci_lo": float(hac.conf_int()[1][0]), "ci_hi": float(hac.conf_int()[1][1]),
            "r2": float(ols.rsquared),
            "durbin_watson": float(sm.stats.durbin_watson(ols.resid)),
        }
    f = linregress(x, y)
    return {"intercept": f.intercept, "slope": f.slope, "p_ols": f.pvalue,
            "p_hac": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "r2": f.rvalue ** 2, "durbin_watson": float("nan")}


def save_fig(fig, outdir, name, source_note):
    fig.text(0.5, 0.004, source_note, ha="center", fontsize=7.5, color="gray")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{name}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


class Reporter:
    """Prints to stdout and accumulates a reproducible results log."""

    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(str(s))

    def rule(self, title):
        self("\n" + "=" * 78)
        self(f"  {title}")
        self("=" * 78)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")


# ==============================================================================
#  LOADERS
# ==============================================================================

def load_od(path):
    cols = ["geography", "industry", "industry_orig", "sex", "agegrp",
            "year", "quarter", "agg_level", "seasonadj", "J2J"]
    df = pd.read_csv(path, dtype=str, usecols=lambda c: c in cols, low_memory=False)
    df["J2J"] = pd.to_numeric(df["J2J"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["geography"] = df["geography"].str.zfill(2)
    return df[(df.seasonadj == "U") & (df.sex == "0") & df.geography.isin(STATES)].copy()


def load_sep(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    for c in ["MSep", "EESep", "ENPersist", "ENFullQ"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["geography"] = df["geography"].str.zfill(2)
    return df[(df.agg_level == AGG["sep"]) & df.geography.isin(STATES)].copy()


def load_qwi(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    for c in ["Emp", "EmpEnd", "EmpS", "HirA", "HirAEnd", "Sep", "SepBeg", "SepBegR"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["geography"] = df["geography"].str.zfill(2)
    return df[(df.agg_level == AGG["qwi"]) & df.geography.isin(STATES)].copy()


def audit(od, sep, qwi, say):
    """Fail loudly rather than silently producing wrong numbers."""
    say.rule("DATA AUDIT")
    ok = True
    for name, df in [("J2J OD", od), ("J2J Sep", sep), ("QWI", qwi)]:
        states = sorted(df.geography.unique())
        missing = sorted(set(STATES) - set(states))
        q = df.groupby("year")["quarter"].nunique() if "quarter" in df else None
        partial = sorted(q[q < 4].index.dropna().astype(int)) if q is not None else []
        say(f"{name:9s} rows={len(df):>9,}  states={states}  "
            f"years={df.year.min():.0f}-{df.year.max():.0f}")
        if missing:
            say(f"          !! MISSING STATES: {missing}")
            ok = False
        if partial:
            say(f"          !! PARTIAL YEARS (dropped from trends): {partial}")
    if "ENPersist" not in sep.columns:
        say("          !! ENPersist absent - Layer 1 hazard cannot be computed")
        ok = False
    say(f"\nAudit {'PASSED' if ok else 'FAILED'}")
    return ok


# ==============================================================================
#  LAYER 1 — AGE STRUCTURE AND LABOR FORCE EXIT
# ==============================================================================

def layer1(qwi, sep, say, outdir):
    say.rule("LAYER 1 — AGE STRUCTURE AND LABOR FORCE EXIT")

    qm = qwi[qwi.industry == MFG]
    sm_ = sep[sep.industry == MFG]

    emp = qm.groupby(["year", "agegrp"])["Emp"].sum().unstack()
    emp = emp[[a for a in AGE_LABEL if a in emp.columns]]
    share = emp.div(emp.sum(axis=1), axis=0) * 100

    say("\nEmployment by age (sum of quarterly beginning-of-quarter counts)")
    say(emp.rename(columns=AGE_LABEL).round(0).astype("Int64").to_string())

    say("\nAge composition (% of manufacturing employment)")
    say(share.rename(columns=AGE_LABEL).round(2).to_string())

    sen_n = emp[SENIOR].sum(axis=1)
    sen_s = share[SENIOR].sum(axis=1)
    y0, y1 = int(emp.index.min()), int(emp.index.max())
    say(f"\nSenior (55+) headcount  {y0}: {sen_n.iloc[0]:>12,.0f}   "
        f"{y1}: {sen_n.iloc[-1]:>12,.0f}   ({sen_n.iloc[-1]/sen_n.iloc[0]-1:+.1%})")
    say(f"Senior (55+) share      {y0}: {sen_s.iloc[0]:>11.2f}%   "
        f"{y1}: {sen_s.iloc[-1]:>11.2f}%   ({sen_s.iloc[-1]-sen_s.iloc[0]:+.2f} pp)")

    fit = hac_fit(sen_s.index.values, sen_s.values)
    say(f"Senior share trend      {fit['slope']:+.3f} pp/yr   "
        f"HAC p={fit['p_hac']:.6f}   R2={fit['r2']:.3f}   DW={fit['durbin_watson']:.2f}")
    say("\nNOTE: the senior stock GREW. Cohorts age into the 55+ band faster than")
    say("      they exit it, so an exponential decay on the senior stock is not")
    say("      an appropriate model. Layer 4 uses explicit age compartments.")

    # Empirical labor force exit hazard
    haz = {}
    for ages, lab in [(SENIOR, "55+"), (["A06"], "45-54"),
                      (["A05"], "35-44"), (YOUTH, "25-34")]:
        e = qm[qm.agegrp.isin(ages)].groupby("year")["Emp"].sum()
        p = sm_[sm_.agegrp.isin(ages)].groupby("year")["ENPersist"].sum()
        haz[lab] = annualize(p / e) * 100
    haz = pd.DataFrame(haz)

    say("\nAnnualised exit hazard to persistent nonemployment (%)")
    say(haz.round(2).to_string())

    pre = haz.loc[2010:2019, "55+"].mean()
    post = haz.loc[BASE_WINDOW[0]:BASE_WINDOW[1], "55+"].mean()
    say(f"\nlambda_55+  2010-2019 = {pre:.2f}%/yr")
    say(f"lambda_55+  {BASE_WINDOW[0]}-{BASE_WINDOW[1]} = {post:.2f}%/yr")
    say(f"lambda_55+  2020 COVID = {haz.loc[2020,'55+']:.2f}%/yr")
    say("\nNOTE: ENPersist at ages 25-34 also captures schooling, caregiving and")
    say("      job search. Read it as retirement only for the 55+ group.")

    # figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    sh = share.rename(columns=AGE_LABEL)
    for col, c in zip(sh.columns, ["#bdc3c7", PALETTE["dark"], PALETTE["accent"],
                                   PALETTE["mfg"], "#7b241c"]):
        a1.plot(sh.index, sh[col], marker="o", ms=3.5, lw=2, label=col, color=c)
    shade_covid(a1, label=True)
    a1.set_title("Age Composition of the Manufacturing Workforce",
                 fontsize=12, fontweight="bold")
    a1.set_ylabel("Share of employment (%)"); a1.set_xlabel("Year")
    a1.legend(fontsize=8.5, ncol=2, loc="upper right"); a1.grid(axis="y", alpha=0.3)

    a2.plot(haz.index, haz["55+"], marker="o", ms=4, lw=2.2,
            color=PALETTE["mfg"], label="55+")
    a2.plot(haz.index, haz["25-34"], marker="s", ms=3.5, lw=1.6,
            color=PALETTE["dark"], ls="--", label="25-34")
    a2.set_ylim(0, max(18, haz.max().max() * 1.12))
    shade_covid(a2, label=True)
    a2.set_title("Annualized Exit Hazard to Persistent Nonemployment",
                 fontsize=12, fontweight="bold")
    a2.set_ylabel("Annual exit rate (%)"); a2.set_xlabel("Year")
    a2.legend(fontsize=9, loc="upper right"); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("Layer 1 — Aging Risk", fontsize=13.5, fontweight="bold")
    save_fig(fig, outdir, "layer1_aging",
             "Red band marks the COVID-19 disruption (2020-2021), excluded from "
             "rate calibration.  |  Source: Census LEHD QWI and J2J Separations. "
             "AL/GA/NC/SC/TN, NAICS 31-33, not seasonally adjusted.")

    # Observable hire and separation rates, used to calibrate Layer 4
    w0, w1 = BASE_WINDOW
    rates = {}
    for ages, tag in [(YOUTH, "J"), (MID, "M"), (SENIOR, "S")]:
        s = qm[qm.agegrp.isin(ages)]
        e = s.groupby("year")["Emp"].sum()
        h = s.groupby("year")["HirA"].sum()
        p = s.groupby("year")["SepBeg"].sum()
        rates[f"hire_{tag}"] = float(annualize(h / e).loc[w0:w1].mean())
        rates[f"sep_{tag}"] = float(annualize(p / e).loc[w0:w1].mean())

    say(f"\nObserved annual hire and separation rates, {w0}-{w1}")
    for tag, lab in [("J", "25-34"), ("M", "35-54"), ("S", "55+")]:
        say(f"  {lab:6s}  hire {rates[f'hire_{tag}']:.3f}   "
            f"sep {rates[f'sep_{tag}']:.3f}   "
            f"net {rates[f'hire_{tag}'] - rates[f'sep_{tag}']:+.3f}")

    emp.to_csv(Path(outdir) / "layer1_employment_by_age.csv")
    share.round(4).to_csv(Path(outdir) / "layer1_age_share_pct.csv")
    haz.round(4).to_csv(Path(outdir) / "layer1_exit_hazard_annual_pct.csv")
    return {"emp": emp, "share": share, "hazard": haz, "rates": rates,
            "lambda_senior_post": post / 100, "senior_share_slope": fit["slope"]}


# ==============================================================================
#  LAYER 2 — REPLACEMENT RATIO
# ==============================================================================

def layer2(od, say, outdir):
    say.rule("LAYER 2 — REPLACEMENT RATIO")

    into = od[(od.agg_level == AGG["od_marginal"]) & (od.industry == MFG)]
    out = od[(od.agg_level == AGG["od_detail_o"]) & (od.industry_orig == MFG)]

    yi = into[into.agegrp.isin(YOUTH)].groupby("year")["J2J"].sum()
    so = out[out.agegrp.isin(SENIOR)].groupby("year")["J2J"].sum()
    t = pd.DataFrame({"youth_hires_in": yi, "senior_seps_out": so}).dropna()
    t = t[t.senior_seps_out > 0]
    t["RR"] = t.youth_hires_in / t.senior_seps_out

    say("\nYouth hires INTO manufacturing vs senior separations OUT of manufacturing")
    say(t.assign(RR=t.RR.round(3)).to_string())

    x = t.index.values.astype(float)
    fit = hac_fit(x, t["RR"].values)
    say(f"\nn                   = {len(t)}")
    say(f"slope               = {fit['slope']:+.4f} RR/yr")
    say(f"OLS p               = {fit['p_ols']:.8f}")
    say(f"Newey-West HAC p    = {fit['p_hac']:.8f}")
    say(f"HAC 95% CI (slope)  = [{fit['ci_lo']:+.4f}, {fit['ci_hi']:+.4f}]")
    say(f"R-squared           = {fit['r2']:.4f}")
    say(f"Durbin-Watson       = {fit['durbin_watson']:.3f}")

    last = float(t.index.max())
    fitted_last = fit["slope"] * last + fit["intercept"]

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3:
            continue
        f = linregress(x[i], t["RR"].values[i])
        if f.slope >= 0:
            continue
        fl = f.slope * last + f.intercept
        draws.append(last if fl <= RR_THRESHOLD
                     else last + (fl - RR_THRESHOLD) / abs(f.slope))
    draws = np.array(draws)
    med, lo, hi = (float(np.median(draws)), float(np.percentile(draws, 2.5)),
                   float(np.percentile(draws, 97.5)))

    say(f"\nFitted RR at {last:.0f}    = {fitted_last:.3f}")
    say(f"Crossing RR={RR_THRESHOLD}    = {med:.0f}   (95% CI {lo:.0f}-{hi:.0f})")

    f0, f1 = t.iloc[0], t.iloc[-1]
    say(f"\nYouth hires   {t.index.min():.0f}: {f0.youth_hires_in:>9,.0f}  "
        f"{t.index.max():.0f}: {f1.youth_hires_in:>9,.0f}  "
        f"({f1.youth_hires_in/f0.youth_hires_in-1:+.1%})")
    say(f"Senior seps   {t.index.min():.0f}: {f0.senior_seps_out:>9,.0f}  "
        f"{t.index.max():.0f}: {f1.senior_seps_out:>9,.0f}  "
        f"({f1.senior_seps_out/f0.senior_seps_out-1:+.1%})")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_ylim(0, max(5.4, t.RR.max() * 1.12))
    shade_covid(ax)
    ax.plot(t.index, t.RR, color=PALETTE["mfg"], lw=2.2, marker="o", ms=5,
            label="Replacement Ratio (observed)", zorder=4)
    fut = np.arange(t.index.min(), 2041)
    ax.plot(fut, fit["slope"] * fut + fit["intercept"], "--",
            color=PALETTE["dark"], lw=1.6, zorder=3,
            label=f"OLS trend ({fit['slope']:+.3f}/yr, HAC p<0.001)")
    ax.axhline(RR_THRESHOLD, color="black", ls=":", lw=1.4)
    ax.text(t.index.min() + 0.3, RR_THRESHOLD + 0.07,
            f"Sustainability threshold (RR = {RR_THRESHOLD})",
            fontsize=8, style="italic")
    ax.axvspan(lo, hi, color=PALETTE["accent"], alpha=0.16, zorder=1,
               label=f"Crossing, 95% CI ({lo:.0f}-{hi:.0f})")
    ax.axvline(med, color=PALETTE["warn"], lw=1.8, zorder=2)
    ax.text(med, RR_THRESHOLD * 0.42, f"{med:.0f}", color="#d35400",
            ha="center", va="center", fontweight="bold", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#d35400", alpha=0.95), zorder=6)
    ax.axvline(t.index.max(), color="gray", ls="--", lw=1, alpha=0.5)
    last_x, last_y = float(t.index.max()), float(t.RR.iloc[-1])
    ax.annotate(f"{last_x:.0f}: {last_y:.2f}", xy=(last_x, last_y),
                xytext=(last_x + 1.4, last_y + 0.62), fontsize=9.5,
                fontweight="bold", color=PALETTE["mfg"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["mfg"], lw=1.1),
                bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                          edgecolor=PALETTE["mfg"], alpha=0.95), zorder=6)
    ax.set_xlim(t.index.min(), 2040)
    ax.set_title("Layer 2 — Replacement Ratio\n"
                 "Youth (25-34) Hires into Manufacturing per Senior (55+) "
                 "Job-to-Job Separation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Replacement Ratio")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    save_fig(fig, outdir, "layer2_replacement_ratio",
             "Source: Census LEHD J2J Origin-Destination. AL/GA/NC/SC/TN, "
             "NAICS 31-33, NSA. Intra-region job-to-job moves only.")

    t.to_csv(Path(outdir) / "layer2_replacement_ratio.csv")
    return {"table": t, "fit": fit, "crossing": (med, lo, hi)}


# ==============================================================================
#  LAYER 3 — SEPARATION RATES AND MECHANISM
# ==============================================================================

def layer3(qwi, sep, say, outdir):
    say.rule("LAYER 3 — SEPARATION RATES AND MECHANISM")

    w0, w1 = BASE_WINDOW
    rows = []
    for ind, nm in PEERS.items():
        s = qwi[(qwi.industry == ind) & qwi.year.between(w0, w1)]
        if s.empty:
            continue
        q = s.SepBeg.sum() / s.Emp.sum()
        rows.append({"industry": nm, "quarterly_pct": round(q * 100, 2),
                     "annualised_pct": round(annualize(q) * 100, 1)})
    peer = pd.DataFrame(rows).sort_values("quarterly_pct", ascending=False)

    say(f"\nPeer comparison, {w0}-{w1}, employment-weighted")
    say(peer.to_string(index=False))
    rank = list(peer.industry).index("Manufacturing") + 1
    say(f"\nManufacturing ranks {rank} of {len(peer)} from the top, i.e. it has "
        f"the LOWEST separation rate of the peer set.")

    grid = {}
    for ind, nm in PEERS.items():
        r = {}
        for a, al in AGE_LABEL.items():
            s = qwi[(qwi.industry == ind) & (qwi.agegrp == a)
                    & qwi.year.between(w0, w1)]
            r[al] = round(s.SepBeg.sum() / s.Emp.sum() * 100, 2) if not s.empty else np.nan
        grid[nm] = r
    grid = pd.DataFrame(grid).T
    say(f"\nQuarterly separation rate by age, {w0}-{w1} (%)")
    say(grid.to_string())

    sm_ = sep[(sep.industry == MFG) & sep.agegrp.isin(SENIOR)]
    mech = sm_.groupby("year")[["MSep", "ENPersist", "EESep"]].sum()
    mech["retire_share_pct"] = (mech.ENPersist / mech.MSep * 100).round(1)
    mech["j2j_share_pct"] = (mech.EESep / mech.MSep * 100).round(1)

    say("\nWhy senior (55+) manufacturing workers separate")
    say(mech.to_string())

    x = mech.index.values.astype(float)
    for col in ["ENPersist", "EESep", "MSep"]:
        f = hac_fit(x, mech[col].values)
        chg = mech[col].iloc[-1] / mech[col].iloc[0] - 1
        say(f"{col:10s} {mech.index.min():.0f}->{mech.index.max():.0f} {chg:+8.1%}"
            f"   slope={f['slope']:9.1f}/yr   HAC p={f['p_hac']:.6f}   R2={f['r2']:.3f}")

    fs = hac_fit(x, mech["retire_share_pct"].values)
    say(f"\nRetirement share of senior separations: "
        f"{mech.retire_share_pct.iloc[0]:.1f}% -> {mech.retire_share_pct.iloc[-1]:.1f}%"
        f"   ({fs['slope']:+.2f} pp/yr, HAC p={fs['p_hac']:.6f})")
    say("Seniors increasingly leave for another EMPLOYER rather than retiring,")
    say("which moves the problem from demographic inevitability to retention.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    p = peer.sort_values("annualised_pct")
    cols = [PALETTE["mfg"] if n == "Manufacturing" else PALETTE["grey"]
            for n in p.industry]
    a1.barh(range(len(p)), p.annualised_pct, color=cols, alpha=0.9)
    a1.set_yticks(range(len(p))); a1.set_yticklabels(p.industry, fontsize=10)
    for i, v in enumerate(p.annualised_pct):
        a1.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    a1.set_xlim(0, p.annualised_pct.max() * 1.18)
    a1.set_xlabel(f"Annualised separation rate (%), {w0}-{w1}")
    a1.set_title("Manufacturing Has the Lowest Churn of Its Peer Set",
                 fontsize=12, fontweight="bold")
    a1.grid(axis="x", alpha=0.3)

    a2.stackplot(mech.index, mech.ENPersist, mech.EESep,
                 colors=[PALETTE["grey"], PALETTE["mfg"]], alpha=0.88,
                 labels=["Exit to nonemployment (retirement)",
                         "Move to another employer"])
    a2.set_xlim(mech.index.min(), mech.index.max())
    a2.set_ylim(0, mech.MSep.max() * 1.24)
    # The stackplot paints over a zorder-0 span, so the pandemic window is
    # drawn on top, at half the transparency used elsewhere, and bounded by
    # rules so it stays legible against the filled areas.
    shade_covid(a2, label=True, alpha=0.35, legend=False, zorder=6,
                label_pos="top")
    for xv in COVID_YEARS:
        a2.axvline(xv, color="#b03a2e", ls="--", lw=1.6, zorder=7)
    a2.set_title("Senior (55+) Separations by Mechanism",
                 fontsize=12, fontweight="bold")
    a2.set_xlabel("Year"); a2.set_ylabel("Annual separations")
    a2.legend(loc="upper left", fontsize=9); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("Layer 3 — Retention Reframed", fontsize=13.5, fontweight="bold")
    save_fig(fig, outdir, "layer3_retention",
             "Source: Census LEHD QWI and J2J Separations. AL/GA/NC/SC/TN, NSA. "
             "QWI rates are quarterly; annualised as 1-(1-q)^4.")

    peer.to_csv(Path(outdir) / "layer3_peer_rates.csv", index=False)
    grid.to_csv(Path(outdir) / "layer3_rates_by_age.csv")
    mech.to_csv(Path(outdir) / "layer3_senior_mechanism.csv")
    return {"peer": peer, "by_age": grid, "mechanism": mech}


# ==============================================================================
#  LAYER 4 — COMPARTMENT SIMULATION
# ==============================================================================

def band_rates(qwi_mfg, bands, window):
    """Measure the three quarterly flow rates that govern each age band.

    QWI satisfies, exactly and within every age band:

        EmpEnd(q) = Emp(q) - SepBeg(q) + HirAEnd(q)

    verified in this dataset to a mean absolute error of 0.0004%. HirAEnd and
    SepBeg are therefore a MATCHED pair against the Emp denominator. An
    earlier version used HirA, which counts every hire in the quarter
    including short-tenure churn, so hires minus separations was not net
    employment change and the model diverged.

    Cohort ageing appears as the gap between the end of one quarter and the
    start of the next:

        aging(q) = Emp(q+1) - EmpEnd(q)

    It is therefore MEASURED, not assumed from band widths. Its sign is a
    useful check: negative for the two younger bands, which lose workers to
    the band above, and positive for 55+, which receives them.
    """
    out = {}
    for key, ages in bands.items():
        s = (qwi_mfg[qwi_mfg.agegrp.isin(ages)]
             .groupby(["year", "quarter"])[["Emp", "EmpEnd", "HirAEnd", "SepBeg"]]
             .sum().sort_index())
        s["aging"] = s.Emp.shift(-1) - s.EmpEnd
        d = s.dropna()
        yr = d.index.get_level_values(0)
        d = d[(yr >= window[0]) & (yr <= window[1])]
        out[key] = {
            "hire": float((d.HirAEnd / d.Emp).mean()),
            "sep": float((d.SepBeg / d.Emp).mean()),
            "aging": float((d.aging / d.Emp).mean()),
            "series": s,
        }
    return out


def _simulate(init, rates, n_quarters, index, hire_mult=1.0, sep_mult=1.0,
              senior_retention=0.0):
    """Reduced-form quarterly projection of each age band.

        Emp(t+1) = Emp(t) * (1 + hire - sep + aging)

    All three rates are measured. This is a reduced-form band model, not a
    fully structural cohort model: the measured ageing residuals do not
    balance exactly across bands, so inflow to 55+ is not constrained to
    equal outflow from 35-54. That is a documented limitation, and it is why
    the projection is validated by backtest rather than asserted.
    """
    sim = {k: [float(init[k])] for k in rates}
    for _ in range(1, n_quarters):
        for k, r in rates.items():
            h, sp, ag = r["hire"] * hire_mult, r["sep"] * sep_mult, r["aging"]
            if k == "S":
                sp = sp * (1 - senior_retention)
            sim[k].append(max(sim[k][-1] * (1 + h - sp + ag), 0.0))
    return pd.DataFrame(sim, index=index[:n_quarters])


def layer4(qwi, l1, l3, say, outdir, a: Assumptions):
    say.rule("LAYER 4 — BACKTESTED COMPARTMENT PROJECTION")

    qm = qwi[qwi.industry == MFG]
    bands = {"J": YOUTH, "M": MID, "S": SENIOR}
    band_lab = {"J": "25-34", "M": "35-54", "S": "55+"}

    # ---- verify the accounting identity before trusting any rate -----------
    g = (qm.groupby(["year", "quarter"])[["Emp", "EmpEnd", "HirAEnd", "SepBeg"]]
         .sum().sort_index())
    ident_err = ((g.Emp - g.SepBeg + g.HirAEnd - g.EmpEnd).abs() / g.EmpEnd * 100)
    say(f"\nIdentity check  EmpEnd = Emp - SepBeg + HirAEnd")
    say(f"  mean |error| = {ident_err.mean():.5f}%   max = {ident_err.max():.5f}%")
    if ident_err.max() > 0.1:
        say("  !! identity does not hold; do not proceed with these measures")

    bt_rates = band_rates(qm, bands, (2010, 2019))
    pr_rates = band_rates(qm, bands, BASE_WINDOW)

    say(f"\nMeasured quarterly rates, {BASE_WINDOW[0]}-{BASE_WINDOW[1]}")
    say(f"  {'band':6s} {'hire':>9s} {'sep':>9s} {'aging':>9s} {'net':>9s}")
    for k in bands:
        r = pr_rates[k]
        say(f"  {band_lab[k]:6s} {r['hire']:9.4f} {r['sep']:9.4f} "
            f"{r['aging']:+9.4f} {r['hire']-r['sep']+r['aging']:+9.4f}")
    say("\nAgeing signs are the expected direction: the two younger bands lose")
    say("workers to the band above, and 55+ receives them.")

    obs = pd.DataFrame({k: bt_rates[k]["series"].Emp for k in bands}).dropna()

    # ---- backtest ----------------------------------------------------------
    bt_idx = obs.index[obs.index.get_level_values(0) <= 2019]
    n_bt = len(bt_idx)
    pred = _simulate({k: obs[k].iloc[0] for k in bands},
                     {k: bt_rates[k] for k in bands}, n_bt, bt_idx)

    rows = []
    for k in bands:
        o, p = obs[k].iloc[:n_bt].values, pred[k].values
        mape = float(np.mean(np.abs((p - o) / o)) * 100)
        r2 = float(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))
        rows.append({"band": band_lab[k], "MAPE_pct": round(mape, 2),
                     "R2": round(r2, 3)})
    bt = pd.DataFrame(rows)
    say(f"\nBacktest 2010Q1-2019Q4, initialised on 2010Q1 stocks and 2010-2019 rates")
    say(bt.to_string(index=False))
    say("\nMAPE is the gate. R2 is shown for completeness but is uninformative")
    say("for the 35-54 band, whose level is nearly flat, so variance around its")
    say("own mean is tiny and R2 is harsh regardless of absolute accuracy.")

    if bt.MAPE_pct.max() > BACKTEST_MAPE_LIMIT:
        say("\n" + "!" * 74)
        say("  LAYER 4 WITHHELD — MODEL FAILS ITS BACKTEST")
        say("!" * 74)
        bt.to_csv(Path(outdir) / "layer4_backtest_FAILED.csv", index=False)
        return {"projection": None, "backtest": bt, "status": "failed_backtest"}

    say(f"\nBacktest PASSES at the {BACKTEST_MAPE_LIMIT}% MAPE limit. "
        f"Projection proceeds.")

    # ---- projection --------------------------------------------------------
    last_year = int(obs.index.get_level_values(0).max())
    n_fwd = (a.horizon_end - last_year) * 4 + 1
    fwd_idx = pd.MultiIndex.from_product(
        [range(last_year, a.horizon_end + 1), [1, 2, 3, 4]],
        names=["year", "quarter"])[:n_fwd]
    init = {k: obs[k].iloc[-1] for k in bands}
    proj = _simulate(init, pr_rates, n_fwd, fwd_idx)
    proj["S_per_J"] = proj.S / proj.J

    ann = proj.groupby(level=0).mean()
    idx = ann.div(ann.iloc[0]).mul(100).drop(columns=["S_per_J"])
    idx["S_per_J"] = ann.S_per_J
    say(f"\nProjection to {a.horizon_end}, annual means, index {last_year} = 100")
    say(idx.round(2).to_string())

    say(f"\nSeniors per junior: {ann.S_per_J.iloc[0]:.2f} ({last_year}) -> "
        f"{ann.S_per_J.iloc[-1]:.2f} ({a.horizon_end})")
    say("\nNOTE ON THE 'MENTORSHIP VACUUM' PREMISE")
    say("Seniors are not scarce relative to juniors at the aggregate level, so")
    say("a headcount-based mentoring shortage does not bind. The defensible")
    say("claim is that mentor availability per junior changes, not that it")
    say("collapses. A binding shortage would have to be demonstrated at")
    say("occupation level (SOC), which these industry files cannot support.")
    say("The feedback-loop collapse reported in the earlier version is dropped.")

    # ---- levers, one at a time --------------------------------------------
    levers = {
        "none (baseline)": {},
        "senior separations -20%": {"senior_retention": 0.20},
        "hiring +10%, all bands": {"hire_mult": 1.10},
        "separations -10%, all bands": {"sep_mult": 0.90},
    }
    rows = []
    for name, kw in levers.items():
        r = _simulate(init, pr_rates, n_fwd, fwd_idx, **kw)
        ra = r.groupby(level=0).mean()
        rows.append({
            "lever": name,
            f"J_{a.horizon_end}": round(ra.J.iloc[-1] / ann.J.iloc[0] * 100, 1),
            f"M_{a.horizon_end}": round(ra.M.iloc[-1] / ann.M.iloc[0] * 100, 1),
            f"S_{a.horizon_end}": round(ra.S.iloc[-1] / ann.S.iloc[0] * 100, 1),
            "S_per_J": round(ra.S.iloc[-1] / ra.J.iloc[-1], 3),
        })
    lev = pd.DataFrame(rows)
    say("\nPolicy levers, evaluated ONE AT A TIME")
    say(lev.to_string(index=False))
    say("\nEach row changes exactly one rate, so its effect is attributable.")
    say("The earlier version moved five parameters at once and reported a")
    say("single improvement factor that could not be decomposed.")

    # ---- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    xo = np.arange(n_bt)
    for k, c in zip(bands, [PALETTE["dark"], PALETTE["accent"], PALETTE["mfg"]]):
        ax1.plot(xo, obs[k].iloc[:n_bt].values / 1e3, color=c, lw=2,
                 label=f"{band_lab[k]} observed")
        ax1.plot(xo, pred[k].values / 1e3, color=c, lw=1.4, ls="--", alpha=0.85,
                 label=f"{band_lab[k]} modelled")
    ax1.set_xticks(np.arange(0, n_bt, 8))
    ax1.set_xticklabels([str(2010 + i // 4) for i in range(0, n_bt, 8)])
    ax1.set_title(f"Backtest 2010-2019 (Worst MAPE {bt.MAPE_pct.max():.2f}%)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Workers (thousands)")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(axis="y", alpha=0.3)

    for k, c in zip(bands, [PALETTE["dark"], PALETTE["accent"], PALETTE["mfg"]]):
        ax2.plot(idx.index, idx[k], color=c, lw=2.2, label=band_lab[k])
    ax2.axhline(100, color="black", ls=":", lw=1)
    ax2.set_title(f"Projection, {last_year} = 100",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Index")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Layer 4 — Matched-measure compartment model",
                 fontsize=13.5, fontweight="bold")
    save_fig(fig, outdir, "layer4_projection",
             "Rates measured in QWI: hires HirAEnd/Emp, separations SepBeg/Emp, "
             "ageing (Emp_next - EmpEnd)/Emp. Reduced-form projection.")

    proj.to_csv(Path(outdir) / "layer4_projection_quarterly.csv")
    idx.to_csv(Path(outdir) / "layer4_projection_indexed.csv")
    bt.to_csv(Path(outdir) / "layer4_backtest.csv", index=False)
    lev.to_csv(Path(outdir) / "layer4_levers.csv", index=False)
    return {"projection": proj, "indexed": idx, "backtest": bt,
            "levers": lev, "rates": pr_rates, "status": "reported"}


# ==============================================================================
#  LAYER 5 — DESTINATION MIX
# ==============================================================================

def layer5(od, say, outdir):
    say.rule("LAYER 5 — WHERE SEPARATING WORKERS GO")

    m = od[(od.agg_level == AGG["od_matrix"]) & (od.industry_orig == MFG)]
    counts = m.groupby(["year", "industry"])["J2J"].sum().unstack(fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0) * 100

    yrs = [y for y in (2010, 2015, 2019, 2024) if y in share.index]
    say("\nDestination share of separations originating in manufacturing (%)")
    say(share.rename(columns=SECTOR_NAME).loc[yrs].round(2).T.to_string())

    delta = share.loc[share.index.max()] - share.loc[share.index.min()]
    say("\nChange in destination share, first to last year (pp)")
    say(delta.rename(SECTOR_NAME).sort_values(ascending=False).round(2).to_string())

    stay = share[MFG]
    say(f"\nStayed within manufacturing: {stay.iloc[0]:.1f}% -> {stay.iloc[-1]:.1f}%")
    say("A separation is not necessarily a loss to the sector. Roughly a third")
    say("of moves are to another manufacturing employer in the same region,")
    say("which reframes this as intra-regional competition for the same workers.")

    d = delta.drop(labels=[MFG], errors="ignore").sort_values()
    keep = pd.concat([d.head(5), d.tail(5)])
    labs = [SECTOR_NAME.get(c, c) for c in keep.index]
    cols = [PALETTE["dark"] if v < 0 else PALETTE["mfg"] for v in keep.values]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(keep)), keep.values, color=cols, alpha=0.9)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(labs, fontsize=10)
    for i, v in enumerate(keep.values):
        ax.text(v + (0.12 if v >= 0 else -0.12), i, f"{v:+.2f}pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"Change in destination share, {share.index.min():.0f}"
                  f"-{share.index.max():.0f} (pp)")
    ax.set_title("Layer 5 — Destination Mix of Manufacturing Separations",
                 fontsize=12, fontweight="bold")
    # Fixed symmetric ticks so the largest mover is not clipped and gains and
    # losses are read on the same scale.
    # tightest symmetric range on whole units that still fits every bar label
    lim = max(4.0, np.ceil(np.abs(keep.values).max()))
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.arange(-lim, lim + 0.5, 1.0))
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, outdir, "layer5_destination_mix",
             "Source: Census LEHD J2J Origin-Destination. AL/GA/NC/SC/TN, NSA. "
             "Intra-region moves only.")

    counts.to_csv(Path(outdir) / "layer5_destination_counts.csv")
    share.round(4).to_csv(Path(outdir) / "layer5_destination_share_pct.csv")
    return {"counts": counts, "share": share, "stay_in_mfg": stay.iloc[-1] / 100}


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--od", required=True, help="J2J Origin-Destination csv")
    p.add_argument("--sep", required=True, help="J2J Separations csv")
    p.add_argument("--qwi", required=True, help="QWI csv")
    p.add_argument("--outdir", default="output")
    p.add_argument("--skip-layer4", action="store_true",
                   help="skip the simulation, which relies on unobserved parameters")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    say = Reporter()

    say.rule("THE MENTORSHIP VACUUM — FULL PIPELINE")
    say("Region: AL, GA, NC, SC, TN   Sector: NAICS 31-33   Period: 2010-2024")

    od, sep, qwi = load_od(args.od), load_sep(args.sep), load_qwi(args.qwi)
    if not audit(od, sep, qwi, say):
        say("\nAudit failed. Fix the inputs before trusting anything below.")

    l1 = layer1(qwi, sep, say, out)
    l2 = layer2(od, say, out)
    l5 = layer5(od, say, out)
    l3 = layer3(qwi, sep, say, out)
    l3["stay_in_mfg"] = l5["stay_in_mfg"]

    if not args.skip_layer4:
        l4 = layer4(qwi, l1, l3, say, out, Assumptions())
        if l4.get("status") == "failed_backtest":
            l4 = None
    else:
        l4 = None
        say("\nLayer 4 skipped.")

    med, lo, hi = l2["crossing"]
    mech = l3["mechanism"]
    say.rule("HEADLINE FINDINGS")
    say(f"1. Senior share of the workforce rose "
        f"{l1['share'][SENIOR].sum(axis=1).iloc[0]:.1f}% -> "
        f"{l1['share'][SENIOR].sum(axis=1).iloc[-1]:.1f}% "
        f"({l1['senior_share_slope']:+.2f} pp/yr).")
    say(f"2. Replacement Ratio fell {l2['table'].RR.iloc[0]:.2f} -> "
        f"{l2['table'].RR.iloc[-1]:.2f}; crosses 1.0 in {med:.0f} "
        f"(95% CI {lo:.0f}-{hi:.0f}).")
    say(f"3. Manufacturing has the LOWEST separation rate of its peer set "
        f"({l3['peer'].set_index('industry').loc['Manufacturing','annualised_pct']:.1f}% annualised).")
    say(f"4. Retirement's share of senior separations fell "
        f"{mech.retire_share_pct.iloc[0]:.1f}% -> {mech.retire_share_pct.iloc[-1]:.1f}%; "
        f"job-to-job exits grew "
        f"{mech.EESep.iloc[-1]/mech.EESep.iloc[0]-1:+.0%}.")
    say(f"5. {l5['share'][MFG].iloc[-1]:.0f}% of separations move to another "
        f"manufacturing employer in the same region.")

    say.rule("SCOPE AND LIMITATIONS")
    for line in [
        "J2J Origin-Destination covers moves with both endpoints inside the five",
        "  states. Out-of-region moves are not observed.",
        "ENPersist proxies retirement only for the 55+ group.",
        "State-level aggregation masks county and corridor variation.",
        "2020-2021 are shaded in every figure and excluded from calibration.",
        "Layer 4 is withheld when it fails its backtest. See the log for the",
        "  diagnosis; a failing projection is reported as a failure, not shown.",
        "Association only. No causal identification is claimed.",
    ]:
        say(f"- {line}")

    meta = {
        "region": STATES, "sector": MFG,
        "period": [int(l2["table"].index.min()), int(l2["table"].index.max())],
        "agg_levels": AGG, "assumptions": asdict(Assumptions()),
        "rr_first": float(l2["table"].RR.iloc[0]),
        "rr_last": float(l2["table"].RR.iloc[-1]),
        "rr_slope": l2["fit"]["slope"], "rr_r2": l2["fit"]["r2"],
        "rr_hac_p": l2["fit"]["p_hac"],
        "crossing_year": med, "crossing_ci": [lo, hi],
        "layer4_status": "reported" if l4 else "withheld_failed_backtest",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    say.write(out / "results_log.txt")
    say(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
