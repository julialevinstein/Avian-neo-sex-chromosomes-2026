
import os

# =============================================================================
# ============================ CONFIGURATION ==================================
# =============================================================================

# --- C0. ENVIRONMENT (must be set before pytensor/pymc are imported) ---------
PYTENSOR_CACHE_DIR = "/dss/dsslegfs01/pr53da/pr53da-dss-0036/projects/2025__neosex/pytensor_cache"

os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={PYTENSOR_CACHE_DIR}"

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# --- C1. FILES ----------------------------------------------------------------
AUTO_GENE_FILE = "auto_genes_values_with_outgroup_branches_with_segments.tsv"
NEO_GENE_FILE = "neo_genes_values_with_outgroup_branches_with_segments.tsv"
CROWN_TIP_FILE = "neo_genes_crown_tip_distances.tsv"

EXCLUDE_GENES = []                  # genes to drop, e.g. ones showing recombination
EXCLUDE_GENES_FILE = None           # one gene name per line; "#" comments ignored

OUTPUT_DIR = "."
OUTPUT_PREFIX = "neo_fusion"

# --- C2. COLUMN NAMES ---------------------------------------------------------
COL_GENE = "Gene"
COL_ALIGN_LEN = "ls_codons"                   # alignment length, for precision weighting
COL_SEGMENTS_RESOLVED = "segments_resolved"   # "True"/"False"

COL_DEEP_SEGMENT = "Stem_dS_anser_falcon"     # stem, outer split -> inner split
COL_RECENT_SEGMENT = "Stem_dS_falcon_crown"   # stem, inner split -> crown
COL_OUTGROUP_BRANCH = "Ans_ans_branch_dS"     # inner outgroup's own branch, 0 -> T_INNER_SPLIT
COL_CROWN_TIP_DS = "Crown_dS_tip"             # in CROWN_TIP_FILE

# --- C3. CALIBRATION TIMES (MYA) ---------------------------------------------
T_CROWN = 39.0          # crown radiation of the focal clade
T_INNER_SPLIT = 60.0    # split from the nearer outgroup
T_OUTER_SPLIT = 88.0    # split from the farther outgroup

# Time spanned by the outgroup branch in COL_OUTGROUP_BRANCH (split -> present).
# Used to turn that branch into a rate, both for mu_A and for kappa_dilution,
# so the two calculations always use the same value.
# Must match the outgroup in COL_OUTGROUP_BRANCH:
#   "Fal_per_branch_dS" -> T_INNER_SPLIT
#   "Ans_ans_branch_dS" -> T_OUTER_SPLIT
T_OUTGROUP_BRANCH_SPAN = T_OUTER_SPLIT

# --- C4. PRIOR ON THE FUSION AGE ---------------------------------------------
# The fusion must postdate the split from a lineage lacking it and predate the
# radiation of a clade in which every species carries it, so t_fusion is bounded
# within [T_CROWN, T_FUSION_MAX]. Both bounds are phylogenetic facts.
#
#   "uniform"           flat over the whole window. The honest default.
#   "uniform_soft_tail" flat over [T_CROWN, T_FUSION_CORE_MAX], then a lower but
#                       NON-ZERO shelf out to T_FUSION_MAX. Use when the upper
#                       bound is an inference you would let the data overturn.
#                       A plain uniform puts zero density past its edge, and
#                       zero prior x any likelihood = zero posterior, so a
#                       uniform's edge can never be argued with. Soft bounds in
#                       the sense of Yang & Rannala (2006), who use 2.5%.
PRIOR_TYPE = "uniform"
T_FUSION_MAX = T_OUTER_SPLIT   # set to T_INNER_SPLIT if the neo genes are
                               # autosomal in the inner outgroup
T_FUSION_CORE_MAX = T_INNER_SPLIT   # "uniform_soft_tail" only
T_FUSION_TAIL_WEIGHT = 0.025        # "uniform_soft_tail" only: prior mass above the core

# --- C5. MODEL OPTIONS --------------------------------------------------------
# "hierarchical": genes share a population mean fusion age but may scatter.
# "complete"    : ONE fusion age shared by all genes. A fusion is a single
#                 event, so this is the biologically strict choice; use it
#                 unless you expect evolutionary strata.
# "none"        : every gene gets its own independent fusion age, and the
#                 reported age is their mean. No pooling at all, so a noisy
#                 gene is not reined in -- useful as a diagnostic for how much
#                 the genes actually disagree, not as a result.
POOLING_MODE = "hierarchical"

# Where mu_A comes from:
#   "outgroup" : per gene, from its own outgroup branch (validated by the
#                negative-control script: unfused genes give a ratio of ~1.0)
#   "autosomal": one pooled value for all genes, from the autosomal reference
#                genes' recent stem segment
MU_A_SOURCE = "outgroup"

# Correct the outgroup branch for dilution. The branch spans 0 -> T_INNER_SPLIT
# while the window we want is T_CROWN -> T_INNER_SPLIT, and it sits on a
# different lineage. The correction is measured on the autosomal genes, where
# both quantities are observable, and so carries over to the neo-Z genes.
CORRECT_DILUTION = True

# Scale both rates in the deep window by the autosomal deep/recent rate ratio.
# Deeper windows generally show lower apparent rates; without this the deep
# segment is systematically mispredicted.
SCALE_DEEP_WINDOW = True

# Weight genes by alignment length: sigma is multiplied by sqrt(mean(L)/L), so
# a short alignment is allowed to scatter further from its expectation.
USE_PRECISION_WEIGHTING = True

# Genes whose calculated gamma = mu_Z/mu_A is at or above this are reported.
# Set DROP_HIGH_GAMMA = True to also remove them -- but read the warning the
# script prints before doing so; it is selection on the outcome.
GAMMA_FLAG_THRESHOLD = 1.0
DROP_HIGH_GAMMA = False

# --- C6. SAMPLING -------------------------------------------------------------
DRAWS = 2000
TUNE = 2500
CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

DS_FLOOR = 1e-4   # LogNormal needs positive data; zero-length branches get floored

# =============================================================================
# ========================== END CONFIGURATION ================================
# =============================================================================

RECENT_SPAN = T_INNER_SPLIT - T_CROWN
DEEP_SPAN = T_OUTER_SPLIT - T_INNER_SPLIT


def load_excluded_genes():
    excluded = set(EXCLUDE_GENES)
    if EXCLUDE_GENES_FILE:
        with open(EXCLUDE_GENES_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    excluded.add(line)
    return excluded


def prep(x, label):
    """Floor zero/near-zero branch lengths so the LogNormal likelihood stays valid."""
    x = np.asarray(x, dtype=float)
    n_low = int((x < DS_FLOOR).sum())
    if n_low:
        print(f"WARNING: {n_low} {label} value(s) < {DS_FLOOR} (zero-length branches?) "
              f"-- floored at {DS_FLOOR}")
    return np.maximum(x, DS_FLOOR)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def resolved(df):
    """String-safe boolean parse: .astype(bool) turns NaN and the text 'False' into True."""
    return df[COL_SEGMENTS_RESOLVED].astype(str).str.strip().str.lower().eq("true")


EXCLUDED = load_excluded_genes()

print(f">>> CONFIG: prior={PRIOR_TYPE} over [{T_CROWN}, {T_FUSION_MAX}] MYA, "
      f"pooling={POOLING_MODE}, mu_A from {MU_A_SOURCE}, dilution={CORRECT_DILUTION}, "
      f"deep scaling={SCALE_DEEP_WINDOW}, precision weighting={USE_PRECISION_WEIGHTING}")
if EXCLUDED:
    print(f">>> Excluding {len(EXCLUDED)} genes by name: {sorted(EXCLUDED)}")

# The prior is flat in MYA, but a flat prior is NOT uninformative: its mean is
# the midpoint of the window, so when the data say little the posterior drifts
# there. Printing it makes that visible.
_prior_mean = 0.5 * (T_CROWN + T_FUSION_MAX)
_prior_sd = (T_FUSION_MAX - T_CROWN) / np.sqrt(12.0)
print(f">>> Prior mean {_prior_mean:.1f} MYA, SD {_prior_sd:.1f} MY "
      f"(a flat prior still has a centre -- compare the posterior against it)")

if T_FUSION_MAX <= T_INNER_SPLIT:
    print(">>> NOTE: t_fusion cannot reach the deep window, so the deep segment carries no\n"
          "    information about the fusion age. It only constrains its own noise term.")

# =============================================================================
# 1. AUTOSOMAL REFERENCE GENES -> the two correction factors
#    Plain arithmetic. Nothing here is sampled.
# =============================================================================
df_auto = pd.read_csv(AUTO_GENE_FILE, sep="\t").set_index(COL_GENE)

need = [COL_RECENT_SEGMENT, COL_DEEP_SEGMENT]
if CORRECT_DILUTION and MU_A_SOURCE == "outgroup":
    need.append(COL_OUTGROUP_BRANCH)
missing = [c for c in need if c not in df_auto.columns]
if missing:
    raise ValueError(f"Missing columns {missing} in {AUTO_GENE_FILE}")

auto_ok = resolved(df_auto).values.copy()   # .values can be read-only; we modify it below
auto_recent_raw = numeric(df_auto, COL_RECENT_SEGMENT).values
auto_deep_raw = numeric(df_auto, COL_DEEP_SEGMENT).values
auto_ok &= ~np.isnan(auto_recent_raw) & ~np.isnan(auto_deep_raw)
if CORRECT_DILUTION and MU_A_SOURCE == "outgroup":
    auto_branch_raw = numeric(df_auto, COL_OUTGROUP_BRANCH).values
    auto_ok &= ~np.isnan(auto_branch_raw)

auto_recent = prep(auto_recent_raw[auto_ok], "autosomal recent segment")
auto_deep = prep(auto_deep_raw[auto_ok], "autosomal deep segment")
print(f">>> {int(auto_ok.sum())}/{len(df_auto)} autosomal reference genes used")

# kappa_deep: deep-window rate / recent-window rate, SAME genes in numerator and
# denominator so gene identity cancels.
kappa_deep = ((np.mean(auto_deep / DEEP_SPAN) / np.mean(auto_recent / RECENT_SPAN))
              if SCALE_DEEP_WINDOW else 1.0)

# kappa_dilution: isolated-window rate / full-outgroup-branch rate, again on the
# same genes, so it translates one into the other.
if CORRECT_DILUTION and MU_A_SOURCE == "outgroup":
    auto_branch = prep(auto_branch_raw[auto_ok], "autosomal outgroup branch")
    kappa_dilution = (np.mean(auto_recent / RECENT_SPAN)
                      / np.mean(auto_branch / T_OUTGROUP_BRANCH_SPAN))
else:
    kappa_dilution = 1.0

# =============================================================================
# 2. NEO-Z GENES
# =============================================================================
df_neo = pd.read_csv(NEO_GENE_FILE, sep="\t").set_index(COL_GENE)

if MU_A_SOURCE == "outgroup" and COL_OUTGROUP_BRANCH not in df_neo.columns:
    raise ValueError(f"MU_A_SOURCE='outgroup' needs column {COL_OUTGROUP_BRANCH} in {NEO_GENE_FILE}")

neo_recent_raw = numeric(df_neo, COL_RECENT_SEGMENT)
neo_deep_raw = numeric(df_neo, COL_DEEP_SEGMENT)
keep = (resolved(df_neo) & neo_recent_raw.notna() & neo_deep_raw.notna()).values.copy()
keep &= ~df_neo.index.isin(EXCLUDED)

if MU_A_SOURCE == "outgroup":
    neo_branch_raw = numeric(df_neo, COL_OUTGROUP_BRANCH)
    keep &= neo_branch_raw.notna().values

# Crown tips: mu_Z comes from them, so a gene without any is unusable.
df_tips = pd.read_csv(CROWN_TIP_FILE, sep="\t")
tip_vals = pd.to_numeric(df_tips[COL_CROWN_TIP_DS], errors="coerce")
tips_ok = df_tips.assign(_v=tip_vals).dropna(subset=["_v"])
crown_mean = tips_ok.groupby(COL_GENE)["_v"].mean()
crown_count = tips_ok.groupby(COL_GENE)["_v"].size()

neo_crown_raw = df_neo.index.map(crown_mean).to_numpy(dtype=float)
keep &= ~np.isnan(neo_crown_raw)

if (~keep).sum():
    print(f"WARNING: dropping {int((~keep).sum())} neo-Z genes (unresolved segments, "
          f"missing values, no crown tips, or excluded by name):")
    print(df_neo.index.values[~keep])

df_neo = df_neo[keep]
genes = df_neo.index.values
if len(genes) == 0:
    raise ValueError("No neo-Z genes left -- check the column names in the config.")

dS_recent = prep(neo_recent_raw[keep].values, "neo recent segment")
dS_deep = prep(neo_deep_raw[keep].values, "neo deep segment")
crown_dS = prep(neo_crown_raw[keep], "neo mean crown-tip")
n_tips = df_neo.index.map(crown_count).to_numpy(dtype=float)
align_len = df_neo[COL_ALIGN_LEN].values.astype(float)

print(f">>> {len(genes)} neo-Z genes, {int(n_tips.sum())} crown-tip observations "
      f"({n_tips.mean():.1f} per gene)")

# =============================================================================
# 3. CALCULATE THE RATES
# =============================================================================
mu_Z = crown_dS / T_CROWN

if MU_A_SOURCE == "outgroup":
    neo_branch = prep(neo_branch_raw[keep].values, "neo outgroup branch")
    mu_A = (neo_branch / T_OUTGROUP_BRANCH_SPAN) * kappa_dilution
else:
    mu_A = np.full(len(genes), float(np.mean(auto_recent / RECENT_SPAN)))

gamma = mu_Z / mu_A

print(f"\n>>> RATES (calculated from the data, not estimated)")
print(f"    kappa_deep      = {kappa_deep:.4f}   (deep / recent window rate ratio)")
print(f"    kappa_dilution  = {kappa_dilution:.4f}   (isolated window / full outgroup branch)")
print(f"    mu_A   median {np.median(mu_A):.6f}   [{mu_A.min():.6f}, {mu_A.max():.6f}]")
print(f"    mu_Z   median {np.median(mu_Z):.6f}   [{mu_Z.min():.6f}, {mu_Z.max():.6f}]")
print(f"    gamma  median {np.median(gamma):.4f}   [{gamma.min():.4f}, {gamma.max():.4f}]")

high = gamma >= GAMMA_FLAG_THRESHOLD
if high.any():
    print(f"\n    {high.sum()}/{len(genes)} genes have gamma >= {GAMMA_FLAG_THRESHOLD}: "
          f"{list(genes[high])}")
    print("    These imply the neo-Z evolves at least as fast as the autosome, which cannot")
    print("    produce a dS deficit, so they carry no fusion signal and only add noise.")
    print("    Usually it means a small, poorly estimated mu_A rather than real biology.")
    if DROP_HIGH_GAMMA:
        print("    DROPPING them (DROP_HIGH_GAMMA=True). CAUTION: this is selection on the")
        print("    outcome -- removing the high-gamma tail biases the remaining gamma down and")
        print("    the fusion age young. Report both runs.")
        sel = ~high
        genes, dS_recent, dS_deep = genes[sel], dS_recent[sel], dS_deep[sel]
        mu_A, mu_Z, gamma = mu_A[sel], mu_Z[sel], gamma[sel]
        align_len, n_tips = align_len[sel], n_tips[sel]

ratio = mu_A / np.median(mu_A)
odd = (ratio > 2.5) | (ratio < 0.4)
if odd.any():
    print(f"\n    {odd.sum()} gene(s) with an unusual mu_A (>2.5x or <0.4x the median):")
    for g, r in sorted(zip(genes[odd], ratio[odd]), key=lambda t: -t[1]):
        print(f"      {g:12s} {r:5.1f}x median")
    print("    Worth checking their alignments for paralogues or misalignment before trusting them.")

# The age is recovered by solving the stem equation, which puts (mu_Z - mu_A) in
# the denominator. As gamma approaches 1 that denominator vanishes and the age
# becomes unidentifiable, whatever the prior says.
if np.median(gamma) > 0.95:
    print(f"\n    WARNING: median gamma {np.median(gamma):.3f} is very close to 1. An early and a")
    print("    late fusion predict almost the same stem dS, so the age is barely identifiable.")

N = len(genes)
weight = np.sqrt(np.mean(align_len) / align_len) if USE_PRECISION_WEIGHTING else np.ones(N)

# =============================================================================
# 4. MODEL
#    t_fusion and the two noise terms are the only things inferred.
# =============================================================================
with pm.Model() as model:

    def observe(name, expected, rel_sigma, observed):
        """LogNormal parameterised so mean == expected and CV == rel_sigma exactly."""
        sd_log = pt.sqrt(pt.log1p(rel_sigma ** 2))
        return pm.LogNormal(name, mu=pt.log(expected) - 0.5 * sd_log ** 2,
                            sigma=sd_log, observed=observed)

    # Noise terms stay free: they describe how far an observed stem dS scatters
    # around its expectation, and fixing them would be arbitrary and would leave
    # the model no way to express how well it fits.
    sigma_recent = pm.Deterministic(
        "sigma_recent", pm.math.exp(pm.Normal("log_sigma_recent", mu=np.log(0.3), sigma=1.0)))
    sigma_deep = pm.Deterministic(
        "sigma_deep", pm.math.exp(pm.Normal("log_sigma_deep", mu=np.log(0.4), sigma=1.0)))

    # --- t_fusion -------------------------------------------------------------
    def draw_t_fusion(name, **kwargs):
        if PRIOR_TYPE == "uniform":
            return pm.Uniform(name, lower=T_CROWN, upper=T_FUSION_MAX, **kwargs)
        if PRIOR_TYPE == "uniform_soft_tail":
            # Inverse-CDF of two stacked uniforms, built from a plain Uniform(0,1)
            # so the sampler stays on a well-behaved space. Mass (1-w) is spread
            # flat over the core and w flat over the shelf; continuous at the join.
            w = T_FUSION_TAIL_WEIGHT
            u = pm.Uniform(f"{name}_u", lower=0.0, upper=1.0, **kwargs)
            core = T_CROWN + (u / (1.0 - w)) * (T_FUSION_CORE_MAX - T_CROWN)
            shelf = T_FUSION_CORE_MAX + ((u - (1.0 - w)) / w) * (T_FUSION_MAX - T_FUSION_CORE_MAX)
            return pm.Deterministic(name, pt.switch(u < (1.0 - w), core, shelf))
        raise ValueError(f"Unknown PRIOR_TYPE: {PRIOR_TYPE}")

    if POOLING_MODE == "complete":
        t_global = draw_t_fusion("t_fusion_global")
        t_fusion = t_global * pt.ones(N)
    elif POOLING_MODE == "none":
        t_fusion = draw_t_fusion("t_fusion", shape=N)
        t_global = pt.mean(t_fusion)
    elif POOLING_MODE == "hierarchical":
        # The population mean carries the prior; genes scatter around it on a
        # logit scale, which keeps every gene inside [T_CROWN, T_FUSION_MAX].
        t_global = draw_t_fusion("t_fusion_global")
        span = T_FUSION_MAX - T_CROWN
        frac = pt.clip((t_global - T_CROWN) / span, 1e-6, 1 - 1e-6)
        centre = pt.log(frac) - pt.log1p(-frac)
        spread = pm.HalfNormal("gene_spread", sigma=0.75)
        offset = pm.Normal("gene_offset", mu=0, sigma=1, shape=N)   # non-centred
        t_fusion = pm.Deterministic(
            "t_fusion", T_CROWN + pm.math.sigmoid(centre + offset * spread) * span)
    else:
        raise ValueError(f"Unknown POOLING_MODE: {POOLING_MODE}")

    pm.Deterministic("t_fusion_MYA", t_global)

    # --- the two stem segments ------------------------------------------------
    # Each segment is split into the part before the fusion (at mu_A) and the
    # part after it (at mu_Z). The clamps keep both durations non-negative when
    # t_fusion falls outside the segment.
    deep_pre = pt.maximum(0.0, T_OUTER_SPLIT - pt.maximum(T_INNER_SPLIT, t_fusion))
    deep_post = pt.maximum(0.0, pt.minimum(T_OUTER_SPLIT, t_fusion) - T_INNER_SPLIT)
    exp_deep = kappa_deep * (mu_A * deep_pre + mu_Z * deep_post)

    recent_pre = pt.maximum(0.0, T_INNER_SPLIT - pt.maximum(T_CROWN, t_fusion))
    recent_post = pt.maximum(0.0, pt.minimum(T_INNER_SPLIT, t_fusion) - T_CROWN)
    exp_recent = mu_A * recent_pre + mu_Z * recent_post

    observe("obs_recent", exp_recent, sigma_recent * weight, dS_recent)
    observe("obs_deep", exp_deep, sigma_deep * weight, dS_deep)

# =============================================================================
# 5. SAMPLE AND REPORT
# =============================================================================
if __name__ == "__main__":
    tag = f"{POOLING_MODE}_{PRIOR_TYPE}_muA{MU_A_SOURCE}_max{int(T_FUSION_MAX)}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_trace_{tag}.nc")
    results_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results_{tag}.tsv")

    with model:
        print(f"\nSampling ({N} genes, {tag})...")
        trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED,
                          return_inferencedata=True)

    trace.to_netcdf(trace_path)
    print(f"\n>>> Trace saved: {trace_path}")

    shown = [v for v in ["t_fusion_MYA", "gene_spread", "sigma_recent", "sigma_deep"]
             if v in trace.posterior]
    print("\n=== POSTERIOR SUMMARY ===")
    print(az.summary(trace, var_names=shown, round_to=4))
    print("\n(mu_A, mu_Z, gamma, kappa_deep and kappa_dilution are fixed inputs -- "
          "see the RATES block above, not this table.)")

    print(f"\n>>> Divergences: {int(trace.sample_stats.diverging.sum())}")

    t = trace.posterior["t_fusion_MYA"].values.flatten()
    hdi = az.hdi(t, hdi_prob=0.95)
    med, mean = float(np.median(t)), float(np.mean(t))
    q = np.quantile(t, [0.05, 0.25, 0.75, 0.95])

    print(f"\n>>> FUSION AGE: median {med:.2f} MYA   (mean {mean:.2f})")
    print(f">>> 95% credible interval: [{hdi[0]:.2f}, {hdi[1]:.2f}] MYA")
    print(f">>> Quantiles: 5% {q[0]:.1f} | 25% {q[1]:.1f} | 50% {med:.1f} | "
          f"75% {q[2]:.1f} | 95% {q[3]:.1f} MYA")
    print(f">>> 95% probability the fusion is younger than {q[3]:.1f} MYA")
    if mean > 1.15 * med:
        print("    NOTE: mean >> median -- long old-age tail. Quote the median and the bound.")

    # How much did the data actually move things? A posterior as wide as the
    # prior means the answer is the prior, not the data.
    post_sd = float(np.std(t))
    print(f"\n>>> Prior SD {_prior_sd:.1f} MY -> posterior SD {post_sd:.1f} MY "
          f"({(1 - post_sd / _prior_sd) * 100:.0f}% of the prior uncertainty removed)")
    if post_sd > 0.7 * _prior_sd:
        print("    WARNING: little was learned -- this posterior is mostly the prior.")

    at_floor = float(np.mean(t < T_CROWN + 0.03 * (T_FUSION_MAX - T_CROWN)))
    at_ceiling = float(np.mean(t > T_FUSION_MAX - 0.03 * (T_FUSION_MAX - T_CROWN)))
    print(f">>> Mass near the {T_CROWN:.0f} MY floor: {at_floor*100:.1f}%   "
          f"near the {T_FUSION_MAX:.0f} MY ceiling: {at_ceiling*100:.1f}%")
    if at_floor > 0.10:
        print("    NOTE: mass piled against the floor -- the lower bound comes from the")
        print("    boundary, not the data. Phrase the result as 'no older than X'.")
    if at_ceiling > 0.10:
        print("    NOTE: mass piled against the ceiling -- the data may prefer an older age")
        print("    than the window allows. Check that the ceiling is justified.")

    # --- per-gene table -------------------------------------------------------
    rows = []
    per_gene = ("t_fusion" in trace.posterior)
    tg = trace.posterior["t_fusion"].values if per_gene else None
    for i in range(N):
        row = {"Gene": genes[i], "mu_A": round(float(mu_A[i]), 6),
               "mu_Z": round(float(mu_Z[i]), 6), "gamma": round(float(gamma[i]), 4),
               "n_crown_tips": int(n_tips[i]), "align_len": int(align_len[i])}
        if per_gene:
            ti = tg[:, :, i].flatten()
            h = az.hdi(ti, hdi_prob=0.95)
            row.update({"median_age_MYA": round(float(np.median(ti)), 3),
                        "HDI_lo": round(float(h[0]), 3), "HDI_hi": round(float(h[1]), 3)})
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False, sep="\t")
    print(f"\n=== PER-GENE VALUES (saved to {results_path}) ===")
    print(results.sort_values("gamma").to_string(index=False))

    print("\n================ DONE ================")
