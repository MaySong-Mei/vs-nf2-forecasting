# Endpoint definition × baseline size × MRI predictability — novelty note

Date: 2026-10-09

## Core question

A more promising methodological question has emerged from the recent UCSD experiments:

> **Does the operational definition of vestibular-schwannoma growth determine what appears predictable from baseline clinical variables and MRI?**

This reframes the current negative / unstable model results. The central object is no longer just architecture performance; it is the interaction between outcome construction, baseline tumor size, measurement reliability, and MRI incremental value.

## Why the endpoint matters mathematically

For a fixed absolute annual growth threshold,

```text
(V1 - V0) / Δt > C
```

is equivalent to

```text
(V1 - V0) / (V0 Δt) > C / V0.
```

Therefore a fixed absolute threshold induces a baseline-volume-dependent relative-growth threshold. Larger tumors need a smaller relative growth rate to cross the same absolute threshold, so baseline volume can acquire predictive power even when true relative growth propensity is weakly related or unrelated to size.

For relative growth,

```text
r = (V1 - V0) / (V0 Δt),
```

measurement error in volume is amplified approximately as a function of `1 / V0`; smaller tumors can therefore have more label instability under fixed relative cutoffs.

This creates a real methodological tension:

- **absolute endpoint:** stronger size coupling;
- **fixed relative endpoint:** reduced size coupling but greater small-tumor measurement sensitivity;
- **size-adaptive / measurement-aware endpoint:** promising, but must be prospectively specified and clinically justified rather than chosen for AUC.

## Current empirical motivation from this project

Existing UCSD endpoint sensitivity analyses already show that matched scalar baseline performance changes substantially with growth definition:

- absolute thresholds (e.g. 50 / 100 / 200 mm³/year) produce much stronger volume-only discrimination;
- simple relative annual thresholds (10% / 20% / 30% per year) produce far weaker volume-only discrimination;
- a simulation where relative growth is independent of baseline volume still allows baseline volume to predict a fixed absolute-growth label strongly, while relative-growth prediction remains near chance;
- MRI incremental value is not uniform across endpoint / tumor-size strata; prior R20 analyses showed a stronger local MRI signal in the `<200 mm³` region, although this is exploratory and not an interaction proof.

These results weaken the interpretation that a strong absolute-growth clinical baseline automatically implies little residual MRI information.

## Literature novelty assessment

### Already established / low novelty

1. **VS growth definitions are heterogeneous.** Prior systematic reviews report substantial variation across linear, absolute volumetric, relative volumetric, and annualized definitions.
2. **Relative volumetric measurement error depends on tumor size.** Inter-observer studies show substantially larger relative uncertainty for small VS lesions.
3. **Size-dependent growth thresholds have precedent.** Recent work (including Cornelissen et al.) uses volume-dependent thresholds and repeated confirmation rather than a single fixed 20% rule.
4. **Baseline MRI / radiomics can be used to predict VS growth.** Multiple prior studies already exist.

### More novel / underexplored

The more defensible novelty claim is the joint analysis:

> **endpoint definition → baseline-size predictability → MRI incremental value**

I have not identified prior VS work that systematically holds cohort / split / model family fixed and asks how changing the outcome definition changes:

- the apparent strength of baseline tumor volume;
- the incremental value of age / sex / clinical covariates;
- the incremental value of MRI intensity relative to matched scalar and geometry controls;
- the stability of those increments across tumor-size regimes.

The conceptual contribution is therefore not “a better threshold,” but **target-induced predictability**: measured model performance can partly be a property of outcome construction.

## Candidate study framing

A strong paper-level question would be:

> **Are reported predictors of vestibular-schwannoma growth partly properties of how growth is defined?**

or more specifically:

> **How does growth-endpoint construction alter the apparent predictive value of baseline tumor size, clinical variables, and MRI?**

## Recommended analysis structure

Construct an endpoint-sensitivity landscape using the same patients, folds, preprocessing discipline, and matched controls:

- absolute annual growth: 50 / 100 / 200 mm³/year;
- simple relative annual growth: 10% / 20% / 30%/year;
- cumulative relative growth (e.g. 20%);
- literature-motivated size-adaptive / measurement-aware definitions where implementation is sufficiently specified;
- continuous absolute and relative growth as complementary non-thresholded endpoints.

For each endpoint, compare:

1. constant / prevalence control;
2. volume-only;
3. B1 clinical model;
4. geometry control;
5. MRI model;
6. MRI incremental value over matched controls.

Report patient-clustered uncertainty and avoid choosing an endpoint because it yields the highest development AUC.

## Interpretation rules

- Higher AUC under one endpoint does not mean that endpoint is clinically superior.
- A strong volume baseline under an absolute endpoint does not by itself imply a biological size-growth relationship.
- A weak relative-growth model can reflect biological unpredictability, limited N, label noise, or measurement sensitivity.
- MRI improvement in one size region is a hypothesis until a formal interaction / replication supports heterogeneity.
- No endpoint should be retrospectively selected to rescue a model.

## Current novelty estimate

- “Different thresholds produce different AUCs”: low-to-moderate novelty.
- “Fixed 20% is imperfect / measurement error depends on size”: low novelty.
- “Endpoint construction changes baseline-size predictability”: moderate-to-high novelty in VS if shown rigorously.
- “Endpoint × size × MRI incremental-value landscape under matched evaluation”: high relative novelty based on the literature reviewed so far.

The project should therefore prioritize **predictor stability across defensible endpoint constructions**, not architecture search alone.

## Relevant literature anchors

- Marinelli et al., large sporadic VS natural-history study using cumulative relative volumetric growth criteria.
- Cornelissen et al., measurement-agreement work showing tumor-size-dependent relative volumetric uncertainty.
- Cornelissen et al. 2026, computer-aided VS growth prediction using size-dependent growth thresholds and repeated confirmation.
- Recent systematic reviews of VS growth-definition heterogeneity and MRI growth-prediction studies.

This note records a novelty / framing decision, not a confirmatory scientific result.