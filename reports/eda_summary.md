# PaddyWise AI — Phase 2 EDA Summary

## Scope
The EDA uses the actual processed dataset at `data\processed\master_dataset.csv`: **3,024 rows × 21 columns**, covering **2017-06-07 to 2025-03-17**.

## Key findings
- The dataset contains **8 markets**, **9 varieties**, and **2 grades**. The dominant variety is **B P T** (2,567 records).
- Mean modal price is **₹1,928.80/quintal**; median is **₹1,900**.
- Mean modal price by year moves from **₹1,991.44** in 2017 to **₹2,424.63** in 2025; this describes the observed sample and is not a forecast.
- By historical mean modal price, **Thanjavur** has the highest observed market mean (₹2,241.67/quintal), while **Papanasam** has the lowest (₹1,739.82/quintal).
- Market-level standard deviation is highest for **Papanasam** (₹477.11/quintal) among these raw market observations.
- Monthly mean modal price is highest in **August** (₹2,019.80) and lowest in **May** (₹1,869.60).
- Weather-to-price Pearson correlations are small in magnitude in this dataset; the largest absolute correlation among the listed weather variables is **0.041** for **temperature_max**. Correlation does not establish causation.
- Missingness is concentrated in `district_y`, `state`, `latitude`, and `longitude`, each with **182** missing records. The price target has **0** missing values.

## Suspicious Vallam observation
The preserved observation is record **2893**, dated **2024-07-22**, with Vallam/B P T/Local prices of **₹2,200 minimum, ₹2,300 modal, and ₹24,000 maximum**. The row satisfies the supplied ordering rule `min ≤ modal ≤ max`, but ₹24,000 is an extreme value relative to Vallam peers. It is retained unchanged for later investigation/modeling decisions.

## Interpretation guardrails
EDA findings are descriptive. They do not establish that weather causes prices, that one market will outperform another in the future, or that an observed high/low price will persist. No observations were deleted or corrected during Phase 2.
