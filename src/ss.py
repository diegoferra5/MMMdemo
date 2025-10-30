# %% [markdown]
# # Load Data

# %%
import pandas as pd
sample_data = pd.read_csv("../data/raw/train.csv")

# %%
sample_data.head()

# %%
sample_data.info()

# %%
sample_data.describe()

# %%
sample_data.columns

# %%
print(sample_data.duplicated().sum())

# %% [markdown]
# # Transformation

# %% [markdown]
# Date tranformation

# %%
sample_data["Date"]= pd.to_datetime(sample_data["Date"], errors= "coerce")
sample_data["Date"].dtype


# %% [markdown]
# Filter only open stores

# %%
sample_data=sample_data.loc[sample_data["Open"]== 1]
sample_data.info()

# %% [markdown]
# Agrupar por semana

# %%
sales_weekly= sample_data.groupby(pd.Grouper(key="Date", freq= "W"))["Sales"].sum().reset_index()

sales_weekly.head()

# %%
sales_weekly.info()

# %% [markdown]
# Plot the weekly sales over the time

# %%
import matplotlib.pyplot as plt

# Data 
x= sales_weekly["Date"]
y= sales_weekly["Sales"]

# Plot
plt.figure(figsize=(12,6))
plt.plot(x,y)

#Labels

plt.xlabel("Date /week")
plt.ylabel("Total Sales")
plt.title("Weekly Sales")
plt.xticks(rotation=45)





plt.show()

# %% [markdown]
# Seasonal pattern is clearly visible. We can see clear peaks and valleys repeating roughly every few months.
# That’s classic retail seasonality, usually tied to:
# 
# holidays (Christmas / New Year)
# 
# local events
# 
# trade promotions or product launches
# 
# Big spikes — that tall one around early 2014 is likely a major holiday campaign (Christmas 2013 → Jan 2014 sales).
# 
# Mid-year dips — notice how sales often soften mid-year (typical for FMCG/beer products when there’s no big event).

# %% [markdown]
# ## Aggregating the Promo variable

# %%
sample_data

# %% [markdown]
# Now we want to aggregate the promo variable.
# Each row in the raw data is one store-day.
# 
# Promo = 1 means that store had a promotion running that day.
# 
# Promo = 0 means it didn’t.
# 
# 
# Our goal now:
# Transform all those store-day 0/1 flags into one weekly value
# that summarizes “how intense promotions were” that week, across the chain.
# 
# Use the mean.
# Why?
# 
# MMM models work better with continuous proportions (0 → 1) than with raw counts.
# 
# It reflects the intensity of promotion activity (e.g., 0.3 = 30% of stores had promos).
# 
# It’s comparable across time even if the number of stores changes.
# 
# So the plan is to compute:
# 
# Weekly average of Promo (percentage of stores on promotion).

# %%
promos_weekly= sample_data.groupby(pd.Grouper(key="Date", freq="W"))["Promo"].mean().reset_index()
promos_weekly.head()

# %%
# Data 
x= promos_weekly["Date"]
y= promos_weekly["Promo"]

# Plot
plt.figure(figsize=(12,6))
plt.plot(x,y)

#Labels

plt.xlabel("Date /week")
plt.ylabel("Total Sales")
plt.title("Weekly Sales")
plt.xticks(rotation=45)





plt.show()

# %% [markdown]
# Merge sales + promo dfs

# %%
promos_sales_weekly= pd.merge(sales_weekly,promos_weekly, on="Date")
promos_sales_weekly.head()

# %%
# Visualize correlation between sales and promos 
# Example data
x = promos_sales_weekly["Promo"]
y = promos_sales_weekly["Sales"]

# Create scatter plot
plt.scatter(x, y)

# Add labels and title
plt.xlabel("Promo")
plt.ylabel("Sales")
plt.title("Simple Scatter Plot")

# Show the plot
plt.show()

# %% [markdown]
# Promotional weeks produce a strong incremental lift in total weekly sales

# %% [markdown]
# ## Add calendar & seasonal features 
# 
# Goal: enrich the dataset with time-based control variables that capture natural sales variation.
# To avoid false attribution, ensuring sales peaks aren’t wrongly credited to promotions.
# 
# To separate natural cycles (like holidays or weather) from the true marketing or promo impact.
# 
# To make the model’s ROI estimates realistic and unbiased across time.

# %%
# Extract month, year, and week number, trend(count var).

promos_sales_weekly["Year"]= promos_sales_weekly["Date"].dt.year
promos_sales_weekly["Month"]= promos_sales_weekly["Date"].dt.month
promos_sales_weekly["Week_num"]= promos_sales_weekly["Date"].dt.isocalendar().week
promos_sales_weekly["trend"]= promos_sales_weekly.index 
promos_sales_weekly["trend"]+=1

promos_sales_weekly.head()


# %%
# plot sales vs trend
plt.figure(figsize=(12,6))
x = promos_sales_weekly["trend"]
y = promos_sales_weekly["Sales"]

# Create scatter plot
plt.plot(x, y)

# Add labels and title
plt.xlabel("trend")
plt.ylabel("Sales")
plt.title("Simple Scatter Plot")

# Show the plot
plt.show()

# %% [markdown]
# ## Adding calendar control variables
# - StateHoliday
# 
# - SchoolHoliday
# 
# Then we’ll aggregate them by week, just like we did with Promo, and merge them into the weekly dataset.
# 
# StateHoliday = big national events → structural impact on store openings and traffic.
# 
# SchoolHoliday = local/regional seasonality → shifts in family-related consumption.
# 
# Both explain natural demand fluctuations unrelated to marketing or promos — so adding them helps your model avoid falsely crediting ads or promotions for those changes.

# %%
sample_data

# %% [markdown]
# - StateHoliday	max()	If any day in the week is a public/state holiday → mark the week as 1
# - SchoolHoliday	mean()	Share of days in that week affected by school closure (0–1)

# %%
# Convert 'a', 'b', 'c' → 1
# Convert '0' → 0
sample_data["StateHoliday"] = sample_data["StateHoliday"].isin(["a","b","c"]).astype(int)

holidays_weekly = (
    sample_data.groupby(pd.Grouper(key="Date", freq="W"))
    .agg({
        "StateHoliday": "max",     # if any holiday in the week → 1
        "SchoolHoliday": "mean"    # proportion of days with school closure
    })
    .reset_index()
)

holidays_weekly.head()

# %% [markdown]
# Merge with promos_sales_weekly

# %%
weekly_sales = pd.merge(promos_sales_weekly, holidays_weekly, on= "Date")
weekly_sales.head()
#weekly_sales.info()

# %%
weekly_sales.isna().sum()
weekly_sales.describe()


# %% [markdown]
# # Baseline Model

# %% [markdown]
# At this stage, we build a multiple linear regression (OLS) using statsmodels to explain the drivers of weekly sales. The model has a log-linear form:

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# 
# 
# This regression quantifies how each factor (promotions, holidays, trend) influences sales while holding the others constant.
# 
# Why linear regression?
# 
# - It captures marginal effects — how much sales change when a variable changes.
# - It’s interpretable, showing direction, strength, and significance of each driver.
# - It serves as the foundation for the full MMM, which will later include media variables and non-linear transformations.
# 
# Why use statsmodels instead of scikit-learn?
# statsmodels provides detailed statistical output — coefficients, p-values, confidence intervals, and model diagnostics — essential for causal interpretation, not just prediction accuracy.
# 
# Why this baseline model matters now:
# - It validates that the data pipeline and logic are correct.
# - It confirms variables behave as expected (e.g., Promo increases sales).
# - It establishes a benchmark before adding marketing spend and advanced MMM effects like adstock and saturation.

# %%
import numpy as np
import statsmodels.api as sm

y = np.log(weekly_sales["Sales"])
X = weekly_sales[["Promo", "StateHoliday", "SchoolHoliday", "trend"]] 
X= sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())



# %% [markdown]
# Business interpretation summary
# 
# Promo (main driver):
# - +0.3578 on log scale → roughly +35.8% sales lift during promotional weeks.
# - Very significant (p < 0.001).
# - Confirms the promo data is correct and meaningful.
# 
# Holidays:
# - State holidays don’t matter much → likely because stores close or customers shift purchases.
# - School holidays slightly reduce demand (−11.9%), likely reflecting behavior patterns unrelated to marketing.
# 
# Trend:
# - Flat — your sales pattern doesn’t drift much once you account for promos and holidays.
# 
# *We’re still building understanding, not optimizing performance.*

# %% [markdown]
# # Generate realistic media spend for TV, Digital, and OOH

# %%
N= len(weekly_sales)
rng = np.random.default_rng(42)

# %% [markdown]
# ## Digital channel
# 
# 1.1 Definir un gasto promedio semanal y una dispersión razonable. Solo base, sin ruido relativo ni boosts todavía.
# 

# %%

base_mean = 70_000   # weekly "always-on"
base_sd   = 7_000    # dispersión semanal del base
min_spend = 40_000
max_spend = 120_000

base_spend = rng.normal(loc=base_mean, scale=base_sd, size=N) #array de N valores que simulan gastos base con cierta variabilidad. campana de gauss
base_spend = np.clip(base_spend, a_min=min_spend*0.7, a_max=max_spend*0.9) #recorta valores extremos

weekly_sales["Digital_spend"] = base_spend 
weekly_sales[["Date","Digital_spend"]].head(10) 



# %% [markdown]
# 1.2 Weekly relative noise (pacing / optimization effects)

# %%
# Relative weekly noise (~±10%), capped at ±25%
noise_pct_sd = 0.10
rel_noise = rng.normal(loc=0.0, scale=noise_pct_sd, size=N)
rel_noise = np.clip(rel_noise, -0.25, 0.25)

# Apply multiplicative noise and clip back to realistic bounds
weekly_sales["Digital_spend"] = weekly_sales["Digital_spend"] * (1.0 + rel_noise)
weekly_sales["Digital_spend"] = np.clip(
    weekly_sales["Digital_spend"], a_min=min_spend, a_max=max_spend
)
weekly_sales[["Date", "Digital_spend"]].head(8)


# %%
weekly_sales["Digital_spend"].plot(title="Digital_spend (base + noise)", figsize=(9,3))


# %% [markdown]
# 1.3 Promo & Holiday boosts
# 
# Make the Digital_spend more realistic by increasing it during weeks with in-store promotions or holidays.
# Digital campaigns typically spend more when there’s a coordinated retail push or higher consumer activity, so we introduce controlled boosts.

# %%
# Convert your existing weekly StateHoliday to a clean 0/1 flag
holiday_flag = (~weekly_sales["StateHoliday"].astype(str).isin(["0", "0.0"])).astype(int)
# Promo flag (just to be sure it’s clean)
promo_flag = (weekly_sales["Promo"].fillna(0).astype(float) > 0).astype(int)

# Random boost ranges (realistic)
promo_lo, promo_hi = 0.20, 0.40   # +20–40% spend on promo weeks
holi_lo, holi_hi   = 0.10, 0.20   # +10–20% spend on holiday weeks

# Generate multipliers (1.0 means “no boost”)
promo_mult = 1.0 + promo_flag * rng.uniform(promo_lo, promo_hi, size=len(weekly_sales))
holi_mult  = 1.0 + holiday_flag * rng.uniform(holi_lo, holi_hi, size=len(weekly_sales))

# Combine both and cap to 1.5× total
mult_total = np.clip(promo_mult * holi_mult, 1.0, 1.5)

# Apply to Digital_spend
weekly_sales["Digital_spend"] = weekly_sales["Digital_spend"] * mult_total
weekly_sales["Digital_spend"] = np.clip(
    weekly_sales["Digital_spend"], a_min=min_spend, a_max=max_spend
)
weekly_sales[["Date", "Promo", "StateHoliday", "Digital_spend"]].head(10)



# %% [markdown]
# Created binary flags for both events:
# 
# holiday_flag → 1 if StateHoliday ≠ 0 (meaning a real holiday week).
# 
# promo_flag → 1 if Promo > 0.
# These ensure the model can detect active promotional or seasonal periods.
# 
# Defined realistic boost ranges:
# 
# Promo weeks: +20% to +40% increase in digital spend.
# 
# Holiday weeks: +10% to +20% increase.
# 
# Generated random multipliers within those ranges using the NumPy random generator.
# 
# If no promo/holiday: multiplier = 1.0 (no change).
# 
# If active: multiplier randomly chosen in the defined range.
# 
# Combined and capped the effect:
# 
# Both boosts were multiplied together.
# 
# The total multiplier was clipped to a maximum of 1.5×, to prevent unrealistic spikes.
# 
# Applied the boost to Digital_spend and re-clipped to keep values between the minimum (40k) and maximum (120k).
# 
# This process adds realistic campaign-driven peaks to the digital spend time series while maintaining overall control over its scale.

# %%
weekly_sales["Digital_spend"].plot(
    figsize=(10,3),
    title="Digital_spend after Promo + Holiday boosts"
);


# %% [markdown]
# ## Creating a realistic CPM (Cost Per Thousand Impressions)

# %% [markdown]
# What CPM represents
# 
# It’s the price you pay per 1 000 ad impressions.
# So when the market is more competitive — for instance in Q4 holidays — your CPM goes up, meaning the same spend buys fewer impressions.
# 
# We’ll make CPM vary with:
# 
# Seasonality – higher in Q4 and slightly higher in summer.
# 
# Small random noise ± 7 %.

# %%
# --- Step 2A: CPM (seasonality + noise) ---
base_cpm = 6.0

# Extract month number from weekly dates
month = pd.to_datetime(weekly_sales["Date"]).dt.month

# Define simple seasonal uplift:
# Q4 (Oct–Dec) more expensive, Summer (Jun–Aug) moderately higher
season_uplift = np.where(
    month.isin([10, 11, 12]), 1.25,
    np.where(month.isin([6, 7, 8]), 1.10, 1.00)
)

# Random week-to-week noise (~±7%), capped at ±20%
cpm_noise = np.clip(rng.normal(0.0, 0.07, size=N), -0.20, 0.20)

# Final CPM = base * seasonal * noise
weekly_sales["CPM"] = base_cpm * season_uplift * (1.0 + cpm_noise)


# %%
weekly_sales["CPM"].describe()

# %%
weekly_sales.plot(
    x="Date",
    y="CPM",
    figsize=(10,3),
    title="CPM (seasonality + noise)"
);


