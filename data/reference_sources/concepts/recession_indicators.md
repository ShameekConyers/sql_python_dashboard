# Recession Indicators and Prediction

A recession is broadly understood as a significant decline in economic activity spread across the economy lasting more than a few months. The official U.S. arbiter is the National Bureau of Economic Research (NBER), which dates business cycle peaks and troughs using a committee judgment process.

## NBER Definition

The NBER Business Cycle Dating Committee considers real GDP, real income, employment, industrial production, and wholesale-retail sales. There is no fixed rule such as "two consecutive quarters of GDP decline." The committee often announces recession start and end dates well after the fact, sometimes 6 to 12 months later.

FRED series USREC is a binary indicator (1 during NBER-dated recessions, 0 otherwise) available monthly. This dashboard uses USREC as the target variable for its recession probability model.

## The Sahm Rule

The Sahm Rule triggers when the three-month moving average of U-3 unemployment rises 0.50 percentage points or more above its low from the prior 12 months. Developed by economist Claudia Sahm, it has identified every U.S. recession since 1970 in real time, with no false positives prior to the 2024 debate around its trigger without a subsequent NBER-dated recession.

## Yield Curve as a Leading Indicator

The 10Y-2Y Treasury spread (T10Y2Y) has preceded every recession since 1970 by inverting 6 to 24 months before the NBER-dated start. The mechanism is that bond markets price in expected Fed rate cuts ahead of an anticipated downturn. See the yield curve concept document for details.

## Model Features in This Dashboard

The recession probability model uses 11 features drawn from the dashboard's FRED series:

- Yield spread level and direction (T10Y2Y)
- Unemployment rates and their gaps (UNRATE, U6RATE)
- GDP growth rate (GDPC1)
- CPI year-over-year change (CPIAUCSL)
- Per-capita employment metrics (USINFO, CES2023800001 normalized by CNP16OV)
- Hacker News sentiment and layoff story frequency (post-2022 only)

## Logistic Regression vs. Random Forest

The dashboard trains both a logistic regression (LR) and a random forest (RF) classifier. LR provides interpretable coefficients and serves as a baseline. RF captures nonlinear interactions between features, such as the combined effect of a flat yield curve and rising unemployment. Both models produce monthly recession probabilities between 0 and 1, stored in the recession_predictions table. The dashboard displays both side by side so users can compare the linear and nonlinear estimates.
