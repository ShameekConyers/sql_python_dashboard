# CPI and Inflation Measurement

The Consumer Price Index for All Urban Consumers (CPI-U), tracked as FRED series CPIAUCSL, measures the average change over time in prices paid by urban consumers for a market basket of goods and services. It is the most widely cited inflation gauge in the United States.

## What CPI Measures

The BLS prices approximately 80,000 items each month across 75 urban areas, covering about 93% of the U.S. population. The basket includes categories like food, housing, transportation, medical care, education, and recreation. Each category carries a weight reflecting its share of consumer spending, updated every two years using Consumer Expenditure Survey data.

CPIAUCSL is an index with a base period of 1982-1984 equal to 100. A value of 310 means the basket costs roughly 3.1 times what it cost in the base period. The "S" in CPIAUCSL stands for "seasonally adjusted," removing predictable patterns like higher energy prices in winter.

## Month-over-Month vs. Year-over-Year

Month-over-month (MoM) CPI change captures short-term price movements but is noisy. Year-over-year (YoY) change compares the current index level to the same month one year earlier, smoothing seasonal effects and providing the inflation rate that policymakers and markets focus on. A YoY rate of 3.2% means prices are 3.2% higher than 12 months ago.

## Headline vs. Core CPI

Headline CPI includes all items. Core CPI excludes food and energy because those prices are volatile and driven by supply shocks (weather, oil markets) rather than underlying demand-side inflation. The Federal Reserve watches both but considers core CPI a better signal of persistent inflation trends when setting interest rates.

## CPIAUCSL in This Dashboard

This dashboard stores monthly CPIAUCSL values and computes YoY percentage changes for trend analysis. The COVID ARIMA adjustment replaces the March 2020 through January 2022 window with a counterfactual to smooth pandemic-era distortions like the sharp deflation in spring 2020 followed by supply-chain-driven price spikes in 2021. The recession model uses CPI YoY change as one of its features.
