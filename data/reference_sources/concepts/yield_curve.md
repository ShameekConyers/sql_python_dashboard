# The Yield Curve and Economic Forecasting

The yield curve plots U.S. Treasury bond yields across maturities, from short-term bills to 30-year bonds. Under normal conditions the curve slopes upward: investors demand higher yields for locking up money longer, reflecting expectations of future growth and inflation.

## The 10-Year Minus 2-Year Spread

The most-watched recession signal is the spread between the 10-year and 2-year Treasury yields, tracked by FRED series T10Y2Y. When this spread turns negative the curve is "inverted," meaning short-term rates exceed long-term rates. An inversion signals that bond markets expect weaker growth ahead, anticipating that the Federal Reserve will eventually cut short-term rates in response to a slowdown.

## Historical Track Record

Every U.S. recession since 1970 was preceded by a yield curve inversion, typically 6 to 24 months before the NBER-dated start. The false-positive rate is low but nonzero: a brief inversion in late 1998 was not followed by a recession until 2001, roughly 2.5 years later.

The spread inverted in mid-2022 and remained negative for over two years, the longest sustained inversion since the early 1980s. As of early 2025 the spread returned to positive territory, consistent with the historical pattern where the curve re-steepens before a recession actually begins.

## T10Y2Y in This Dashboard

This dashboard tracks T10Y2Y as a daily series. The COVID ARIMA adjustment replaces the March 2020 through January 2022 window with a counterfactual, smoothing the sharp drop that reflected emergency Fed rate cuts rather than a normal business-cycle signal. The recession probability model uses the yield spread as one of its 11 features.

A positive and rising spread generally suggests markets have priced in the risk of a downturn and now expect recovery, while a persistently negative spread elevates recession probability in the model output.
