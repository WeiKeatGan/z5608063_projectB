# Prompt log - Streamlit "Build your allocation" $nan bug

## What I wanted
An interactive tab letting a user blend multiple funds into a custom
allocation and see the resulting growth/Sharpe/drawdown.

## Prompt(s)
Asked for a portfolio blender: multiselect funds, sliders for weights,
pivot fund_returns.csv to wide, sum weighted daily returns.

## What the assistant produced
wide = fr.pivot(index="date", columns="fund", values="daily_return")
port = sum(w[f] * wide[f] for f in chosen)
met = _metrics_from_daily(port)  # PERIODS hardcoded to 252

## What was wrong or risky
Selecting a mix of Combined/Equity funds (252-day calendar) with a
Crypto fund (365-day, trades weekends) produced a literal `$nan` growth
figure. wide's date index is the UNION of both calendars, so weekend
dates have real crypto data but no equity data, and pd.Series addition
propagates NaN - which landed on the LAST date in the index, breaking
growth.iloc[-1]. Separately, PERIODS=252 was hardcoded regardless of
which funds were actually selected, silently misannualising an
all-crypto allocation.

## What I changed and why
I have changed the dataframe pivot logic to utilise the strict intersection of dates
(filtering out all the rows containing any missing values), as this will ensure that
the portfolio will calculate its blend only when each selected fund traded on that day, 
thus avoiding the weekend NaNs interfering with the mathematics and messing up the 
cumulative growth calculation. Furthermore, the hardcoding of the number of 
PERIODS=252 was substituted with a dynamically calculated periods per year based on 
the length of the intersected period range, as this is crucial for the custom portfolio tool, 
since it will ensure the accuracy of annualized metrics (Sharpe and annualized return, 
in particular) in case of a crypto-only portfolio (~365 trading days). 
