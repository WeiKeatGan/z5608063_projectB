# Prompt log - Report Section 2: reversed Combined Maximum-Sharpe claim

## What I wanted
An accurate opening paragraph for Section 2 summarising the headline
out-of-sample results across all 12 funds.

## Prompt(s)
Drafted the results summary from memory of the numbers discussed earlier
in the conversation, without re-checking performance_metrics.csv directly.

## What the assistant produced (my own draft, reviewed)
My draft stated Combined Maximum-Sharpe "produced the worst outcome
across all 12 funds (Sharpe 0.40)" and cited several other Sharpe values
that didn't match the actual results table.

## What was wrong or risky
Checking against results/tables/performance_metrics.csv, Combined
Maximum-Sharpe was actually my BEST-performing fund (Sharpe 1.092, the
highest of all 12) - the exact opposite conclusion. Several other cited
numbers (Crypto Minimum-Variance, Crypto Risk Parity, Equity Equal
Weight, Combined Risk Parity) were also incorrect. This happened because
I wrote from memory of a long conversation rather than the source file
directly.

## What I changed and why
The new summary that I came up with is one where I sourced all the statistics from 
performance_metrics.csv since it was impossible not to get confused with the figures
if the work involved too much data and took place over a lengthy period. It is the
only way to make sure the results of the analysis were based on factual evidence. 
