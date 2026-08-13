# Prompt log - fusion plot filename collision silently overwriting a figure

## What I wanted
A separate, correctly-named saved PNG for each of the three fused
funds' base-vs-fused growth comparison.

## Prompt(s)
Asked for plot_fusion() to be called once per fund in the Station 3c
loop, saving each to its own file in results/figures/.

## What the assistant produced
FIGURES_DIR / f"fusion_{family.lower()}_max_sharpe.png"
where family = fund.split()[0] (e.g. "Combined")

## What was wrong or risky
family is only the fund's asset universe, not its optimisation method -
both "Combined Maximum-Sharpe" and "Combined Minimum-Variance" produce
family="Combined", so both computed the identical output filename
fusion_combined_max_sharpe.png. The second fund processed in the loop
(Minimum-Variance) silently overwrote the first fund's saved plot with
no warning or error - I only caught this because I expected 3 files in
results/figures/ and found 2.

## What I changed and why
I have changed the way file names were being generated, to use the full name of 
the fund instead of the prefix of the asset universe, making sure that all
combinations of universes and methods of optimization have unique file paths 
associated with them. That such a clash was made without any warning whatsoever 
shows that just because a script executes without crashing does not mean it is 
logically correct; it is shown that quality assurance needs to extend beyond just 
looking for runtime crashes, but also validate outputs explicitly, like checking the
file count, among others.
