# Prompt log - missing Streamlit theme causing unreadable UI

## What I wanted
The deployed app to match NoBearTrader's FT-inspired design system
(cream background, dark readable text) consistently across custom HTML
and Streamlit's own native widgets/charts.

## Prompt(s)
Asked for custom CSS matching the FT style used in the static report
figures, injected via st.markdown with unsafe_allow_html.

## What the assistant produced
A <style> block styling specific custom HTML classes (.nb-hero,
.nb-metric, headings) - no [theme] section in .streamlit/config.toml.

## What was wrong or risky
Injected CSS can only style elements it explicitly targets. Streamlit's
own native widget labels (slider captions, multiselect text) and native
charts (st.line_chart, st.bar_chart) are rendered through Streamlit's
own theme engine, which the CSS block never reaches - resulting in
washed-out, low-contrast labels and charts rendering on a black
background regardless of the custom cream styling elsewhere on the page.

## What I changed and why
The .streamlit/config.toml file that is created by me, which had a [theme] block for 
setting up the background color, text color, and primary color explicitly. This is 
how I fixed something that was not possible through CSS only since the color scheme 
of the React components built natively by Streamlit, and those of Altair-based charts
are dependent on the theme engine inside the framework and not on any custom style
injected using the DOM. It is important to fix this issue now since Streamlit 
automatically picks up the settings from the host operating system and an app 
that worked perfectly well in a local editor could turn out to be totally unreadable
in dark mode after deployment.

