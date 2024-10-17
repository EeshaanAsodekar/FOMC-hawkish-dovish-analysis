# FOMC Hawkish-Dovish Sentiment Analysis

## Overview
This project evaluates the hawkish and dovish sentiment across FOMC meeting minutes, Fed speeches, and Fed Chair press conferences. Using NLP techniques such as dictionary-based TFIDF hawkish/dovish scoring and cosine similarity, the sentiment is quantified, and its relationship with key financial indicators like bond yields, yield spreads, and equity markets is explored.

## Data and Hawkish/Dovish Scoring Approach Implemented
### Data Sources: 
- [FOMC meeting minutes](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/FOMC/meeting_minutes)
- [FOMC statements](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/FOMC/statements)
- [Fed Governor speeches](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/fed_speeches)
- [Fed Chair Press conferences](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/fomc_press_conf/texts)

We analyze all these Fed transcripts from 2012 to 2024, the reason being the distinct shift in the manner of communication by the Fed post-2012; the communication became more direct and transparent.

### Hawkish/Dovish Scoring: 
1. Dictionary-Based Approach
    - Custom dictionaries [link](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/processed/hawkish_gpt_dict.txt) were created with hawkish and dovish terms to score sentiment within Fed communications. The composite score measures the relative balance of hawkish vs. dovish sentiment using:
    **Composite Score = (h_score - d_score) / (h_score + d_score)**
    - This approach applies a **TF-IDF-based** analysis to assess how sentiment shifts align with movements in key market variables.
2. Cosine similarity-based sentiment analysis.
    - This method uses **FinBERT embeddings** to compute **cosine similarity** between new text data and pre-defined [hawkish/dovish phrases](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/factor_similarity.py), adding a semantic dimension to sentiment tracking.

## Analysis on Hawkish/Dovish Score changes and Market moves:
Quintile-based visualizations and Regression Analysis show how shifts in hawkish sentiment (via hawkish/dovish scores by both methods) influence moves of these [market indicators](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/raw/FOMC_Data_2011_2024.xlsx):
- Treasury yields (2Y, 10Y)
- Yield curve spreads (2s10s)
- Gold
- VIX
- S&P 500 Index

## Running Instructions:
1. **Getting FOMC meeting minutes and statements:** Run the [FOMC_minutes_statements_scraper.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/FOMC_minutes_statements_scraper.py) and [FOMC_minutes_statements_processing.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/FOMC_minutes_statements_processing.py) to get a processed version of the FOMC Meeting Minutes and Statements, which is stored in the [data/processed/](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/processed) folder.

2. **Getting Fed Chair Press Conference Transcripts:** Run the [press_conference_scraper.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/press_conference_scraper.py) to get all the Fed Chair FOMC press conference transcripts.

3. **Getting Fed Governor's Speeches' Transcripts:** Run the [fed_speeches_scraper.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/fed_speeches_scraper.py) to get all the Fed Governor speeches' transcripts.

4. **Getting the Dictionary based Hawkish/Dovish Scores:** Run the [dictionary_based_analysis.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/dictionary_based_analysis.py) to get the dictionary-based hawkish/dovish scores for all the Fed communications we extracted.

5. **Getting the Cosine Similarity based Hawkish/Dovish Scores:** Run the [factor_similarity.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/factor_similarity.py) to get the factor similarity approach-based hawkish/dovish scores for all the Fed communications.

6. **Visualizing the Fed Hawkish/Dovish Sentiment vs Market moves:** Run the [results.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/results.py) to get the quintile plots of the Fed Hawkish/Dovish sentiment vs the market moves. Quintile plots are chosen so that we can analyze the overall trend in the hawkish/dovish scores and the market variables, to decide if the market moves make intuitive sense to facilitate further analysis or tweaking the models/dictionaries.

7. **Performing Statistical Analysis on the hawkish/dovish sentiment and the market moves:** Run the [regression_analysis.py](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/src/regression_analysis.py) to get R-squared and other parameters to gauge the significance of the hawkish/dovish score and the market moves.

## Key Results Obtained
### Dictionary Based Approach
1. **Moves in the VIX against change in the hawkish-score-1**
![Hawkish-score-1 v/s VIX Moves](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/results-vizl/Hawkishness-score-1/dict-hawkish-scored_Fed-chair-press-conf%20based%20Median%205-Day%20Cumulative%20VIX_pct_change%20Across%20Hawkishness-score-1%20Quintiles.png)
**Intuitive underlying relationship:** As the Fed’s hawkishness increases, there is an increase in volatility. The intuitive logic being that tightening monetary policy leads to a decrease in asset prices. This makes sense because a big move in hawkishness translates to a big potential Fed rate hike, which then has its effects on the market valuations, which in turn leads to a correction, thereby increasing the VIX.

2. **Moves in the 2 yr 10yr spread against change in the composite-score**
![composite-score v/s 2yr 10yr spread](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/results-vizl/Composite-score/composite-scored_FOMC-meeting-minutes%20based%20Median%205-Day%20Cumulative%202s10s_Spread_pct_change%20Across%20Composite-score%20Quintiles.png)
**Intuitive underlying relationship:** As the composite score (measuring hawkishness vs dovishness) increases, we observe a widening 2s10s spread, indicating that higher hawkish sentiment drives up short-term interest rates more significantly than long-term rates. This reflects the market's expectation of aggressive short-term rate hikes and a more moderate increase in long-term yields due to concerns about a future economic slowdown, which intuitively aligns with hawkish Fed policies leading to tighter monetary conditions.

3. **Moves in the S&P500 against change in the hawkish-score-2**
![hawkish-score-2 v/s S&P500](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/results-vizl/Hawkishness-score-2/dict-hawkish-scored_FOMC-statements%20based%20Median%205-Day%20Cumulative%20SP500_pct_change%20Across%20Hawkishness-score-2%20Quintiles.png)
**Intuitive underlying relationship**: A more hawkish stance typically signals the potential for tighter monetary policy, which increases borrowing costs, reduces corporate profits, and leads to lower equity valuations, thus causing negative impacts on the SP500. The pattern in this graph aligns with the idea that moderate to high hawkishness signals typically lead to market corrections, while mild hawkish stances may still support market optimism.

We can see that we get significant and intuitive results from the prototyped dictionaries and further refinement for these dictionaries would lead to stronger results and more intuitive alignment with the movements in the market.

### Cosine similarity Based Approach
1. **Moves in GT2 against hawk-sim-score**
![hawk-sim-score v/s GT2](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/results-vizl/hawk-sim-score/dict-hawkish-scored_FOMC-meeting-minutes%20based%20Median%205-Day%20Cumulative%20GT2_pct_change%20Across%20hawk-sim-score%20Quintiles.png)
**Intuitive underlying relationship:** The hawk-sim score shows that as hawkishness increases, short-term interest rates (represented by the GT2 yield) react accordingly. Initially, with lower hawkishness (quintiles 1 and 2), there is a decline in GT2 rates as expectations for aggressive tightening remain muted. However, as hawkish sentiment intensifies (quintiles 3 to 5), market participants anticipate rate hikes, driving up GT2 yields. The sharp increase in the higher quintiles reflects heightened expectations of tighter monetary policy, pushing short-term rates upward in response to the anticipated Fed actions.

2. **Moves in the 2 yr 10yr spread against hawk-sim-score**
![hawk-sim-score v/s GT2 GT10 spread](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/blob/main/data/results-vizl/hawk-sim-score/dict-hawkish-scored_Fed_speeches%20based%20Median%205-Day%20Cumulative%202s10s_Spread_pct_change%20Across%20hawk-sim-score%20Quintiles.png)
**Intuitive underlying relationship:** As the hawkishness score increases from the first to the second quintile, there is a sharp rise in the 2s10s spread. This could indicate that initial hawkish signals lead to a steepening yield curve, possibly because investors expect higher short-term rates but do not yet fully price in longer-term rate hikes. However, as the hawkishness increases further (from quintiles 3 to 5), the effect on the yield curve flattens or even slightly reverses, likely due to expectations that the economy may slow down, reducing the longer-term yields. This fits with the idea that extreme hawkishness, while signaling higher short-term rates, also signals reduced economic growth prospects, which can dampen longer-term yields, thus flattening or narrowing the 2s10s spread.