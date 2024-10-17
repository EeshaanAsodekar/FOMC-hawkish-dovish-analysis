# FOMC Hawkish-Dovish Sentiment Analysis

## Overview
This project evaluates the hawkish and dovish sentiment across FOMC meeting minutes, Fed speeches, and Fed Chair press conferences. Using NLP techniques such dictionary based TFIDF hawkish/dovish scoring and cosine similarity, the sentiment is quantified and its relationship with key financial indicators like bond yields, yield spreads, and equity markets is explored.

## Data and Hawkish/Dovish Scoring Approach Implemented
### Data Sources: 
- [FOMC meeting minutes](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/FOMC/meeting_minutes)
- [FOMC statments](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/FOMC/statements)
- [Fed Governor speeches](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/fed_speeches)
- [Fed Chair Press conferences](https://github.com/EeshaanAsodekar/FOMC-hawkish-dovish-analysis/tree/main/data/raw/fomc_press_conf/texts)
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

