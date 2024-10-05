import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertModel.from_pretrained('yiyanghkust/finbert-tone')

# Define Hawkish and Dovish sentences
HAWKISH_SENTENCES = [
    "Interest rates will rise.",
    "The economy is overheating.",
    "We will tighten monetary policy.",
    "Inflation is high.",
    "We are concerned about rising prices.",
    "Yields in most advanced foreign economies also rose sharply as central banks continued to lift policy rates.",
    "Nominal wage growth continued to be rapid and broad-based.",
    "We will continue to tighten monetary policy to address inflation pressures.",
    "Inflation is expected to remain elevated over the near term.",
    "Labor demand remained strong, and the labor market continued to be very tight.",
    "The expected path of the federal funds rate rose, reflecting restrictive monetary policy.",
    "Credit standards tightened, reflecting strong demand and rising interest rates.",
    "Energy prices remained considerably higher, and upside risks to inflation remained.",
    "Participants noted that inflationary pressures persisted, supporting the need for further rate hikes.",
    "Participants stressed the importance of maintaining a restrictive policy stance to contain inflation.",
    "The continued increase in core goods prices underscores the need for decisive policy action.",
    "The Committee agreed that additional tightening would be necessary to restore price stability.",
    "Increased borrowing costs have dampened the demand for credit, but inflation remains a key concern.",
    "Real interest rates have increased significantly, indicating the tightening of financial conditions.",
    "While inflation data has shown occasional signs of relief, the broader trend suggests sustained upward pressure, requiring further decisive actions.",
    "We expect a prolonged period of economic adjustment as we work to ensure inflation does not become entrenched.",
    "Economic growth is expected to remain below trend, as supply and demand continue to rebalance over the coming quarters.",
    "Despite tightening financial conditions, inflation remains far from our target, necessitating ongoing policy restraint.",
    "The path ahead for inflation remains uncertain, and we may need to raise rates higher than initially anticipated.",
    "Inflationary pressures have proven more persistent than expected, justifying further restrictive monetary policy.",
    "Labor market tightness continues to drive nominal wage growth, amplifying inflationary risks that we need to address.",
    "We remain vigilant to prevent inflation expectations from becoming unanchored, even if it requires sustained policy intervention.",
    "Slowing demand across interest-sensitive sectors signals the effectiveness of our tightening, but more needs to be done.",
    "The economy will likely experience below-trend growth as we continue efforts to reduce inflationary pressures.",
]

DOVISH_SENTENCES = [
    "Interest rates will fall.",
    "The economy is slowing down.",
    "We will implement quantitative easing.",
    "Inflation is low.",
    "We are concerned about slow growth.",
    "Weaker growth in the global economy weighed on export-oriented markets.",
    "We will implement quantitative easing to support economic growth.",
    "The unemployment rate edged up, and the labor market showed signs of softening.",
    "The nominal U.S. trade deficit continued to narrow, contributing positively to GDP growth.",
    "Economic activity in the second half of this year is expected to grow at a modest pace.",
    "Longer-term inflation expectations remained stable or moved lower.",
    "Weaker growth in China and global supply chain bottlenecks weighed on economic activity.",
    "Participants judged that a softening in the labor market would help reduce inflationary pressures.",
    "Consumer price inflation is expected to moderate over the next two years.",
    "Global headwinds, such as weakening activity abroad, pose downside risks to growth.",
    "The Committee's commitment to maintaining a stable inflation target will remain flexible based on data.",
    "Inflation expectations have become more anchored, reducing the need for aggressive policy measures.",
    "A slowdown in residential investment reflects the anticipated easing of financial conditions.",
    "Recent declines in inflation, coupled with easing labor market conditions, indicate that our prior policy stance is achieving its intended goals.",
    "Growth in economic activity remains solid, with inflation gradually moving closer to our 2 percent objective.",
    "The risks to achieving both maximum employment and price stability now appear more balanced, justifying a recalibration of our policy.",
    "We are seeing growing evidence that inflationary pressures are diminishing, allowing for a more neutral policy stance.",
    "With inflation nearing our target, we are confident that a reduction in policy restraint will support continued economic stability.",
    "As inflation moves sustainably toward our goal, the downside risks to employment have become more prominent.",
    "The labor market has cooled from its overheated state, providing room for a more measured policy approach.",
    "With inflation having moderated substantially, we are poised to maintain employment gains while ensuring price stability.",
    "As inflation continues to ease, we will adjust our policy to prevent any undue weakening of economic activity.",
    "Ongoing reductions in the policy interest rate reflect our confidence in the economy's progress and a reduced need for aggressive tightening.",
]

# Function to generate embeddings from text using FinBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling to get single embedding

# Function to calculate similarity between text and hawkish/dovish sentences
def calculate_similarity(text, hawkish_sentences, dovish_sentences):
    text_embedding = get_embedding(text)

    hawkish_scores = [cosine_similarity(text_embedding, get_embedding(sentence)) for sentence in hawkish_sentences]
    dovish_scores = [cosine_similarity(text_embedding, get_embedding(sentence)) for sentence in dovish_sentences]

    avg_hawkish = np.mean(hawkish_scores)
    avg_dovish = np.mean(dovish_scores)

    return avg_hawkish, avg_dovish

# Main function to process CSVs and calculate factor similarity scores
def process_fomc_documents():
    # Load the cleaned data for Meeting Minutes and Statements
    minutes_df = pd.read_csv('data/processed/cleaned_meeting_minutes.csv')
    statements_df = pd.read_csv('data/processed/cleaned_statements.csv')

    # Add columns for storing Hawkish and Dovish scores
    minutes_df['Hawkish_Score'] = 0.0
    minutes_df['Dovish_Score'] = 0.0
    statements_df['Hawkish_Score'] = 0.0
    statements_df['Dovish_Score'] = 0.0

    # Calculate similarity scores for each row in the Meeting Minutes
    print("Processing FOMC Meeting Minutes...")
    for idx, row in minutes_df.iterrows():
        hawkish, dovish = calculate_similarity(row['Federal_Reserve_Mins'], HAWKISH_SENTENCES, DOVISH_SENTENCES)
        minutes_df.at[idx, 'Hawkish_Score'] = hawkish
        minutes_df.at[idx, 'Dovish_Score'] = dovish
        if idx % 10 == 0:  # Progress log
            print(f"Processed {idx + 1} meeting minutes.")

    # Calculate similarity scores for each row in the FOMC Statements
    print("Processing FOMC Statements...")
    for idx, row in statements_df.iterrows():
        hawkish, dovish = calculate_similarity(row['FOMC_Statements'], HAWKISH_SENTENCES, DOVISH_SENTENCES)
        statements_df.at[idx, 'Hawkish_Score'] = hawkish
        statements_df.at[idx, 'Dovish_Score'] = dovish
        if idx % 10 == 0:  # Progress log
            print(f"Processed {idx + 1} statements.")

    # Save the results back to CSV
    minutes_df.to_csv('data/processed/hawkish_dovish_meeting_minutes.csv', index=False)
    statements_df.to_csv('data/processed/hawkish_dovish_statements.csv', index=False)

    print("Factor similarity analysis complete. Results saved.")

if __name__ == "__main__":
    process_fomc_documents()
