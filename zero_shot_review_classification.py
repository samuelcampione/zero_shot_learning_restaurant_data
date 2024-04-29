import numpy as np
import pandas as pd
from transformers import pipeline


MODEL_CHECKPOINT = "facebook/bart-large-mnli"
REVIEWS_CSV_PATH = '/Users/scampione/MSDS/Spring_24_2/Entrepreneurship/philly_reviews_cleaned.csv'


def classify_review(text, classifier, candidate_labels):
    """
    Classify a text review into provided categories and return the scores.
    
    Args:
        text (str): The review text.
        classifier (pipeline): Pre-initialized Transformers pipeline for zero-shot text classification.
        candidate_labels (list): List of categories as strings.
        
    Returns:
        dict: Scores for each category if successful, empty dictionary otherwise.
    """
    try:
        classifications = classifier(text, candidate_labels, multi_label=True)
        labels = classifications['labels']
        scores = classifications['scores']
        return dict(zip(labels, scores))
    
    except Exception as e:
        print(f"Failed to process review: {str(e)}")
        return {}

def classify_restaurant(df, classifier, candidate_labels):
    """
    Process a DataFrame of reviews and aggregate their classification scores by business_id.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'business_id' and 'text'.
        classifier (pipeline): Pre-initialized Transformers pipeline.
        candidate_labels (list): List of target categories for classification.
    
    Returns:
        pd.DataFrame: DataFrame with average scores for each category per business_id.
    """
    results = {'business_id': []}
    for label in candidate_labels:
        results[label] = []

    # Classify each review and collect scores
    for _, row in df.iterrows():
        review_scores = classify_review(row['text'], classifier, candidate_labels)
        results['business_id'].append(row['business_id'])
        for label in candidate_labels:
            results[label].append(review_scores.get(label, 0))

    results_df = pd.DataFrame(results)
    
    # Average scores for each business_id
    avg_scores_df = results_df.groupby('business_id').agg({label: 'mean' for label in candidate_labels}).reset_index()
    return avg_scores_df



def main():
    classifier = pipeline('zero-shot-classification', model=MODEL_CHECKPOINT)
    try:
        reviews_df = pd.read_csv(REVIEWS_CSV_PATH, quotechar='"', escapechar='\\', on_bad_lines='skip')
    except Exception as e:
        print("An error occurred:", e)

    candidate_labels = [
        "Coworking Cafe——Coffee shop with people working on laptops, Wi-Fi available", 
        "Brunch——Outdoor brunch with breakfast and mimosas", 
        "Romantic Date Night——Romantic restaurant with candlelit tables", 
        "Upscale Special Occasion——Elegant restaurant with white tablecloths and fine dining", 
        "Rooftop——Rooftop bar or restaurant with city views", 
        "Pub——Lively bar with draft beer and sports TV", 
        "None of the above"
    ]

    aggregated_scores = classify_restaurant(reviews_df, classifier, candidate_labels)
    aggregated_scores.to_csv("aggregated_zero_shot_review_scores.csv", index=False)

if __name__ == "__main__":
    main()
