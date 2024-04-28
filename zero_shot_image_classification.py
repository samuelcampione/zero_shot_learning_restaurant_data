import os
import numpy as np
import pandas as pd
from PIL import Image
from transformers import pipeline


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MODEL_CHECKPOINT = "openai/clip-vit-large-patch14"
PHOTOS_DIRECTORY = '/Users/scampione/MSDS/Spring_24_2/Entrepreneurship/philly_business_photos'


def classify_image(image_path, detector, candidate_labels):
    """
    Classify an image into provided categories and return the scores.
    
    Args:
        image_path (str): Path to the image file.
        detector (pipeline): Pre-initialized Transformers pipeline for zero-shot image classification.
        candidate_labels (list): List of categories as strings.
        
    Returns:
        dict: Scores for each category if successful, empty dictionary otherwise.
    """
    try:
        with Image.open(image_path) as image:
            classifications = detector(images=image, candidate_labels=candidate_labels, num_workers=8)
            return {result['label']: result['score'] for result in classifications}
    except Exception as e:
        print(f"Failed to process image {image_path}: {str(e)}")
        return {}


def classify_business_images(directory, detector, candidate_labels):
    """
    Process all valid image files in a business directory and average their classification scores.
    
    Args:
        directory (str): Directory path containing images.
        detector (pipeline): Pre-initialized Transformers pipeline.
        candidate_labels (list): List of target categories for classification.
    
    Returns:
        dict: Average scores for each category.
    """
    scores = {label: [] for label in candidate_labels}
    for filename in os.listdir(directory):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(directory, filename)
            image_scores = classify_image(image_path, detector, candidate_labels)
            for label, score in image_scores.items():
                scores[label].append(score)
    return {label: np.mean(score_list) if score_list else float('nan') for label, score_list in scores.items()}


def main():
    detector = pipeline(model=MODEL_CHECKPOINT, task="zero-shot-image-classification", multi_label=True)
    business_ids = [d for d in os.listdir(PHOTOS_DIRECTORY) if d != '.DS_Store']
    business_directories = [os.path.join(PHOTOS_DIRECTORY, id) for id in business_ids]
    
    candidate_labels = [
        "Coworking Cafe——Coffee shop with people working on laptops, Wi-Fi available", 
        "Brunch——Outdoor brunch with breakfast and mimosas", 
        "Romantic Date Night——Romantic restaurant with candlelit tables", 
        "Upscale Special Occasion——Elegant restaurant with white tablecloths and fine dining", 
        "Rooftop——Rooftop bar or restaurant with city views", 
        "Pub——Lively bar with draft beer and sports TV", 
        "None of the above"
    ]

    results = {'bid': []}
    for label in candidate_labels:
        results[label] = []

    for directory in business_directories:
        results['bid'].append(os.path.basename(directory))
        average_scores = classify_business_images(directory, detector, candidate_labels)
        for label, average_score in average_scores.items():
            results[label].append(average_score)

    results_df = pd.DataFrame(results)
    results_df.columns = ['bid'] + [f'{label.lower().replace(" ", "_").replace("——", "_")}score' for label in candidate_labels]
    results_df.to_csv("zero_shot_scores.csv", index=False)

if __name__ == "__main__":
    main()
