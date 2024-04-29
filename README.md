# Pan: *What's your vibe?*

We bring to you a Vibe-based, Ambience-based restaurant recommender. Tired of navigating through biased ads and generic recommendations? We tackle this issue by harnessing the power of computer vision and natural language processing to pinpoint the unique vibe of each restaurant. Utilizing [OpenAI's CLIP model](https://openai.com/research/clip) and [Facebook's BART-large-MNLI model](https://huggingface.co/facebook/bart-large-mnli), we've processed 20,000 images and 500,000 reviews to deliver authentic, user-centered dining suggestions.


Explore our methodology and see how it works in: 
- [Zero-Shot Image Classification](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_image_classification.py)
- [Zero-Shot Review Classification](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_review_classification.py)
- [PySpark Database Filtering](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/get_restaurant_results.ipynb)
- [requirements.txt](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/requirements.txt)

# So, what are our Vibes?
Informed by customer interviews, we created specific Vibe categories to meet the consumer's needs for the perfect dining experience.

- **Coworking Cafe**: Fresh coffee for productive working
- **Brunch**: Relaxed and trendy brunch spots
- **Green**: Fresh and vegetarian options
- **Family-Friendly**: For the whole family
- **Local Delicacies**: Local spots worth trying
- **Date Night**: Sweet romantic night out vibes
- **Special Occassion**: The Nines! Fancy, luxe and delicious
- **Rooftop**: Trendy rooftop scenery and craft drinks

# What is Zero-Shot Image Classification?

Zero-shot image classification with CLIP involves the model understanding and classifying images it hasn't been explicitly trained on by comparing the image's features to text descriptions. CLIP,  Contrastive Language-Image Pretraining, encodes both the image and a range of potential textual labels into a common embedding space. It then predicts the most relevant labels for the image based on the proximity of the image and text embeddings, thus enabling the model to classify images without having been trained on those specific labels or classes.

<div align="center">
    <img src="https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_diagram.png" alt="Screenshot" width="600">
</div>


# What about Zero-Shot Text Classification?
We did zero-shot text classification with BART-large-MNLI enables categorization of texts into labels it hasn't been explicitly trained on. We use a pretrained model, BART, that was fine tuned on the MNLI dataset which includes sentence pairs labeled for entailment or contradiction and uses these relationships to evaluate text. The model compares embeddings of the text and potential labels to determine the most appropriate categories, allowing it to classify texts into new labels without specific prior training.


# Data
We leverage the [Yelp Open Dataset](https://www.yelp.com/dataset), which includes:
- Structured business data
- Unstructured textual business reviews
- Unstructured business image data

# Tools Used
- OpenAI's CLIP model: For zero-shot image classification.
- Facebook's BART Model fine-tuned on MNLI dataset: For zero-shot text classification.
- PySpark: For robust database creation, data storage, and SQL querying.
- [requirements.txt](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/requirements.txt) 
