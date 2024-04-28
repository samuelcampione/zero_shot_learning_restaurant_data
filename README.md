# Pan: *What's your vibe?*

We bring to you a Vibe-based, Ambience-based restaurant recommender. Tired of navigating through biased ads and generic recommendations? We tackle this issue by harnessing the power of computer vision and natural language processing to pinpoint the unique vibe of each restaurant. Utilizing [OpenAI's CLIP model](https://openai.com/research/clip), we've processed 20,000 images (1GB) to deliver authentic, user-centered dining suggestions.


Explore our methodology and see how it works in our: 
- [Zero-Shot Image Classification](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_image_classification.py)
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

# What even is Zero-Shot Image Classification?

Zero-shot image classification with CLIP involves the model understanding and classifying images it hasn't been explicitly trained on by comparing the image's features to text descriptions. CLIP,  Contrastive Language-Image Pretraining, encodes both the image and a range of potential textual labels into a common embedding space. It then predicts the most relevant labels for the image based on the proximity of the image and text embeddings, thus enabling the model to classify images without having been trained on those specific labels or classes.

<div align="center">
    <img src="https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_diagram.png" alt="Screenshot" width="600">
</div>



# Data
We leverage the [Yelp Open Dataset](https://www.yelp.com/dataset), which includes:
- Structured business data
- Unstructured textual business reviews
- Unstructured business image data

# Tools Used
- OpenAI's CLIP model: For zero-shot image classification.
- PySpark: For robust database creation, data storage, and SQL querying.
- [requirements.txt](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/requirements.txt) 
