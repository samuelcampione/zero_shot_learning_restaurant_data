# Pan: *What's your vibe?*

We bring to you a Vibe-based, Ambience-based restaurant recommender. Sick of sponsored ads, we decided to tackle the problem by taking advantage of computer vision and natural language proessing to extract the specific Vibe of a restaurant. Leveraging zero-shot learning with [OpenAI's CLIP model](https://openai.com/research/clip), we processed around 20,000 images (1 GB).

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

See how it works in this notebook: [zero_shot_classification.ipynb](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_classification.ipynb)!

# What even is Zero-Shot Image Classification?

![](https://github.com/samuelcampione/zero_shot_learning_restaurant_data/blob/main/zero_shot_diagram.png "Screenshot")


# Data 

We used data from the [Yelp Open Dataset](https://www.yelp.com/dataset) including structured business data, unstructured textual business review data, and unstructured business image data. 
