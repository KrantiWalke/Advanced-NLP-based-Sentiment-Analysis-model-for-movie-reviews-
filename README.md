# Sentiment Analysis model for movie reviews using ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white) ![RNN](https://img.shields.io/badge/RNN-%23FF6F00.svg?style=for-the-badge&logo=deeplearning.ai&logoColor=white) ![LSTM](https://img.shields.io/badge/LSTM-%23000000.svg?style=for-the-badge&logo=deeplearning.ai&logoColor=white)


## Table of Contents
- [Overview](#overview)
- [Deep Learning Techniques and Key Elements](#deep-learning-techniques-and-key-elements)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Data Preparation and Processing](#data-preparation-and-processing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results and Analysis](#results-and-analysis)
- [User Input Predictions](#user-input-predictions)
- [Conclusion](#conclusion)

## Overview
The project appears to be focused on sentiment analysis using deep learning techniques, specifically utilizing Long Short-Term Memory (LSTM) networks. Sentiment analysis is a popular natural language processing task that involves determining the sentiment expressed in a piece of text, such as whether a movie review is positive or negative.

## Deep Learning Techniques and Key Elements:
- **Tokenization and Numericalization**: Utilizing TorchText for efficient text processing.
- **Packed Padded Sequences**: Optimizing RNN processing with dynamic computation.
- **Pre-trained Word Embeddings**: Initializing our model with GloVe embeddings.
- **Advanced RNN Architectures**: Implementing an LSTM model to better capture dependencies.
- **Bidirectionality and Multi-layering**: Expanding the model's capacity to understand context from both directions and at multiple levels of abstraction.
- **Regularization**: Implementing dropout to prevent overfitting.

These Techniques aim to push the model's accuracy even further, targeting a goal of around 85%.

## Libraries and Dependencies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchText](https://img.shields.io/badge/TorchText-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%23D9042B.svg?style=for-the-badge&logo=matplotlib&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-%232C8EBB.svg?style=for-the-badge&logo=tqdm&logoColor=white)
![Datasets](https://img.shields.io/badge/HuggingFace_Datasets-%23F7931E.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![GloVe](https://img.shields.io/badge/GloVe-%23E34F26.svg?style=for-the-badge&logo=glove&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-008EDD?style=for-the-badge&logo=nvidia&logoColor=white)

## Data Preparation and Processing
The project uses the IMDb dataset, which consists of movie reviews labeled with sentiment (positive or negative). The data is preprocessed, tokenized, and converted into numerical format suitable for training deep learning models.
- **Dataset**: Utilizing the IMDb dataset available through the HuggingFace `datasets` library.
- **Tokenization**: Applying basic English tokenizer from TorchText.
- **Numericalization and Padding**: Converting tokens to numerical IDs and padding sequences to a uniform length.
- **Data Splitting**: Creating training, validation, and test sets.
- **Vocabulary Building**: Constructing a vocabulary from the training set tokens, with a minimum frequency threshold for inclusion.

## Model Architecture
An LSTM neural network is chosen for sentiment analysis. LSTMs are well-suited for sequential data processing tasks due to their ability to capture long-term dependencies. The model architecture includes an embedding layer, LSTM layers, and a fully connected layer for classification.
- **LSTM (Long Short-Term Memory)**: Chosen for its ability to mitigate vanishing gradients and capture long-range dependencies.
- **Bidirectionality**: Enhancing context capture by processing data in both forward and backward directions.
- **Dropout**: Applied for regularization to reduce overfitting chances.

## Training and Evaluation
The model is trained using the training dataset and evaluated using the validation dataset. The training loop includes optimizing the model parameters with an Adam optimizer and calculating the loss and accuracy metrics. Additionally, the model is evaluated on a separate test dataset to assess its generalization performance.
- **Optimizer**: Adam optimizer for adaptive learning rate adjustments.
- **Loss Function**: CrossEntropyLoss for classification tasks.
- **Batch Processing**: Custom collate function to handle variable-length sequences within batches.

## Results and Analysis
The visualization of training and validation metrics such as loss and accuracy over epochs. These visualizations provide insights into the training process and model performance.
- **Training Loop**: Monitoring loss and accuracy over epochs, adjusting parameters as needed to improve performance.
- **Evaluation**: Assessing the model on a held-out test set to gauge its generalization capabilities.

- test_loss: 0.327, test_acc: 0.862

![image](https://github.com/KrantiWalke/Advanced-NLP-based-Sentiment-Analysis-model-for-movie-reviews-/assets/72568005/30dcf609-4265-4288-b1f4-018485cc9193)
![image](https://github.com/KrantiWalke/Advanced-NLP-based-Sentiment-Analysis-model-for-movie-reviews-/assets/72568005/e2ac2dd1-d951-4a6e-bb58-aea45ffe9b2f)

## User Input Predictions
After training, the model can be deployed for real-world applications such as sentiment analysis of user-generated content. The project demonstrates how to use the trained model for predicting sentiment on new text inputs.

The final block of code demonstrates the use of the `predict_sentiment` function with four different text inputs to evaluate the performance of the trained LSTM model in sentiment analysis. Each text input represents a different sentiment or a nuanced expression that combines negative and positive sentiments. The function returns the predicted sentiment class and the probability associated with that class for each input text.

1. **"This film is terrible!"**: This is a straightforward negative sentiment. The model's prediction for this text will show how well it can identify negative sentiments.
   
2. **"This film is great!"**: This text expresses a clear positive sentiment. The model's prediction here will indicate its ability to recognize positive sentiments.

3. **"This film is not terrible, it's great!"**: This sentence is more complex because it contains negation ("not terrible") followed by a positive statement ("it's great"). This tests the model's ability to understand the context and the overall sentiment of a sentence when negation is involved.

4. **"This film is not great, it's terrible!"**: Similar to the third sentence, this one also tests the model's understanding of negation and context. However, in this case, the sentiment is reversed from positive to negative.

The outcomes of these predictions can provide insights into the model's strengths and weaknesses:

- If the model correctly predicts the sentiment for the straightforward positive and negative statements (sentences 1 and 2), it indicates that the model has a basic understanding of sentiment analysis.

- If the model also accurately predicts the sentiment for the more complex sentences involving negation (sentences 3 and 4), it suggests that the model is capable of understanding context and negation to a certain extent, which is a more challenging task in natural language processing.

Overall, these predictions can help in evaluating the model's performance and understanding how well it can handle different types of sentiment expressions, including those with negations and mixed sentiments.

## Conclusion
This project demonstrates the power of combining PyTorch's dynamic computation capabilities with pre-trained embeddings and advanced RNN architectures for sentiment analysis. The resulting model not only achieves high accuracy but also provides a foundation for further exploration and enhancement in natural language processing tasks.
