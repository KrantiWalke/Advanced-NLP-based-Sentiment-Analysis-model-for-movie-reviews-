# Sentiment Analysis model for movie reviews using Pytorch,RNN,LSTM

## Table of Contents
- [Deep Learning Techniques and Key Elements](#deep-learning-techniques-and-key-elements)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Data Preparation and Processing](#data-preparation-and-processing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results and Analysis](#results-and-analysis)
- [User Input Predictions](#user-input-predictions)
- [Conclusion](#conclusion)

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

## Data Preparation and Processing
- **Dataset**: Utilizing the IMDb dataset available through the HuggingFace `datasets` library.
- **Tokenization**: Applying basic English tokenizer from TorchText.
- **Numericalization and Padding**: Converting tokens to numerical IDs and padding sequences to a uniform length.
- **Data Splitting**: Creating training, validation, and test sets.
- **Vocabulary Building**: Constructing a vocabulary from the training set tokens, with a minimum frequency threshold for inclusion.

## Model Architecture
- **LSTM (Long Short-Term Memory)**: Chosen for its ability to mitigate vanishing gradients and capture long-range dependencies.
- **Bidirectionality**: Enhancing context capture by processing data in both forward and backward directions.
- **Dropout**: Applied for regularization to reduce overfitting chances.

## Training and Evaluation
- **Optimizer**: Adam optimizer for adaptive learning rate adjustments.
- **Loss Function**: CrossEntropyLoss for classification tasks.
- **Batch Processing**: Custom collate function to handle variable-length sequences within batches.

## Results and Analysis
- **Training Loop**: Monitoring loss and accuracy over epochs, adjusting parameters as needed to improve performance.
- **Evaluation**: Assessing the model on a held-out test set to gauge its generalization capabilities.

## User Input Predictions
- **Functionality**: The model can predict the sentiment of custom text inputs, showcasing its practical application.

## Conclusion
This project demonstrates the power of combining PyTorch's dynamic computation capabilities with pre-trained embeddings and advanced RNN architectures for sentiment analysis. The resulting model not only achieves high accuracy but also provides a foundation for further exploration and enhancement in natural language processing tasks.
