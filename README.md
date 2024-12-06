Team Members :

A.Saras Chandrika - 21BDS003

Chandana.R - 21BDS010

G.Thanmai - 21BDS033

Repository Structure :

The original dataset Twitter_Data.csv is uploaded in the repository. The preprocessed dataset is stored as Twitter_Dataset_After_Preprocessing.csv and uploaded in releases. All code, including preprocessing, model training, and the Flask web application, is available in Colab notebook and vs code uploaded to the repository.

Overview :

This project conducts sentiment analysis on Twitter data collected during the Indian General Elections 2019. The dataset comprises approximately 1,63,000 tweets discussing various political leaders, including Modi, and their impact on the elections. The project aims to analyze the sentiment expressed in these tweets and compare the performance of different machine learning (ML) and deep learning (DL) models for sentiment analysis.

Dataset Collection :

The initial dataset containing tweets on Modi and other leaders during the Indian General Elections 2019 was collected from Kaggle. The dataset consists of approximately 1,63,000 tweets.

Data Visualization :

Data visualization techniques such as

1.Bar Graph

2.Pie Chart

3.Word Clouds(for each sentiment)

were employed to gain insights into the distribution of sentiment across various tweets.

Data Preprocessing :

Preprocessing steps including converting text to lowercase, removing special character,symbols,stop words, punctuation, tokenization, and lemmatization were done to clean the text data.The preprocessed dataset is stored as Twitter_Dataset_After_Preprocessing.csv. It is uploaded in releases.

Models Used :

Four models were selected for sentiment analysis:

1.Naive Bayes

2.Naive Bayes with bag of words vectorization

3.LSTM (Long Short-Term Memory)

4.BERT (Bidirectional Encoder Representations from Transformers)

Each model was trained on the preprocessed Twitter dataset, and their accuracies were evaluated along with classification reports. BERT achieved the highest accuracy among all models.

Comparative Analysis :

A comparative analysis was conducted to compare the accuracies and performance of the different models used. Bert model outperformed in this analysis compared to other models used.

Web Interface :

A web interface was developed using Flask, allowing users to input a tweet and predict its sentiment using the trained BERT model.

Technologies used are

Flask: Python web framework for building the interface.
BERT: Pre-trained transformer model for sentiment analysis.
API calls: To interact with the BERT model for predictions.
