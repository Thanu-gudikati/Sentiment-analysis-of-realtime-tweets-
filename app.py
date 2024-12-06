from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the model
model_path = 'c:/Users/akkis/Downloads/model_weights (1).pth'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function
def predict_sentiment(tweet):
  max_len = 128
  with torch.no_grad():
    encoded_dict = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    # Fix: convert logits to actual predicted class (0, 1, or 2)
    prediction = torch.argmax(logits, dim=1).cpu().item()  # .item() to get scalar
  return prediction

# Route to serve the main webpage
@app.route('/')
def index():
  return render_template('index.html')

# Route to handle sentiment analysis requests
@app.route('/predict', methods=['POST'])
def get_prediction():
  data = request.get_json()
  tweet = data['tweet']
  prediction = predict_sentiment(tweet)
  # Map prediction (0, 1, or 2) to labels (negative, neutral, positive)
  if prediction == 0:
      sentiment = "Negative"
  elif prediction == 1:
      sentiment = "Neutral"
  else:
      sentiment = "Positive"
  return jsonify({'prediction': sentiment})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
