#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')


# In[136]:


# Might need this 
nltk.download('punkt')


# In[137]:


# Commented this out since I have b4 installed
# get_ipython().system(" pip3 install bs4 # in case you don't have it installed")


# # Data Generation (from HW1)

# In[138]:


df = pd.read_csv("data.csv", sep=',', on_bad_lines='skip')


# In[139]:


# Extract Reviews and Ratings fields
df = df.loc[:, ['review_body', 'star_rating']]

print(df.head())


# In[140]:


# Converting into binary classification problem
df['label'] = df['star_rating'].apply(lambda x: 1 if x in [1,2,3] else 2)
df['review'] = df['review_body']

# Selecting 50,000 random reviews from each rating class
# Randomizing to avoid biases 
df_class_1 = df[df['label'] == 1].sample(n=50000, random_state=55)
df_class_2 = df[df['label'] == 2].sample(n=50000, random_state=55)

# Creating a new df concatenating both classes 
balanced_df = pd.concat([df_class_1, df_class_2])

print(balanced_df.head(10))


# # Preprocessing (from HW1)

# In[141]:


average_len_before = balanced_df['review'].str.len().mean()

# 1)converting all reviews into lowercase. 
# Ensures consistency: "Hello" and "hello" are now the same. 
balanced_df['review'] = balanced_df['review'].str.lower()

# 2)removing the HTML and URLs from the reviews
# HTML/URLs don't provide valuable information for sentiment analysis, so we remove them. 
balanced_df['review'] = balanced_df['review'].str.replace(r'<.*?>', '', regex=True)
balanced_df['review'] = balanced_df['review'].str.replace(r'http\S+', '', regex=True)

# 5)performing contractions on the reviews
    # Need to process this before before removing non-alphanum chars and extra spaces 
# This task provides uniformity and simplifies tokenization
def contractions_helper(ss):

    # To avoid attributionError
    if type(ss) != str: 
        return
    contractions_dict = {
        "won't": "will not",
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "hasn't": "has not",
        "coudn't": "could not",
        "they're": "they are",
        "you're": "you are",
        "we'll": "we will",
        "it's": "it is",
        "i'll": "i will",
        "he's": "he is",
        "she's": "she is"
    }
    # loop through the string and replace all contractions
    for cont, exp in contractions_dict.items():
            if cont in ss:
                 ss = ss.replace(cont, exp)
    return ss

balanced_df['review'] = balanced_df['review'].apply(contractions_helper)

# 3)removing non-alphabetical characters
# Removing non-alphanum characters since they could be noise in sentiment analysis
balanced_df['review'] = balanced_df['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# 4)removing extra spaces
balanced_df['review'] = balanced_df['review'].str.replace(r'\s+', ' ', regex=True)


average_len_after = balanced_df['review'].str.len().mean()

# average length decreased after cleaning due to the removal of unwanted characters, spaces, and expansion of contractions
print(f'Average length before data cleaning:{average_len_before:.4f}, Average length after data cleaning:{average_len_after:.4f}')


# # Feature Extraction

# In[142]:


# imports 
from gensim.models import Word2Vec
import gensim.downloader as api


# In[143]:


# Filtering out rows where 'review' is not a string
balanced_df = balanced_df[balanced_df['review'].apply(lambda x: isinstance(x, str))]

# Word2Vec model trained with amazon reviews
sentences = balanced_df['review'].str.split().tolist() # get review sentences
# train word2vec model
my_model = Word2Vec(sentences, vector_size=300, window=13, min_count=9, workers=4)
my_model.train(sentences, total_examples=len(sentences), epochs=10)

# Pre-trained word2vec model
wv_model = api.load('word2vec-google-news-300')


# ### 2. Word Embedding: part (a) and (b) combined

# In[ ]:


print("Example 1")
# Pretrained: check for semantic similarities
print("Word2Vec Model: Similarity for words 'excellent' and 'outstanding':", wv_model.similarity('excellent', 'outstanding'))

# My model: check semantic similarities
if 'excellent' in my_model.wv and 'outstanding' in my_model.wv:
    print("My Model: Similarity between 'excellent' and 'outstanding':", my_model.wv.similarity('excellent', 'outstanding'))

print("Example 2")
# Pretrained: check for analogy: King - Man + Woman
result = wv_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print("Word2Vec Model: King - Man + Woman = ", result[0][0])

# My model: check analogy: King - Man + Woman
if all(word in my_model.wv for word in ['woman', 'king', 'man']):
    result = my_model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print("My Model: King - Man + Woman = ", result[0][0])

print("Example 3 ('delicious' and 'tasty' not in my model)")
# Pretrained model:
print("Word2Vec Model: Similarity for words 'delicious' and 'tasty':", wv_model.similarity('delicious', 'tasty'))

# My model:
if 'delicious' in my_model.wv.key_to_index and 'tasty' in my_model.wv.key_to_index:
    print("My Model: Similarity between 'delicious' and 'tasty':", my_model.wv.similarity('delicious', 'tasty'))


# # Simple Models: Perceptron and SVM

# In[ ]:


# imports
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

# TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# TFIDF Features

# In[ ]:


from sklearn.model_selection import train_test_split

# TFIDF Features
tfidf_vector = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vector.fit_transform(balanced_df['review'])

# Splitting data into train and test sets
x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    tfidf_features, 
    balanced_df['label'],
    test_size= 0.2,
    random_state=55
)


# Word2Vec Features

# In[ ]:


# Calculating the average Word2Vec for each review
def average_word2vec(review, model, dimension):
    avg_w2v = np.zeros((dimension,))
    num_words = 0
    for word in review:
        if word in model:  
            avg_w2v += model[word]
            num_words += 1
    if num_words > 0:
        avg_w2v /= num_words
    return avg_w2v

# Splitting reviews into sentences for Word2Vec
sentences = balanced_df['review'].str.split().tolist()

# Convert reviews into feature vectors using average Word2Vec helper function
word2vec_features = np.array([average_word2vec(review, wv_model, 300) for review in sentences])

# Splitting into train and test sets
x_train_word2vec, x_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
    word2vec_features, 
    balanced_df['label'],
    test_size= 0.2,
    random_state=55
)


# Perceptron Model w/ TFIDF features and word2vec features

# In[ ]:


# Perceptron on tfidf features
tfidf_perceptron = Perceptron(max_iter=1000, random_state=55)

# Fit training set into Perceptron Model 
tfidf_perceptron.fit(x_train_tfidf, y_train_tfidf)

# Make predictions w/ testing set
tfidf_prediction = tfidf_perceptron.predict(x_test_tfidf)

# Report Precision, Recall, and f1-score
tfidf_test_precision = precision_score(y_test_tfidf, tfidf_prediction, average='binary')
tfidf_test_recall = recall_score(y_test_tfidf, tfidf_prediction, average='binary')
tfidf_test_f1 = f1_score(y_test_tfidf, tfidf_prediction, average='binary')

# Print
print(f"TF-IDF Perceptron Model~ Precision:{tfidf_test_precision:.4f}, Recall:{tfidf_test_recall:.4f}, F1-Score:{tfidf_test_f1:.4f}")


# Perceptron on word2vec features
word2vec_perceptron = Perceptron(max_iter=1000, random_state=55)

# Fit training set into Perceptron Model
word2vec_perceptron.fit(x_train_word2vec, y_train_word2vec)

# Make predictions w/ testing set
word2vec_prediction = word2vec_perceptron.predict(x_test_word2vec)

# Report Precision, Recall, and f1-score
word2vec_test_precision = precision_score(y_test_word2vec, word2vec_prediction, average='binary')
word2vec_test_recall = recall_score(y_test_word2vec, word2vec_prediction, average='binary')
word2vec_test_f1 = f1_score(y_test_word2vec, word2vec_prediction, average='binary')

# Print
print(f"Word2Vec Perceptron Model~ Precision:{word2vec_test_precision:.4f}, Recall:{word2vec_test_recall:.4f}, F1-Score:{word2vec_test_f1:.4f}")


# SVM Model w/ TFIDF features and word2vec features

# In[ ]:


# SVM on tfidf features
tfidf_svm = LinearSVC(dual=False, max_iter=1000, random_state=55)

# Fit training set into SVM classifier 
tfidf_svm.fit(x_train_tfidf, y_train_tfidf)

# Predict on testing set
tfidf_svm_prediction = tfidf_svm.predict(x_test_tfidf)

# Report Precision, Recall, and f1-score
tfidf_svm_precision = precision_score(y_test_tfidf, tfidf_svm_prediction, average='binary')
tfidf_svm_recall = recall_score(y_test_tfidf, tfidf_svm_prediction, average='binary')
tfidf_svm_f1 = f1_score(y_test_tfidf, tfidf_svm_prediction, average='binary')

# Print
print(f"TF-IDF SVM Model~ Precision:{tfidf_svm_precision:.4f} Recall:{tfidf_svm_recall:.4f} F1-Score:{tfidf_svm_f1:.4f}")

# SVM on word2vec features
word2vec_svm = LinearSVC(dual=False, max_iter=1000, random_state=55)

# Fit training set into SVM classifier
word2vec_svm.fit(x_train_word2vec, y_train_word2vec)

# Predict on testing set
word2vec_svm_prediction = word2vec_svm.predict(x_test_word2vec)

# Report Precision, Recall, and f1-score
word2vec_svm_precision = precision_score(y_test_word2vec, word2vec_svm_prediction, average='binary')
word2vec_svm_recall = recall_score(y_test_word2vec, word2vec_svm_prediction, average='binary')
word2vec_svm_f1 = f1_score(y_test_word2vec, word2vec_svm_prediction, average='binary')

# Print
print(f"Word2Vec SVM Model~ Precision:{word2vec_svm_precision:.4f} Recall:{word2vec_svm_recall:.4f} F1-Score:{word2vec_svm_f1:.4f}")


# My explanation:
# - Deep learning models perform better with embeddings like Word2Vec than with sparse representations like TF-IDF
# - Linear models like Perceptron and SVM might sometimes favor the discriminative power of TF-IDF

# # Feedforward Neural Networks

# In[ ]:


# import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score


# In[ ]:


# Convert to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train_word2vec)
y_train_tensor = torch.LongTensor(y_train_word2vec.values) 
x_test_tensor = torch.FloatTensor(x_test_word2vec)
y_test_tensor = torch.LongTensor(y_test_word2vec.values)

# Check before moving forward
print("Unique values in y_train_tensor:", torch.unique(y_train_tensor))
print("Unique values in y_test_tensor:", torch.unique(y_test_tensor))

print("Shape of X_train_tensor:", x_train_tensor.shape)
# print("Sample values from X_train_tensor:", X_train_tensor[:5])
print("Shape of X_test_tensor:", x_test_tensor.shape)
# print("Sample values from X_test_tensor:", X_test_tensor[:5])

# Re-map the values from [1, 2] to [0, 1] for proper training
y_train_tensor -= 1
y_test_tensor -= 1

print("Updated unique values in y_train_tensor:", torch.unique(y_train_tensor))
print("Updated unique values in y_test_tensor:", torch.unique(y_test_tensor))


# In[ ]:


# Create train loader with batch size 64
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
        )
    
    def forward(self, x):
        return self.layers(x)

# Hyperparameters: two hidden layers, each with 50 and 5 nodes, respectively
input_size = 300  
hidden1_size = 50  
hidden2_size = 5   
output_size = 2    

# Initialize model, loss, and optimizer
model = MLP(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    _, predicted = test_outputs.max(1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_word2vec)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Concatenate the first 10 Word2Vec vectors for each review as the input feature

# In[ ]:


def concat_first_10_word2vec(review, model, dimension):
    feature_vec = []
    for i in range(10):
        if i < len(review) and review[i] in model:
            feature_vec.extend(model[review[i]])
        else:
            feature_vec.extend(np.zeros((dimension,)))
    return feature_vec

sentences = balanced_df['review'].str.split().tolist()

# Convert reviews into feature vectors using concatenated Word2Vec
word2vec_10words_features = np.array([concat_first_10_word2vec(review, wv_model, 300) for review in sentences])


# In[ ]:


# Split into train and test sets using word2vec_10words_features
x_train_10words, x_test_10words, y_train_10words, y_test_10words = train_test_split(
    word2vec_10words_features, 
    balanced_df['label'],
    test_size= 0.2,
    random_state=55
)

# Convert to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train_10words)
y_train_tensor = torch.LongTensor(y_train_10words.values) 
x_test_tensor = torch.FloatTensor(x_test_10words)
y_test_tensor = torch.LongTensor(y_test_10words.values)

# re-map the values from [1, 2] to [0, 1] for proper training
y_train_tensor -= 1
y_test_tensor -= 1


# In[ ]:


# Create train loader with batch size 64
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Hyperparameters: two hidden layers, each with 50 and 5 nodes, respectively
class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
        )
    
    def forward(self, x):
        return self.layers(x)

# Hyperparameters: two hidden layers, each with 50 and 5 nodes, respectively
input_size = 3000  
hidden1_size = 50  
hidden2_size = 5   
output_size = 2    

# Initialize model, loss, and optimizer
model = MLP(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    _, predicted = test_outputs.max(1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_10words)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# My explanation:
# - The test accuracy drops after concatenating the first 10 words of each review.
# - This is because when you concatenate the first 10 Word2Vec vectors, you might be losing some information that is crucial for the classification task
# - Word2Vec vectors capture semantic meaning. By taking only the first 10 words, the model might be missing out on important context that comes from the rest of the review

# # Recurrent Neural Network

# In[ ]:


# imports
import torch.nn.functional as F


# A little bit of preprocessing

# In[ ]:


# NOTE: using x_train_word2vec, x_test_word2vec, y_train_word2vec, y_test_word2vec

# Maintain the sequence of vectors since RNN processes inputs one step at a time and maintains a hidden state across those steps
def get_word2vec_sequence(review, model, dimension):
    w2v_sequence = []
    for word in review:
        if word in model:
            w2v_sequence.append(model[word])
        else:
            w2v_sequence.append(np.zeros((dimension,)))  # Using a zero vector for unknown words
    return w2v_sequence

# Convert reviews into sequences of Word2Vec vectors
sentences = balanced_df['review'].str.split().tolist()
word2vec_sequences = [get_word2vec_sequence(review, wv_model, 300) for review in sentences]

# To feed your data into our RNN, limit the maximum review length to 10 
# by truncating longer reviews and padding shorter reviews with a null value (0)
def pad_or_truncate_sequence(sequence, max_length):
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        while len(sequence) < max_length:
            sequence.append(np.zeros((300,)))  # Padding with zero vectors
    return sequence

padded_word2vec_sequences = np.array([pad_or_truncate_sequence(seq, 10) for seq in word2vec_sequences])

# Split into train and test sets using padded_word2vec_sequences
x_train_word2vec, x_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
    padded_word2vec_sequences, 
    balanced_df['label'],
    test_size= 0.2,
    random_state=55
)


# In[ ]:


# check before moving forward 
print(x_train_word2vec.shape, x_test_word2vec.shape)


# In[ ]:


# Hyperparameters: one hidden layer with 10
input_size = 300  
hidden_size = 10
output_size = 2 


# ### Simple RNN

# In[ ]:


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # Initialize hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Only take the last time step's output for classification
        return out
  
model = SimpleRNN(input_size, hidden_size, output_size)


# In[ ]:


# Convert to PyTorch tensors
x_train_rnn_tensor = torch.FloatTensor(x_train_word2vec)
x_test_rnn_tensor = torch.FloatTensor(x_test_word2vec)
y_train_rnn_tensor = torch.LongTensor(y_train_word2vec.values)
y_test_rnn_tensor = torch.LongTensor(y_test_word2vec.values)

# print("Unique values in y_train_tensor:", torch.unique(y_train_rnn_tensor))
# print("Unique values in y_test_tensor:", torch.unique(y_test_rnn_tensor))

# re-map the values from [1, 2] to [0, 1] for proper training
y_train_rnn_tensor -= 1
y_test_rnn_tensor -= 1

# print("Updated unique values in y_train_tensor:", torch.unique(y_train_rnn_tensor))
# print("Updated unique values in y_test_tensor:", torch.unique(y_test_rnn_tensor))


# In[ ]:


# Create train loader with batch size 64
train_data_rnn = TensorDataset(x_train_rnn_tensor, y_train_rnn_tensor)
train_loader_rnn = DataLoader(train_data_rnn, batch_size=64, shuffle=True)

# Initialize loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    for _, (data, target) in enumerate(train_loader_rnn):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

with torch.no_grad():
    test_outputs = model(x_test_rnn_tensor)
    _, predicted = test_outputs.max(1)
    accuracy = (predicted == y_test_rnn_tensor).sum().item() / len(y_test_word2vec)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ### GRU

# In[ ]:


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # NOTE: Only taking the last time step's output for classification
        return out

# Initialize model, loss, and optimizer
model = SimpleGRU(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# Training
epochs = 10
for epoch in range(epochs):
    for _, (data, target) in enumerate(train_loader_rnn):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing
with torch.no_grad():
    test_outputs = model(x_test_rnn_tensor)
    _, predicted = test_outputs.max(1)
    accuracy = (predicted == y_test_rnn_tensor).sum().item() / len(y_test_word2vec)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ### LSTM

# In[ ]:


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        
        # Using LSTM instead of GRU
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize both hidden state and cell state for LSTM
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only take the last time step's output for classification
        return out

# Model, Criterion, Optimizer Initialization
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 10
for epoch in range(epochs):
    for _, (data, target) in enumerate(train_loader_rnn):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing
with torch.no_grad():
    test_outputs = model(x_test_rnn_tensor)
    _, predicted = test_outputs.max(1)
    accuracy = (predicted == y_test_rnn_tensor).sum().item() / len(y_test_word2vec)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# My explanation: 
# - Both GRU and LSTM have significantly higher accuracies compared to the Simple RNN. 
# - This underscores the effectiveness of gating mechanisms in capturing long-term dependencies and mitigating issues like vanishing and exploding gradients that simple RNNs suffer from.
