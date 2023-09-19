# Importing all the required libraries
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
nltk.download('stopwords')
nltk.download('wordnet')

# Setting repo path and initializing lemmatizer
repo_path = "datasets/readmes/"
lemmatizer = WordNetLemmatizer()

# Function to preprocess the text
def preprocess_text(text):
    # Regex to remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Regex to remove HTML Tags
    text = re.sub('<.*?>', '', text)
    # Regex to temove special characters and numbers
    text = re.sub(r'\W+|\d+', ' ', text)
    # Converting the text to lowercase
    text = text.lower()
    # Removing the word kubernetes since all projects are related to kubernetes
    text = re.sub("kubernetes", "", text)
    tokens = text.split()
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocessing text and storing vocabulary of all repos
corpus = []
for filename in tqdm.tqdm(os.listdir(repo_path)):
    with open(os.path.join(repo_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        processed_content = preprocess_text(content)
        corpus.append(processed_content)

# Vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, min_df = 5, max_features = 20000)
X = tfidf_vectorizer.fit_transform(corpus)

# Number of topics to be considered by LDA
n_topics = 10

# Initializing and training the LDA algorithm
lda = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=0.8, topic_word_prior=0.8, max_iter = 5, learning_method = 'online', random_state=42, learning_decay=0.7)
lda.fit(X)

# Extracting feature names from TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Extracting, storing, and printing the topic index and the top 10 words associated with that topic
topics = []
for topic_idx, topic in enumerate(lda.components_):
    txt = "Topic #%d: " % topic_idx
    txt += " ".join([feature_names[i] for i in topic.argsort()[:-11:-1]])
    topics.append((topic_idx, txt))
    print(txt)

# Finding the topic for each repo using the LDA model
assigned_topics_list = lda.transform(X)
assigned_topics = assigned_topics_list.argmax(axis = 1)

# Storing the Repo name and the associated Topic ID in a text file
with open("topic_assignments.txt", "w") as f:
    for i in range(len(assigned_topics)):
        repo_name = os.listdir(repo_path)[i]
        f.write(f"{repo_name}: Topic {assigned_topics[i]}\n")

# Finding the count of the number of repos assigned to each Topic ID
count_ = {}
for i in assigned_topics:
    if i not in count_.keys():
        count_[i] = 0
    count_[i]+=1

# Plotting a bar chart for number of repos assigned to each topic
plt.figure(figsize=(10, 6))
plt.bar(count_.keys(), count_.values())
plt.xticks(list(count_.keys()))
plt.title('Number of Repos per Topic')
plt.xlabel('Topic IDs')
plt.ylabel('Number of Repos')
plt.show()

# Plotting a Heatmap for each topic's similarity score for each repo
plt.figure(figsize=(15, 10))
sns.heatmap(assigned_topics_list, cmap="YlGnBu")
plt.title('Topic Score Heatmap for Each Repo')
plt.xlabel('Topic IDs')
plt.ylabel('Repos')
plt.show()