# Yelp Review Sentiment Analysis

This project focuses on analyzing Yelp reviews to classify sentiment into **Positive**, **Neutral**, and **Negative** categories using natural language processing (NLP) and deep learning techniques. The project also includes data visualizations to provide insights into sentiment trends and review patterns.

---

## **Project Overview**

The analysis involves:
1. **Preprocessing Yelp Reviews**: Cleaning raw review text for efficient analysis.
2. **Sentiment Classification**: Classifying sentiments into Positive, Neutral, and Negative categories.
3. **Transformer-Based Embeddings**: Utilizing `DistilBERT` to generate contextual embeddings.
4. **Deep Learning Model**: Training a custom sentiment classifier using PyTorch.
5. **Evaluation**: Assessing the model's performance through metrics like accuracy, precision, recall, and F1-score.
6. **Visualizations**: Generating word clouds, sentiment distributions, and temporal trends for better insights.

---

## **Stages of the Project**

### **1. Data Loading and Exploration**
- Loaded a subset of the Yelp dataset (`yelp_academic_dataset_review.json`) containing 50,000 reviews.
- Explored the dataset to understand its structure and performed exploratory data analysis (EDA).

### **2. Data Preprocessing**
- Removed URLs, punctuation, and numbers from review text.
- Converted all text to lowercase for uniformity.
- Mapped `stars` ratings into three sentiment categories:
  - **Positive**: `stars > 2`
  - **Neutral**: `stars == 2`
  - **Negative**: `stars < 2`

### **3. Embedding Generation**
- Used `DistilBERT` from Hugging Face to generate contextual embeddings for the cleaned review text.
- Generated embeddings for both training and testing datasets.

### **4. Deep Learning Model**
- Built a custom sentiment classifier using PyTorch:
  - **Architecture**: A fully connected neural network with one hidden layer and dropout regularization.
  - **Loss Function**: Cross-entropy loss.
  - **Optimizer**: Adam optimizer.

### **5. Model Training and Evaluation**
- Trained the model for 100 epochs using the training dataset.
- Evaluated the model on the testing dataset, reporting:
  - Accuracy
  - Precision, Recall, and F1-score
  - Confusion Matrix

### **6. Data Visualizations**
- **Word Clouds**: Showed frequent words for Positive, Neutral, and Negative sentiments.
- **Sentiment Distribution**: Visualized the overall sentiment distribution.
- **Temporal Trends**: Analyzed sentiment trends over time.
- **Most Frequent Words**: Displayed top 10 most common words for each sentiment.
- **Review Length vs Sentiment**: Box plot of review length grouped by sentiment.

---

## **Directory Structure**

```
project_directory/
│
├── data/
│   └── yelp_academic_dataset_review.json  # Input dataset
│
├── notebooks/
│   └── sentiment_analysis.ipynb          # Jupyter Notebook for full analysis
│
├── requirements.txt                      # Dependencies for the project
│
└── README.md                             # Project documentation
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/yelp-sentiment-analysis.git
cd yelp-sentiment-analysis
```

### **2. Install Dependencies**
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**
Launch the Jupyter Notebook to execute the code:
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## **Dependencies**

The project uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`
- `wordcloud`
- `jupyter`

To install these dependencies, use the `requirements.txt` file.

---

## **Key Insights**

1. **Sentiment Trends**:
   - Positive reviews are the most frequent in the dataset.
   - Neutral reviews are rare, indicating polarizing opinions in Yelp reviews.

2. **Review Length**:
   - Positive reviews tend to have longer text, suggesting more detailed feedback.
   - Negative reviews are concise, often indicating dissatisfaction.

3. **Temporal Analysis**:
   - Sentiments show seasonal variation, possibly related to holidays or events.

4. **Model Performance**:
   - Achieved high accuracy in sentiment classification with a well-balanced dataset.

---

## **Future Work**
- **Fine-Tune Transformer**: Improve model performance by fine-tuning `DistilBERT`.
- **Aspect-Based Sentiment Analysis**: Extract sentiments for specific aspects like service, food, or ambiance.
- **Multimodal Analysis**: Incorporate images or other media for richer insights.

---

## **Acknowledgments**
This project utilizes the Yelp dataset from the https://www.yelp.com/dataset.
