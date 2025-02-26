### **What is TF-IDF Vectorization?**

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic used in natural language processing (NLP) to reflect the importance of a word in a document relative to a collection of documents (corpus). It is commonly used for text feature extraction in machine learning models.

---

### **Components of TF-IDF**

1. **Term Frequency (TF)**:
   - Measures how frequently a term appears in a document.
   - Formula:
     \[
     \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
     \]

2. **Inverse Document Frequency (IDF)**:
   - Measures how important a term is across the entire corpus.
   - It downweights terms that appear frequently across documents (e.g., common words like "the", "is").
   - Formula:
     \[
     \text{IDF}(t, D) = \log \frac{\text{Total number of documents in corpus } D}{\text{Number of documents containing term } t}
     \]

3. **TF-IDF**:
   - Combines TF and IDF to give a weight to each term in a document.
   - Formula:
     \[
     \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
     \]

---

### **How TF-IDF Works**
1. **Term Frequency (TF)**:
   - Captures the local importance of a term in a document.
   - Example: If the word "apple" appears 5 times in a document with 100 words, its TF is \( \frac{5}{100} = 0.05 \).

2. **Inverse Document Frequency (IDF)**:
   - Captures the global importance of a term.
   - Example: If the word "apple" appears in 10 out of 1000 documents, its IDF is \( \log \frac{1000}{10} = 2 \).

3. **TF-IDF**:
   - Combines TF and IDF to give a weight to each term.
   - Example: For the word "apple", TF-IDF = \( 0.05 \times 2 = 0.1 \).

---

### **Steps to Compute TF-IDF**
1. **Tokenize the Text**:
   - Split the text into individual words or terms.

2. **Compute Term Frequency (TF)**:
   - Calculate the frequency of each term in each document.

3. **Compute Inverse Document Frequency (IDF)**:
   - Calculate the IDF for each term across the corpus.

4. **Compute TF-IDF**:
   - Multiply TF and IDF for each term in each document.

5. **Vectorize the Text**:
   - Represent each document as a vector of TF-IDF scores.

---

### **Example of TF-IDF**

#### **Corpus**
1. Document 1: "The cat sat on the mat."
2. Document 2: "The dog sat on the log."
3. Document 3: "The cat and the dog are friends."

#### **Step 1: Tokenize and Compute TF**
| **Term** | **Document 1** | **Document 2** | **Document 3** |
|----------|----------------|----------------|----------------|
| the      | 2/5 = 0.4      | 2/5 = 0.4      | 2/6 ≈ 0.33     |
| cat      | 1/5 = 0.2      | 0/5 = 0        | 1/6 ≈ 0.17     |
| sat      | 1/5 = 0.2      | 1/5 = 0.2      | 0/6 = 0        |
| on       | 1/5 = 0.2      | 1/5 = 0.2      | 0/6 = 0        |
| mat      | 1/5 = 0.2      | 0/5 = 0        | 0/6 = 0        |
| dog      | 0/5 = 0        | 1/5 = 0.2      | 1/6 ≈ 0.17     |
| log      | 0/5 = 0        | 1/5 = 0.2      | 0/6 = 0        |
| and      | 0/5 = 0        | 0/5 = 0        | 1/6 ≈ 0.17     |
| are      | 0/5 = 0        | 0/5 = 0        | 1/6 ≈ 0.17     |
| friends  | 0/5 = 0        | 0/5 = 0        | 1/6 ≈ 0.17     |

#### **Step 2: Compute IDF**
| **Term** | **Number of Documents** | **IDF**               |
|----------|-------------------------|-----------------------|
| the      | 3                       | \( \log \frac{3}{3} = 0 \) |
| cat      | 2                       | \( \log \frac{3}{2} ≈ 0.176 \) |
| sat      | 2                       | \( \log \frac{3}{2} ≈ 0.176 \) |
| on       | 2                       | \( \log \frac{3}{2} ≈ 0.176 \) |
| mat      | 1                       | \( \log \frac{3}{1} ≈ 0.477 \) |
| dog      | 2                       | \( \log \frac{3}{2} ≈ 0.176 \) |
| log      | 1                       | \( \log \frac{3}{1} ≈ 0.477 \) |
| and      | 1                       | \( \log \frac{3}{1} ≈ 0.477 \) |
| are      | 1                       | \( \log \frac{3}{1} ≈ 0.477 \) |
| friends  | 1                       | \( \log \frac{3}{1} ≈ 0.477 \) |

#### **Step 3: Compute TF-IDF**
| **Term** | **Document 1** | **Document 2** | **Document 3** |
|----------|----------------|----------------|----------------|
| the      | 0.4 × 0 = 0    | 0.4 × 0 = 0    | 0.33 × 0 = 0   |
| cat      | 0.2 × 0.176 ≈ 0.035 | 0 × 0.176 = 0 | 0.17 × 0.176 ≈ 0.03 |
| sat      | 0.2 × 0.176 ≈ 0.035 | 0.2 × 0.176 ≈ 0.035 | 0 × 0.176 = 0 |
| on       | 0.2 × 0.176 ≈ 0.035 | 0.2 × 0.176 ≈ 0.035 | 0 × 0.176 = 0 |
| mat      | 0.2 × 0.477 ≈ 0.095 | 0 × 0.477 = 0 | 0 × 0.477 = 0 |
| dog      | 0 × 0.176 = 0   | 0.2 × 0.176 ≈ 0.035 | 0.17 × 0.176 ≈ 0.03 |
| log      | 0 × 0.477 = 0   | 0.2 × 0.477 ≈ 0.095 | 0 × 0.477 = 0 |
| and      | 0 × 0.477 = 0   | 0 × 0.477 = 0 | 0.17 × 0.477 ≈ 0.081 |
| are      | 0 × 0.477 = 0   | 0 × 0.477 = 0 | 0.17 × 0.477 ≈ 0.081 |
| friends  | 0 × 0.477 = 0   | 0 × 0.477 = 0 | 0.17 × 0.477 ≈ 0.081 |

---

### **Advantages of TF-IDF**
1. **Handles Common Words**:
   - Downweights terms that appear frequently across documents (e.g., "the", "is").
2. **Captures Important Terms**:
   - Highlights terms that are unique or important to a specific document.
3. **Simple and Effective**:
   - Easy to compute and works well for many text-based tasks.

---

### **Applications of TF-IDF**
1. **Text Classification**:
   - Used as a feature extraction method for tasks like spam detection or sentiment analysis.
2. **Information Retrieval**:
   - Used in search engines to rank documents based on query relevance.
3. **Document Similarity**:
   - Used to compute similarity between documents.

---

### **Summary**
- **TF-IDF** measures the importance of a term in a document relative to a corpus.
- It combines **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.
- It is widely used in NLP for feature extraction and text representation.

---

Let me know if you need further clarification or examples! 😊

## **📌 TF-IDF Vectorization (Term Frequency-Inverse Document Frequency) 📝🔢**  

### **🔹 What is TF-IDF?**  
TF-IDF is a statistical technique used in **Natural Language Processing (NLP)** to convert text into numerical values, helping machines understand the **importance of words** in a document relative to a collection of documents (corpus).  

It is commonly used in **text mining, search engines, and machine learning models** to identify the most relevant words in a document while **reducing the impact of common words like "the", "is", "and", etc.**  

---

## **🔹 Formula for TF-IDF Calculation**  

### ✅ **Step 1: Term Frequency (TF) – How often a word appears in a document**  
\[
TF = \frac{\text{Number of times a term appears in a document}}{\text{Total number of terms in the document}}
\]
- Higher **TF** means the word is important **within the document**.  

### ✅ **Step 2: Inverse Document Frequency (IDF) – How unique a word is across documents**  
\[
IDF = \log \left(\frac{\text{Total number of documents}}{\text{Number of documents containing the term}}\right)
\]
- **Common words (like "the") have low IDF**, reducing their importance.  
- **Rare words (like "quantum") have high IDF**, increasing their importance.  

### ✅ **Step 3: Compute TF-IDF Score**  
\[
TF-IDF = TF \times IDF
\]
- Higher **TF-IDF** means the word is **important in the document but rare in the corpus**.  

---

## **🔹 Example of TF-IDF Calculation 📝**  

**Documents in the Corpus:**  
1️⃣ "Machine learning is amazing"  
2️⃣ "Deep learning is a subset of machine learning"  
3️⃣ "Artificial intelligence and machine learning are related"  

| Word             | TF (Doc 1) | TF (Doc 2) | TF (Doc 3) | IDF  | TF-IDF (Doc 1) |
|-----------------|-----------|-----------|-----------|------|---------------|
| Machine        | 1/4       | 2/7       | 1/6       | log(3/3) = 0 | 0 × TF = 0  |
| Learning       | 1/4       | 2/7       | 1/6       | log(3/3) = 0 | 0 × TF = 0  |
| Amazing        | 1/4       | 0         | 0         | log(3/1) = 1.1 | 1.1 × (1/4) = 0.275 |

👉 **Common words like "machine" & "learning" get lower TF-IDF** (since they appear in all documents).  
👉 **Rare words like "amazing" get higher TF-IDF** (since they appear in only one document).  

---

## **🔹 Why is TF-IDF Useful? 🚀**  

✅ **Removes the dominance of frequent words** (like "is", "the").  
✅ **Identifies important words** in a document.  
✅ **Used in search engines** to rank relevant results.  
✅ **Helps in text classification & sentiment analysis**.  

---

## **🔹 Python Code for TF-IDF in Scikit-Learn 🐍**  

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
documents = [
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "Artificial intelligence and machine learning are related"
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert text to TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Print feature names & matrix
print(vectorizer.get_feature_names_out())  # List of words
print(tfidf_matrix.toarray())  # TF-IDF scores
```

---

## **🔹 Key Takeaways 🎯**  
✅ **TF-IDF helps prioritize important words** while reducing the impact of common words.  
✅ **Used in search engines, text classification, and NLP tasks**.  
✅ **Higher TF-IDF score = More relevant word in a document**.  

Would you like an explanation on Word2Vec or BERT for deep learning-based text embeddings? 😊

