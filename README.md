# Trigram Language Model using NLP

## Project Overview

This project implements a **Trigram Language Model** using **Natural Language Processing (NLP)** techniques. The model generates text based on probability distributions and evaluates performance using **Perplexity**.

This project also compares different smoothing techniques:

* Laplace Smoothing
* Linear Interpolation (Unigram + Bigram + Trigram)

The dataset used is **Shakespeare's Julius Caesar** from **NLTK Gutenberg Corpus**.

---

## Features

* Text preprocessing
* Trigram language model implementation
* Laplace smoothing
* Linear interpolation
* Text generation
* Perplexity calculation
* Performance comparison

---

## Technologies Used

* Python
* NLTK
* Collections
* Math
* Random

---

## Dataset

Dataset used in this project:

**Shakespeare — Julius Caesar**
Source: **NLTK Gutenberg Corpus**

The dataset is automatically downloaded using NLTK. If downloading fails, the program uses a fallback dummy dataset.

---

## How It Works

### 1. Text Preprocessing

The text is processed by:

* Converting to lowercase
* Removing punctuation
* Tokenizing words

---

### 2. Trigram Model

The trigram model predicts the next word using:

(w1, w2) → next word

Example:

```
the king → is  
king is → dead  
```

---

### 3. Laplace Smoothing

Laplace smoothing prevents zero probability for unseen words.

Formula:

```
P = (count + 1) / (total + vocabulary size)
```

---

### 4. Text Generation

The model generates sentences using probability distribution.

Example:

Seed Words:

```
the king
```

Generated Output:

```
the king was a noble man and the king ruled the land
```

---

### 5. Perplexity Evaluation

Perplexity measures model performance.

* Lower Perplexity = Better Model
* Higher Perplexity = Poor Model

---

## Linear Interpolation

The project combines:

* Unigram
* Bigram
* Trigram

Using:

```
P = λ1 * unigram + λ2 * bigram + λ3 * trigram
```

This improves model accuracy.

---

## Results

| Method               | Performance          |
| -------------------- | -------------------- |
| Laplace Smoothing    | Higher Perplexity    |
| Linear Interpolation | Lower Perplexity     |
| Best Model           | Linear Interpolation |

---

## Installation

Install dependencies:

```
pip install nltk
```

---

## Run the Project

```
python ngram_model.py
```

---

## Output

The program displays:

* Vocabulary size
* Generated text
* Perplexity score
* Performance comparison

---

## Author

Joydip Dey
Computer Science Student
Natural Language Processing Project

---

## Course

Natural Language Processing (NLP)

---

## Future Improvements

* Use larger dataset
* Implement neural language models
* Improve text coherence
* Add visualization

---

## License

This project is created for academic and educational purposes.
