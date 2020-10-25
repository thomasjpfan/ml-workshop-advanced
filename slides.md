title: Advanced Machine Learning with scikit-learn: Text Data, Imbalanced Data, and Poisson Regression
use_katex: True
class: title-slide

# Advanced Machine Learning with scikit-learn
## Text Data, Imbalanced Data, and Poisson Regression

![](images/scikit-learn-logo-notext.png)

.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-advanced" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-advanced</a>

---

name: table-of-contents
class: title-slide, left

# Table of Contents

1. [Text Data](#text)
1. [Imbalanced Data](#imbalanced)
1. [Poisson Regression](#poisson)

---

name: text
class: chapter-slide

# 1. Text Data

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

---

# Types of text data

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullName</th>
      <th>country</th>
      <th>politicalGroup</th>
      <th>nationalPoliticalGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Magdalena ADAMOWICZ</td>
      <td>Poland</td>
      <td>Group of the European People's Party (Christian Democrats)</td>
      <td>Independent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Asim ADEMOV</td>
      <td>Bulgaria</td>
      <td>Group of the European People's Party (Christian Democrats)</td>
      <td>Citizens for European Development of Bulgaria</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Isabella ADINOLFI</td>
      <td>Italy</td>
      <td>Non-attached Members</td>
      <td>Movimento 5 Stelle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Matteo ADINOLFI</td>
      <td>Italy</td>
      <td>Identity and Democracy Group</td>
      <td>Lega</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alex AGIUS SALIBA</td>
      <td>Malta</td>
      <td>Group of the Progressive Alliance of Socialists and Democrats in the European Parliament</td>
      <td>Partit Laburista</td>
    </tr>
  </tbody>
</table>


---

# Text data we are considering

I've just had the evidence that confirmed my suspicions. A bunch of kids, 14 to 22 put on the DVD of "Titanic" on a fantastic state of the art mega screen home entertainment type deal. Only two of them had actually seen it before. But they all had seen the moment of Kate, Leo and Celine Dion so many times that most of them felt they had seen the whole movie. Shortly after the epic started, they started to get restless, some of them left asking the others

This independent, B&W, DV feature consistently shocks, amazes and amuses with it's ability to create the most insane situations and then find humor and interest in them. It's all hilarious and ridiculous stuff, yet as absurd as much of the film should be, there is a heart and a reality here that keeps the film grounded, keeps the entire piece from drifting into complete craziness and therein lies the real message here.

---

# Bag of words

![:scale 100%](images/bag_of_words.png)

---

# Text processing in scikit-learn

```py
from sklearn.feature_extraction.text import CountVectorizer

sample_text = ["Can we go to the mountain tomorrow?",
               "The mountain is really tall"]

vect = CountVectorizer()
vect.fit(sample_text)

vect.get_feature_names()
# ['be', 'can', 'careful', 'finished', 'go', 'hill', 'homework', 'is', 'my',
# 'please', 'tall', 'the', 'to', 'very', 'we']

X = vect.transform(sample_text)
X.toarray()
# array([[0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
#        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]])
```

---

class: chapter-slide

# Notebook 📓!
## notebooks/01-text-data.ipynb

---

# N-grams

- Tries to keep words together
- "really tall" and "not tall" has different contexts

![](images/single_words.png)

---

class: chapter-slide

# Notebook 📓!
## notebooks/01-text-data.ipynb

---

# Tf-idf rescaling

$$
\text{tf-idf}(t, d) = tf(t, d) \cdot \text{idf}(t)
$$
$$
\text{idf}(t) = \text{log}\frac{1 + n_d}{1 + \text{df}(d, t)} + 1
$$

- $\text{tf}(t, d)$ = The count of term $t$ in document $d$.
- $n_d$ = total number of documents
- $\text{df}(d, t)$ = number of documents containing term $r$

- scikit-learn divides each row by its length (L2 normalization)

```py
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

class: chapter-slide

# Notebook 📓!
## notebooks/01-text-data.ipynb

---

# Notes

???

- Example data
- Text data
- Bag of words
- CounterVectorizer example
- imdb move reviews
- use logistic regression
- stop words
- infrequent words
- n-grams
- tfidf
- words vs characters

---

name: imbalanced
class: chapter-slide

# 2. Imbalanced Data

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

???

- Sources of imbalance
- roc curve
- roc vs average_precision
- resampling
- imbalance-learn
- undersampling
- oversampling
- compare pr and roc curve
- class weight
- balanced bagging
- smote

---

name: poisson
class: chapter-slide

# 3. Poisson Regression

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

???

- What is generalized linear models
- link functions
- Focus on Poisson
- Example notebook
- Tree based regression
- Example notebook

---

class: title-slide, left

# Closing

.g.g-middle[
.g-7[
![:scale 30%](images/scikit-learn-logo-notext.png)
1. [Text data](#text)
1. [Imbalanced Data](#imbalanced)
1. [Poisson Regression](#poisson)
]
.g-5.center[
<br>
.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-advanced" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-advanced</a>
]
]
