I always had Python debates with my friends; there was always something about me and the language that didn't match up. Who'd have guessed that two years later, I'd be considering a career in Data Science centered on Python? This is a step-by-step guide to completing a data science project that detects fake news. For this project,  I collected a Data Set from (https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)
To begin you will need the following installed in your computer;
1.	Jupyter Notebook that can be installed using Anaconda
2.	Python 3
3.	Download the csv file of the data set from the link shared above.

## **Let's get started**
We'll design a TfidfVectorizer with sklearn and then establish a PassiveAgressorClassiffer to help fit the model.
You'll need to install the following prerequisites before you can use your Jupyter library. We begin by installing numpy sklearn.

```
pip install numpy pandas sklearn
```
Your application will install the required tools and create a fresh input space for you to type your next code.
You'll need to use the codes below to make the appropriate imports for this project;

```
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```
After executing the application, the screen will show no output, indicating that the data set is ready to be read into your notebook.

```
#news is just a variable name we assigned to the project to simply our access to the dataset, you are free to select another name.
news = pd.read_csv(r'C:\Users\lian.s\Desktop\Sign Recognition\news.csv')
#We print out the number of rows and columns
news.shape
#Prints out the top 5 rows in a dataframe or series
news.head()
```
The result will be; 

![If you run the program, you'll notice something like this.](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/8crnz6turnmgk3nqxaot.png)
That means we successfully read the dataset to our notebook and described the top 5 rows and columns. Now we can call the labels from the data set;

```
labels= news.label
labels.head()
```
Congratulations, your project is off to a solid start
![Congratulations, your project is off to a solid start](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/8a8m6kxnsz5ka4dlfl8i.png)
You've undoubtedly dealt with training and testing as a young data scientist. This will be the next phase in the development of our project.
We utilize data to construct a training set, which is a subset used to fit the model, and then we use the trainset to test it. As a result, we must first establish a training set and then test it to convert it to a test set. The models developed are used to predict an unknown outcome.
To accomplish this, we employ;

```
x_train,x_test,y_train,y_test=train_test_split(news['text'], labels, test_size=0.2, random_state=7)

```
We continue by creating a TfidfVectorizer;
Term Frequency (TF) is the number of times a word appears in a document, whereas Inverse Document Frequency (IDF) is the number of times a word appears in one document relative to others. The function creates its own matrix from raw data sets. We first generate the matrix, then fit and transform the vectorizer on the train set, as well as transform the vectorizer on the test set.

```
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
```
With the help of sklearn PassiveAggressiveClassifier, which operates by reacting passively to accurate classifications and aggressively to any misclassifications, we'll be able to calculate the accuracy of our test set.

```
#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```

As you can see, we have an accuracy result of 92.9%

![an accuracy result of 92.9%](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/o4trcnypn4l24c77qmao.png)

Finally, we print out the matrix of how many fake and true news exist amongst our set. This is what we call a confusion matrix.

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

![Results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ll095rzirm07ax3b3471.png)
From our test, we have 588 true positives, 40 false positives , 589 true negatives and 50 false negatives.

There you have it, for more practice you can use the set to calculate over and underfitting. I hope you found this post useful; please share your thoughts in the comments section below.


