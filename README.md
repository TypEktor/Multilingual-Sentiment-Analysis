# Multilingual-Sentiment-Analysis
---

This work comes from a recent research project I volunteered to help out with the Aristotle University of Thessaloniki for investigating the sentiment of Twitter accounts. As part of the initial research, an extensive Multilingual Sentiment Analysis on a set of tweets was necessary for extracting useful insights.

A high-level Sentiment analysis Pipeline flowchart

![Multilingual Sentiment Analysis](https://user-images.githubusercontent.com/32909949/166109672-2378697d-ed71-4ccc-bdf0-bf5a7d10b5cf.png)

---

## Contents
- **FunctionsMLSA.py:**  A file that contains a set of general-purpose functions. Mostly used within the Multi-Sentiment_Analysis.
- **Multi_Sentiment_Analysis.py:** A class for creating the Multi-Sentiment Analysis object and calling all the relevant functions (Plots, Text Cleaning, etc.).
- **main.py:** The file that contains the main function
- **Output:** A folder containing all the outputs from executing the project
- **requirement.txt:** File that contains all the packages used for this project

---

## Data
The tweet extraction has been done using vicinitas.io. Tweets from 10 accounts have been selected and their tweets have been saved in different spreadsheets (CSV).  This dataset contains 36k tweets that are not classified – labeled.

At this point, the data can't be published and thus can't be uploaded to GitHub

---

## Pre-requisites
The project was developed using python 3.6.13. There is a requirement.txt file that contains all the appropriate packages and their versions for this project.
Furthermore there are also two levixons that must be downloaded:
- nltk.download('vader_lexicon')
- nltk.download('wordnet')

Installation with pip:

```pip install -r requirements.txt```

---

## How to use
- Have Python >= 3.6 installed on your machine
- Clone or download this repository
- Create a folder called Data and add your spreadsheets that contain your tweets
- In a shell, execute the main.py script with Python 3

---

## Resources

Useful Resources:

- https://aip.scitation.org/doi/pdf/10.1063/1.5136197

- https://github.com/UmarIgan/Python-Turkish-Sentiment-Analysis/blob/master/Twitter%20Api.ipynb

- https://medium.com/analytics-vidhya/rule-based-sentiment-analysis-with-python-for-turkeys-stock-market-839f85d7daaf

- https://github.com/otuncelli/turkish-stemmer-python

- https://www.google.com/searchq=sentiment+analysis+vader+for+multilingualism&oq=sentiment+analysis+VADER+for+multiliguar&aqs=chrome.1.69i57j33i10i160.24745j0j4&sourceid=chrome&ie=UTF-8

#
---

## Work in progress
