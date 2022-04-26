import Multi_Sentiment_Analysis as msa
import os

def main():
    # Pick used method 'Vader', 'Text Full Cleaned', 'Vader Lemmatization'
    sa = msa.MultiSA("Vader Lemmatization", os.path.abspath(os.path.dirname(__file__)))
    # Read each file and merge
    df = sa.GetData()
    # Cleaning text based on the method and
    df = sa.Cleaning(df)
    df = sa.VADER(df)
    # sa.PiePlots(df)

if __name__ == "__main__":
    main()