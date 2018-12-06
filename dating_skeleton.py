import pandas as pd
import numpy as np
from html.parser import HTMLParser
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)
def strip_html_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

df = pd.read_csv("profiles.csv")

drugs_mapping = smokes_mapping = drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
df["drinks_code"] = df.drinks.map(drink_mapping)
df["smokes_code"] = df.drinks.map(smokes_mapping)
df["drugs_code"] = df.drinks.map(drink_mapping)

# Combine essays and strip off html tags
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: strip_html_tags(' '.join(x)), axis=1)

# Add column all essay length
df["essay_len"] = [len(line) for line in all_essays]
# Split into words list for real words
essay_avg_word_len = [sum([len(word) for word in line.split()])/len(line) for line in all_essays]
# Add column word count per all essays
df["essay_avg_word_len"] = essay_avg_word_len
# Add column I/am count per all essays
df["i_me_count"] = pd.Series([sum([x.lower().count('i')+x.lower().count('am') for x in line.split()]) for line in all_essays])


# store normalized data to file
df.to_csv("enriched_profiles.csv")

