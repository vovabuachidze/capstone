import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Can we predict sex with education level and income??


df = pd.read_csv("enriched_profiles.csv")

#creating labelEncoder
le = preprocessing.LabelEncoder()


#add feature data and drop empty lines
feature_data = df[["income", "education"]].replace(-1, np.nan)

# map labels
labels = le.fit_transform(df.sex)
le_sex_mapping = dict(zip(le.classes_, le.fit_transform(le.classes_)))


# Converting string labels into numbers.
education_code = le.fit_transform(feature_data.education.astype(str))
le_education_mapping = dict(zip(le.classes_, le.fit_transform(le.classes_)))
income_code = le.fit_transform(feature_data.income.astype(str))
le_income_mapping = dict(zip(le.classes_, le.fit_transform(le.classes_)))


X = [[education_code[i], income_code[i]] for i in range(len(education_code))]


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, labels)
test = [key for (key, value) in le_sex_mapping.items() if value == classifier.predict([[1, 1]])]
print(test[0])


