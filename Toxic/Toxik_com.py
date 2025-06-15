import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
import joblib

data = pd.read_csv('C:\\Users\\Egor\\OneDrive\\Рабочий стол\\ЯП\\Python\\Toxik\\labeled.csv')

data['comment'] = data['comment'].str.lower()
data['comment'] = data['comment'].str.replace(r'[^\w\s]', '', regex=True)  # Удаление пунктуации
stop_words = set(stopwords.words('russian'))
data['comment'] = data['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Разделение данных
X = data['comment']
y = data['toxic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование текста в числовой вид
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

f1_lr = f1_score(y_test, lr_pred, pos_label=1)

print("🔹 Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("F1 (токсичные):", f1_lr)
print(classification_report(y_test, lr_pred))


joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
