import tkinter as tk
from tkinter import messagebox
import joblib

# Загрузка модели и векторизатора (делай это один раз при старте программы)
model = joblib.load('lr_model.pkl')
vectorize = joblib.load('vectorizer.pkl')

def classify_comment():
    comment = entry.get()
    if not comment:
        messagebox.showwarning("Внимание", "Введите комментарий!")
        return
    
    # Преобразуем введённый текст в числовой вид
    vectorized_comment = vectorize.transform([comment])
    
    # Предсказание
    prediction = model.predict(vectorized_comment)[0]
    
    if prediction == 1:
        result_label.config(text=f"Комментарий негативный")
    else:
        result_label.config(text=f"Комментарий положительный")

root = tk.Tk()
root.title("Анализ комментариев")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

entry_label = tk.Label(frame, text="Введите комментарий:")
entry_label.pack()

entry = tk.Entry(frame, width=50)
entry.pack()

classify_button = tk.Button(frame, text="Оценить", command=classify_comment)
classify_button.pack()

result_label = tk.Label(frame, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()
