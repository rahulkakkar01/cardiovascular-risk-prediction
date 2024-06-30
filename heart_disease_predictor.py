import tkinter as tk
from tkinter import ttk, messagebox
from joblib import load
import pandas as pd

# Load the trained model
model = load('model.joblib')

# Define the application
class HealthPredictor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Health Condition Predictor")
        self.geometry("600x600")

        # Create a scrollable canvas
        canvas = tk.Canvas(self)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)

        # Add widgets to the scrollable frame
        self.create_widgets_heart_disease()
        self.create_widgets_additional_conditions()

        # Predict Button
        predict_button = ttk.Button(self.scrollable_frame, text="Predict", command=self.predict)
        predict_button.pack(pady=20)

        # Configure the canvas to resize with the window
        self.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def create_widgets_heart_disease(self):
        # Labels and Entries for heart disease prediction
        labels_heart_disease = [
            "General Health (Excellent, Very Good, Good, Fair, Poor):",
            "Checkup (Never, <1 year, 1 year, 2 years, 3 years, >3 years):",
            "Exercise (None, Occasional, Regular):",
            "Sex (Male, Female):",
            "Age Category (18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+):",
            "Smoking History (Never smoker, Former smoker, Current smoker):",
            "Height (cm):",
            "BMI:",
            "Fruit Consumption per day:",
            "Alcohol Consumption per week:",
            "Green Vegetables Consumption per day:",
            "Fried Potato Consumption per week:"
        ]

        self.entries_heart_disease = {}
        for label in labels_heart_disease:
            lbl = ttk.Label(self.scrollable_frame, text=label)
            lbl.pack(anchor=tk.W, padx=10, pady=5)
            entry = ttk.Entry(self.scrollable_frame)
            entry.pack(fill=tk.X, padx=10, pady=5)
            self.entries_heart_disease[label] = entry

    def create_widgets_additional_conditions(self):
        # Labels and Checkbuttons for additional health conditions
        labels_conditions = {
            "Diabetes": "No",
            "Arthritis": "No",
            "Other Cancer": "No",
            "Depression": "No"
        }

        self.condition_vars = {}
        for label, default_value in labels_conditions.items():
            lbl = ttk.Label(self.scrollable_frame, text=f"{label} (Yes/No):")
            lbl.pack(anchor=tk.W, padx=10, pady=5)
            var = tk.StringVar(value=default_value)
            chk = ttk.Checkbutton(self.scrollable_frame, text="Yes", variable=var, onvalue="Yes", offvalue="No")
            chk.pack(anchor=tk.W, padx=10, pady=5)
            self.condition_vars[label] = var

    def predict(self):
        # Gather input data for heart disease prediction
        try:
            data_heart_disease = {
                'General_Health': self.entries_heart_disease["General Health (Excellent, Very Good, Good, Fair, Poor):"].get(),
                'Checkup': self.entries_heart_disease["Checkup (Never, <1 year, 1 year, 2 years, 3 years, >3 years):"].get(),
                'Exercise': self.entries_heart_disease["Exercise (None, Occasional, Regular):"].get(),
                'Sex': self.entries_heart_disease["Sex (Male, Female):"].get(),
                'Age_Category': self.entries_heart_disease["Age Category (18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+):"].get(),
                'Smoking_History': self.entries_heart_disease["Smoking History (Never smoker, Former smoker, Current smoker):"].get(),
                'Height_(cm)': float(self.entries_heart_disease["Height (cm):"].get()),
                'BMI': float(self.entries_heart_disease["BMI:"].get()),
                'Fruit_Consumption': float(self.entries_heart_disease["Fruit Consumption per day:"].get()),
                'Alcohol_Consumption': float(self.entries_heart_disease["Alcohol Consumption per week:"].get()),
                'Green_Vegetables_Consumption': float(self.entries_heart_disease["Green Vegetables Consumption per day:"].get()),
                'FriedPotato_Consumption': float(self.entries_heart_disease["Fried Potato Consumption per week:"].get()),
                'Diabetes': self.condition_vars["Diabetes"].get(),
                'Arthritis': self.condition_vars["Arthritis"].get(),
                'Other_Cancer': self.condition_vars["Other Cancer"].get(),
                'Depression': self.condition_vars["Depression"].get()
            }
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for heart disease prediction: {e}")
            return

        df_heart_disease = pd.DataFrame([data_heart_disease])

        # Make heart disease prediction
        prediction_heart_disease = model.predict(df_heart_disease)
        probability_heart_disease = model.predict_proba(df_heart_disease)[:, 1][0]

        prediction_label_heart_disease = 'Yes' if prediction_heart_disease[0] == 1 else 'No'

        # Show heart disease result
        result_text = f"Heart Disease Prediction: {prediction_label_heart_disease}\nProbability: {probability_heart_disease:.2f}\n\n"

        # Show additional health conditions
        result_text += "Additional Health Conditions:\n"
        result_text += f"Diabetes: {data_heart_disease['Diabetes']}\n"
        result_text += f"Arthritis: {data_heart_disease['Arthritis']}\n"
        result_text += f"Other Cancer: {data_heart_disease['Other_Cancer']}\n"
        result_text += f"Depression: {data_heart_disease['Depression']}\n"

        messagebox.showinfo("Prediction Result", result_text)

if __name__ == "__main__":
    app = HealthPredictor()
    app.mainloop()
