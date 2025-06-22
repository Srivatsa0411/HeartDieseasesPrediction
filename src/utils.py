from sklearn.metrics import classification_report, f1_score
import pandas as pd

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print(f"🌟 Validation F1: {f1_score(y_val, y_pred):.5f}\n")
    print(classification_report(y_val, y_pred))

def save_submission(test_df, predictions, output_path='y_predict.csv'):
    pd.DataFrame({
        'PatientID': test_df['PatientID'],
        'HeartDisease': predictions
    }).to_csv(output_path, index=False)
    print(f"✅ Saved {output_path}")
