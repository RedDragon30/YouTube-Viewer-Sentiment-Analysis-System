import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': matrix
        }
        
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
        print("\nConfusion Matrix:\n", matrix)
        plt.figure(figsize=(8,6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        logging.info("Model Evaluation complete.")
        return metrics
    except Exception as e:
        logging.error(f'Error in Model Evaluation: {e}')
        raise e