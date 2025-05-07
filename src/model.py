import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('model buiding')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_buiding.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Ensure models directory exists
model_dir = '../models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'loan_approval_model.pkl')

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logger.debug("Model training complete.")
   

    # Save the trained model
    try:
        joblib.dump(model, model_path)
        logger.debug("Model saved at %s", model_path)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    logger.debug("Evaluation complete. Accuracy: %.2f", acc)
    logger.debug("Classification Report:\n%s", report)


