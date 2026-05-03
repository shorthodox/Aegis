import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class ModelTrainer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            objective='binary:logistic', # Predicts probability
            tree_method='hist' # Faster training for large datasets
        )

    def train(self, df):
        # Define Features (X) and Target (y)
        X = df.drop(columns=['target', 'timestamp'])
        y = df['target']

        # Split into training and testing (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        print(f"Training XGBoost for {self.symbol}...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Quick Accuracy Check
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model Accuracy for {self.symbol}: {acc * 100:.2f}%")

        # Save the model to src/ml/model_store/
        filename = f"src/ml/model_store/{self.symbol.replace('/', '_')}_model.bin"
        self.model.save_model(filename)
        print(f"Model saved as {filename}")