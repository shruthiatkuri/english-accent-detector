import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# These match exactly what you trained in Colab
ACCENT_LABELS = ['american', 'australian', 'british', 'indian']
MODEL_SAVE_PATH = "accent_model.pkl"
EMBEDDINGS_PATH = "embeddings.npz"


def load_model():
    """
    Loads the trained model from accent_model.pkl.
    This file was created in Colab and downloaded to this folder.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_SAVE_PATH}' not found. "
            f"Make sure you downloaded it from Google Drive."
        )

    with open(MODEL_SAVE_PATH, "rb") as f:
        model_pipeline = pickle.load(f)

    return model_pipeline


def retrain_from_saved_embeddings():
    """
    Retrains the classifier using saved embeddings (no audio processing needed).
    Use this if you want to experiment with different classifier settings.
    Run with: python train.py
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file '{EMBEDDINGS_PATH}' not found."
        )

    # Load saved embeddings from Colab
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']

    print(f"Loaded embeddings: {X.shape}")
    print(f"Label distribution: { {l: int(c) for l, c in zip(*np.unique(y, return_counts=True))} }")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            
        ))
    ])

    print("\nTraining...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nResults:")
    print(classification_report(y_test, y_pred))

    # Save
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    return pipeline


if __name__ == "__main__":
    retrain_from_saved_embeddings()