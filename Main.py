import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
from Model import build_model
import os

def main():
    # 1. Load the dataset
    print("Loading data...")
    # Assuming relative path or specific absolute path
    # In this case using the path specified by the user
    data_path = r'd:\Artificial_Neural_Network_Case_Study_data.csv'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}. Please place the file at the correct path or update the code.")
        return

    dataset = pd.read_csv(data_path)
    
    # Matrix of features and dependent variable vector
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    # 2. Preprocess Data
    print("Preprocessing data...")
    # Label Encoding for Gender (column index 2 out of our feature space)
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # One Hot Encoding for Geography (column index 1)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Splitting the dataset into Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Save preprocessing objects for later use in prediction app
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('column_transformer.pkl', 'wb') as f:
        pickle.dump(ct, f)

    # 3. Build Model
    print("Building model...")
    # Input dimension is the number of features after encoding
    input_dim = X_train.shape[1]
    ann = build_model(input_dim)

    # 4. Train Model
    print("Training model...")
    # Batch size 25, Epochs 10 as specified by the user
    ann.fit(X_train, y_train, batch_size=25, epochs=10)

    # 5. Evaluate Model
    print("Evaluating model...")
    loss, accuracy = ann.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # 6. Save the trained model
    print("Saving model...")
    ann.save('churn_model.keras')
    print("Model training and saving complete!")

if __name__ == '__main__':
    main()
