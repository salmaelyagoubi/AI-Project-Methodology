"""
This is the main entry point of the churn prediction application.
"""
from preprocess import load_data, preprocess_data, split_data
from train import train_model, grid_search
from inference import predict, evaluate_model

if __name__ == "__main__":
    df = load_data("/Users/slaiby/Desktop/Methodo/E Commerce Dataset.xlsx")
    X, y, transformer = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, transformer)
    grid_cv_model = grid_search(model, X_train, y_train)

    y_pred = predict(grid_cv_model, X_test)
    accuracy = evaluate_model(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}%")