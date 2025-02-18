import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load dataset
    wine = pd.read_csv('winequality-red.csv', sep=';')
    
    # Pre-process data
    bins = (2, 6.5, 9)
    group_names = ['bad', 'good']
    wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
    
    # Encode quality labels
    label_quality = LabelEncoder()
    wine['quality'] = label_quality.fit_transform(wine['quality'])
    
    return wine

def prepare_train_test_data(wine):
    # Separate features and target
    X = wine.drop('quality', axis=1)
    y = wine['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    
    print("\nRandom Forest Results:")
    print(classification_report(y_test, pred_rfc))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_rfc))
    
    return rfc

def train_svm(X_train, X_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    
    print("\nSVM Results:")
    print(classification_report(y_test, pred_clf))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_clf))
    
    return clf

def train_neural_network(X_train, X_test, y_train, y_test):
    mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)
    
    print("\nNeural Network Results:")
    print(classification_report(y_test, pred_mlpc))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_mlpc))
    
    return mlpc

def main():
    # Load and preprocess data
    wine = load_and_preprocess_data()
    
    # Prepare train/test splits
    X_train, X_test, y_train, y_test = prepare_train_test_data(wine)
    
    # Train and evaluate models
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    svm_model = train_svm(X_train, X_test, y_train, y_test)
    nn_model = train_neural_network(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 