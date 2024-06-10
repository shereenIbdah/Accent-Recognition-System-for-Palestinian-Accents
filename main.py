import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier



def display_classification_results(y_test, predictions, encoder):
    # Classification Report
    report = classification_report(y_test, predictions, target_names=encoder.classes_)
    print("Classification Report:\n", report)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {:.2f}%\n".format(accuracy * 100))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return report, accuracy

def extract_features(file_path, n_mfcc=24, hop_length=512, n_fft=2048):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order= 4)
    rms = librosa.feature.rms(y=audio)
    delta_rms = librosa.feature.delta(rms)
    delta2_rms = librosa.feature.delta(rms, order=4)
    combined_features = np.concatenate((np.mean(mfccs.T, axis=0),
                                        np.mean(rms.T, axis=0),
                                        np.mean(delta_rms.T, axis=0),
                                        np.mean(delta2_rms.T, axis=0),
                                        np.mean(delta_mfccs.T, axis=0),
                                        np.mean(delta2_mfccs.T, axis=0)))  # Combine MFCCs with deltas
    return combined_features


def load_data_and_labels(base_dir, regions):
    features = []
    labels = []
    for region in regions:
        print(f"Processing region: {region}")
        region_dir = os.path.join(base_dir, region)
        audio_files = [os.path.join(region_dir, file) for file in os.listdir(region_dir) if file.endswith('.wav')]
        for file_path in audio_files:
            try:
                feature = extract_features(file_path)
                features.append(feature)
                labels.append(region)
                print('*', end=' ')
            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
        print('\n')
    return features, labels

def prepare_data(features, labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, encoded_labels, encoder

def train_classifier(X_train, y_train):
    classifier = SVM(kernel='linear', class_weight='balanced')
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_model(classifier, X_test, y_test, encoder):
    predictions = classifier.predict(X_test)
    report, accuracy = display_classification_results(y_test, predictions, encoder)
    return accuracy

def main():
    base_train_dir = 'voices__train'
    base_test_dir = 'voices__test'
    regions = ['hebron', 'jerusalem', 'nablus', 'ramallah_reef']

    # Load and prepare training data
    train_features, train_labels = load_data_and_labels(base_train_dir, regions)
    X_train, y_train, encoder = prepare_data(train_features, train_labels)

    plt.figure(figsize=(8,6))
    plt.pie([train_labels.count(region) for region in regions], labels=regions, autopct='%1.1f%%', startangle=140)



    plt.title('Data Distribution')
    plt.show()

    # Train the classifier with the training data
    classifier = train_classifier(X_train, y_train)

    # Load and prepare test data
    test_features, test_labels = load_data_and_labels(base_test_dir, regions)
    X_test, y_test, _ = prepare_data(test_features, test_labels)

    # Evaluate the classifier using the test data
    print("Evalution for SVM")
    accuracy = evaluate_model(classifier, X_test, y_test, encoder)
    print("Model Accuracy: {:.2f}%".format(accuracy * 100))


    print("Evalution for KNN")
    classifier = train_KNN(X_train, y_train)
    accuracy = evaluate_model(classifier, X_test, y_test, encoder)
    print(" KNN Model Accuracy: {:.2f}%".format(accuracy * 100))
    print("Evalution for Random Forest")
    classifier = train_RF(X_train, y_train)
    accuracy = evaluate_RF(classifier, X_test, y_test, encoder)
    print(" Random Forest Model Accuracy: {:.2f}%".format(accuracy * 100))





#
## method used KNN
def train_KNN(X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    return classifier

# Random forest
def train_RF(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_RF(classifier, X_test, y_test, encoder):
    predictions = classifier.predict(X_test)
    report, accuracy = display_classification_results(y_test, predictions, encoder)
    return accuracy

if __name__ == '__main__':
    main()
