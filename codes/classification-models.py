import sys
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path, test_size=0.33, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split_data()

        self.models = {
            'rf': RandomForestClassifier(criterion='gini', max_depth=3, random_state=0),
            'svm': SVC(),
            'dt': DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0),
            'xgb': xgb.XGBClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'nb': GaussianNB(),
            'lr': LogisticRegression(),
            'cnn': self.build_cnn_model(),
            'fnn': self.build_fnn_model(),
            'lstm': self.build_lstm_model()
        }
    
    def load_and_split_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(['label'], axis=1).values
        y = df['label'].values
        
        unique_labels = np.unique(y)
        label_map = {original: new for new, original in enumerate(unique_labels)}
        y_remapped = np.array([label_map[label] for label in y])
        
        # Store mapping for later reference
        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()}
        
        return train_test_split(X, y_remapped, test_size=self.test_size, random_state=self.random_state)
    
    def build_cnn_model(self):
        num_classes = len(np.unique(self.y_train))
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def build_fnn_model(self):
        num_classes = len(np.unique(self.y_train))
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def build_lstm_model(self):
        num_classes = len(np.unique(self.y_train))
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_and_evaluate(self):
        results = {}
        for name, model in self.models.items():
            if name in ['cnn', 'fnn', 'lstm']:
                X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
                model.fit(X_train_reshaped, self.y_train, epochs=10, batch_size=32, verbose=0)
                y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            report = classification_report(self.y_test, y_pred, output_dict=True)
            results[name] = {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': report['weighted avg']['f1-score'],
                'accuracy': accuracy_score(self.y_test, y_pred)
            }
        return results

    def display_results(self, results):
        df_results = pd.DataFrame.from_dict(results, orient='index')
        print("\nModel Performance Metrics:")
        print(df_results.to_string())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <[fbs_nas/msa_nas/fbs_rrc/msa_rrc].csv>")
        sys.exit(1)
    data_path = sys.argv[1]  # Change as needed
    trainer = ModelTrainer(data_path)
    results = trainer.train_and_evaluate()
    trainer.display_results(results)
