from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import argparse
from get_data import get_data
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def m_evaluate(config_file):
    config = get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']
    
    # Load trained model
    model = load_model('saved_models/trained.h5')
    
    # Create test data generator
    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_directory(
        te_set,
        target_size=(225, 225),
        batch_size=batch,
        class_mode=class_mode
    )
    
    # Get dynamic class labels
    label_map = test_set.class_indices
    target_names = list(label_map.keys())
    print("Class Index Map:", label_map)

    # Predict
    Y_pred = model.predict(test_set, steps=len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)

    # Confusion Matrix
    print("Confusion Matrix")
    cm = confusion_matrix(test_set.classes, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/Confusion_Matrix.png')
    # plt.show()

    # Classification Report
    print("Classification Report")
    report = classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).T
    df['support'] = df['support'].astype(int)
    df.to_csv('reports/classification_report.csv')
    print('Reports saved in the "reports" folder.')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    passed_args = args_parser.parse_args()
    m_evaluate(config_file=passed_args.config)
