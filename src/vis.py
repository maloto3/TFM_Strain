
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns
from tensorflow.keras.models import load_model

from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import numpy as np

import pennylane as qml
from pennylane import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import pickle

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def plot_kfold_distribution(kf, df, class_names):
    train_c_counts = []
    train_mca_counts = []
    val_c_counts = []
    val_mca_counts = []
    fold_numbers = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        y_train_fold = df["class"].values[train_idx]
        y_val_fold = df["class"].values[val_idx]

        train_counts = pd.Series(y_train_fold).value_counts()
        train_c_counts.append(train_counts.get("C", 0))
        train_mca_counts.append(train_counts.get("MCA", 0))

        val_counts = pd.Series(y_val_fold).value_counts()
        val_c_counts.append(val_counts.get("C", 0))
        val_mca_counts.append(val_counts.get("MCA", 0))
        
        fold_numbers.append(f"Fold {fold_idx+1}")

    df_train = pd.DataFrame({
        'Fold': fold_numbers,
        'C': train_c_counts,
        'MCA': train_mca_counts
    })

    df_val = pd.DataFrame({
        'Fold': fold_numbers,
        'C': val_c_counts,
        'MCA': val_mca_counts
    })

    df_train_long = pd.melt(df_train, id_vars=['Fold'], value_vars=class_names, 
                            var_name='Clase', value_name='Conteo')
    df_val_long = pd.melt(df_val, id_vars=['Fold'], value_vars=class_names, 
                        var_name='Clase', value_name='Conteo')

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 1, 1)
    sns.barplot(x='Fold', y='Conteo', hue='Clase', data=df_train_long, palette='Blues')
    plt.title('Distribución de clases en los conjuntos de Entrenamiento', fontsize=14)
    plt.ylabel('Número de muestras', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Clase')

    for p in plt.gca().patches:
        plt.gca().annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', fontsize=10)

    plt.subplot(2, 1, 2)
    sns.barplot(x='Fold', y='Conteo', hue='Clase', data=df_val_long, palette='Blues')
    plt.title('Distribución de clases en los conjuntos de Validación', fontsize=14)
    plt.ylabel('Número de muestras', fontsize=12)
    plt.xlabel('Fold', fontsize=12)
    plt.legend(title='Clase')

    for p in plt.gca().patches:
        plt.gca().annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(models, X_test, y_test, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Convert one-hot encoded labels to class indices
    y_true = np.argmax(y_test, axis=1)
    
    for i, model in enumerate(models):
        # Make predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Plot confusion matrix
        ax = axes[i]

        for edge in ['top', 'right', 'bottom', 'left']:
            ax.spines[edge].set_visible(False)
        ax.grid(False)

        cm = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        cm_display.plot(cmap='Blues', colorbar=True, xticks_rotation='horizontal', ax=ax)

        ax.set_title(f'Model {i+1} Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Add an overall confusion matrix (ensemble)
    if len(models) > 1:
        ax = axes[5]
        # Ensemble predictions (average probabilities)
        ensemble_preds = np.zeros_like(models[0].predict(X_test, verbose=0))
        for model in models:
            ensemble_preds += model.predict(X_test, verbose=0)
        ensemble_preds /= len(models)
        
        # Get predicted class labels
        ensemble_pred_classes = np.argmax(ensemble_preds, axis=1)
        
        # Compute confusion matrix for ensemble
        cm_ensemble = confusion_matrix(y_true, ensemble_pred_classes)
        
        for edge in ['top', 'right', 'bottom', 'left']:
            ax.spines[edge].set_visible(False)
        ax.grid(False)

        cm = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=class_names)
        cm_display.plot(cmap='Blues', colorbar=True, xticks_rotation='horizontal', ax=ax)
        
        axes[5].set_title('Ensemble Model Confusion Matrix')
        axes[5].set_ylabel('True Label')
        axes[5].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()


def pad_history(metric_list, histories):
    max_epochs = max(len(h['loss']) for h in histories)
    return [np.pad(h[metric_list], (0, max_epochs - len(h[metric_list])), constant_values=np.nan) for h in histories]

def plot_metric(metric_array, val_array, title, ylabel):
        epochs = np.arange(metric_array.shape[1])
        
        train_mean = np.nanmean(metric_array, axis=0)
        train_std = np.nanstd(metric_array, axis=0)

        val_mean = np.nanmean(val_array, axis=0)
        val_std = np.nanstd(val_array, axis=0)

        plt.figure(figsize=(12, 6))
        
        # Train
        plt.plot(epochs, train_mean, label='Train (Mean)', color='blue')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)

        # Validation
        plt.plot(epochs, val_mean, label='Validation (Mean)', color='orange')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)

        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_ensemble_curve(histories):
    losses = pad_history('loss',histories)
    val_losses = pad_history('val_loss',histories)
    accuracies = pad_history('accuracy',histories)
    val_accuracies = pad_history('val_accuracy',histories)

    losses = np.array(losses)
    val_losses = np.array(val_losses)
    accuracies = np.array(accuracies)
    val_accuracies = np.array(val_accuracies)

    plot_metric(losses, val_losses, 'Curva de Pérdida (Ensemble)', 'Pérdida')
    plot_metric(accuracies, val_accuracies, 'Curva de Precisión (Ensemble)', 'Precisión')


def plot_ensemble_patients(models,histories, X_test, y_test, enconder):
    all_preds = [model.predict(X_test, verbose=0) for model in models]

    ensemble_preds = np.mean(np.stack(all_preds, axis=0), axis=0)

    final_preds = [enconder.categories_[0][i] for i in ensemble_preds.argmax(axis=1)]

    df = y_test.copy()
    df["pred"] = final_preds
    df["correct"] = df["pred"] == df["class"]
    df_grouped = df.groupby("patient_name")["correct"].value_counts().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(df_grouped))

    ax.bar(x - bar_width/2, df_grouped[True], width=bar_width, label="Correcto", color="green")
    ax.bar(x + bar_width/2, df_grouped[False], width=bar_width, label="Incorrecto", color="red")

    colores = ["blue" if df[df["patient_name"] == p]["class"].iloc[0] == "C" else "black" for p in df_grouped.index]
    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped.index)
    for tick, color in zip(ax.get_xticklabels(), colores):
        tick.set_color(color)
    plt.xticks(rotation=90)

    ax.set_xlabel("Paciente")
    ax.set_ylabel("Cantidad de imágenes")
    ax.set_title("Aciertos y errores por paciente - Ensemble")

    handle_azul = mlines.Line2D([], [], marker='o', color='blue', label='Clase Real C', markersize=10)
    handle_negro = mlines.Line2D([], [], marker='o', color='black', label='Clase Real MCA', markersize=10)
    handle_correcto = mlines.Line2D([], [], color='green', lw=4, label='Correcto')
    handle_incorrecto = mlines.Line2D([], [], color='red', lw=4, label='Incorrecto')
    ax.legend(handles=[handle_azul, handle_negro, handle_correcto, handle_incorrecto])

    plt.tight_layout()
    plt.show()