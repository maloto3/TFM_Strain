
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from pennylane import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import pandas as pd


def plot_kfold_distribution(kf, df, class_names):
    train_counts_per_class = {cls: [] for cls in class_names}
    val_counts_per_class = {cls: [] for cls in class_names}
    fold_numbers = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        y_train_fold = df["class"].values[train_idx]
        y_val_fold = df["class"].values[val_idx]

        train_counts = pd.Series(y_train_fold).value_counts()
        val_counts = pd.Series(y_val_fold).value_counts()

        for cls in class_names:
            train_counts_per_class[cls].append(train_counts.get(cls, 0))
            val_counts_per_class[cls].append(val_counts.get(cls, 0))

        fold_numbers.append(f"Fold {fold_idx+1}")

    df_train = pd.DataFrame({'Fold': fold_numbers})
    df_val = pd.DataFrame({'Fold': fold_numbers})

    for cls in class_names:
        df_train[cls] = train_counts_per_class[cls]
        df_val[cls] = val_counts_per_class[cls]

    df_train_long = pd.melt(df_train, id_vars=['Fold'], value_vars=class_names, 
                            var_name='Clase', value_name='Conteo')
    df_val_long = pd.melt(df_val, id_vars=['Fold'], value_vars=class_names, 
                          var_name='Clase', value_name='Conteo')

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    sns.barplot(x='Fold', y='Conteo', hue='Clase', data=df_train_long, palette='Blues')
    plt.title('Distribución de clases en los conjuntos de Entrenamiento', fontsize=14)
    plt.ylabel('Número de muestras', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Clase')

    plt.subplot(2, 1, 2)
    sns.barplot(x='Fold', y='Conteo', hue='Clase', data=df_val_long, palette='Greens')
    plt.title('Distribución de clases en los conjuntos de Validación', fontsize=14)
    plt.ylabel('Número de muestras', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Clase')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(models, X_test, y_test, class_names):
    # Obtener etiquetas verdaderas (como índices)
    y_true = np.argmax(y_test, axis=1)

    # Calcular predicciones promediadas (ensemble softmax)
    ensemble_preds = np.zeros_like(models[0].predict(X_test, verbose=0))
    for model in models:
        ensemble_preds += model.predict(X_test, verbose=0)
    ensemble_preds /= len(models)

    # Etiquetas predichas
    ensemble_pred_classes = np.argmax(ensemble_preds, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, ensemble_pred_classes)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', colorbar=True, xticks_rotation='horizontal', ax=ax)

    ax.set_title("Matriz de confusión - Modelo Ensemble")
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta verdadera")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(X_test, y_test, models, encoder):
    true_labels = y_test
    ensemble_preds = np.zeros_like(models[0].predict(X_test, verbose=0))
    for model in models:
        ensemble_preds += model.predict(X_test, verbose=0)
    ensemble_preds /= len(models)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = true_labels.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], ensemble_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Graficar curva ROC
    plt.figure(figsize=(6, 5))
    for i in range(n_classes):
        clase = encoder.categories_[0][i]
        plt.plot(fpr[i], tpr[i], label=f'Clase {clase} (AUC={roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC - Ensemble de folds")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.grid(True)
    plt.legend(loc="lower right")
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

        plt.figure(figsize=(6, 5))
        
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


def plot_ensemble_patients(models, histories, X_test, y_test, encoder):
    all_preds = [model.predict(X_test, verbose=0) for model in models]
    ensemble_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    final_preds = [encoder.categories_[0][i] for i in ensemble_preds.argmax(axis=1)]

    df = y_test.copy()
    df["pred"] = final_preds
    df["correct"] = df["pred"] == df["class"]
    df_grouped = df.groupby("patient_name")["correct"].value_counts().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 0.35
    x = np.arange(len(df_grouped))

    correct_vals = df_grouped[True] if True in df_grouped.columns else [0] * len(df_grouped)
    incorrect_vals = df_grouped[False] if False in df_grouped.columns else [0] * len(df_grouped)

    ax.bar(x - bar_width/2, correct_vals, width=bar_width, label="Correcto", color="green")
    ax.bar(x + bar_width/2, incorrect_vals, width=bar_width, label="Incorrecto", color="red")

    # Colores para clases reales
    class_colors = sns.color_palette("hls", len(encoder.categories_[0]))
    class_color_map = dict(zip(encoder.categories_[0], class_colors))
    patient_classes = df.groupby("patient_name")["class"].first()
    colores = [class_color_map[patient_classes[p]] for p in df_grouped.index]

    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped.index)
    for tick, color in zip(ax.get_xticklabels(), colores):
        tick.set_color(color)
    plt.xticks(rotation=90)

    ax.set_xlabel("Paciente")
    ax.set_ylabel("Cantidad de imágenes")
    ax.set_title("Aciertos y errores por paciente - Ensemble")

    # Leyendas dentro del gráfico
    class_handles = [
        mlines.Line2D([], [], marker='o', color=col, label=f'Clase Real {cls}', markersize=10, linestyle='')
        for cls, col in class_color_map.items()
    ]
    result_handles = [
        mlines.Line2D([], [], color='green', lw=4, label='Correcto'),
        mlines.Line2D([], [], color='red', lw=4, label='Incorrecto'),
    ]

    # Coloca la leyenda dentro del gráfico, en la esquina superior derecha
    ax.legend(handles=class_handles + result_handles, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()



def plot_fold_curves(histories):
    colors = ['b', 'g', 'r', 'c', 'm']
    linestyles = ['-', '--', '-.', ':', '-']

    # Gráfica de pérdida
    plt.figure(figsize=(6, 5))
    for i, history in enumerate(histories):
        plt.plot(history['loss'], label=f'Fold {i+1} (Train)', color=colors[i], linestyle=linestyles[0])
        plt.plot(history['val_loss'], label=f'Fold {i+1} (Val)', color=colors[i], linestyle=linestyles[1])

    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curvas de Pérdida por Fold')
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfica de precisión
    plt.figure(figsize=(6, 5))
    for i, history in enumerate(histories):
        plt.plot(history['accuracy'], label=f'Fold {i+1} (Train)', color=colors[i], linestyle=linestyles[0])
        plt.plot(history['val_accuracy'], label=f'Fold {i+1} (Val)', color=colors[i], linestyle=linestyles[1])

    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Curvas de Precisión por Fold')
    plt.legend()
    plt.grid()
    plt.show()

def plot_fold_info(X_test, y_test, encoder, df, models, histories):
    fold_results = []

    for fold_idx, (model, history) in enumerate(zip(models, histories)):
        print(f"Evaluando fold {fold_idx+1}...")
        
        # Predicciones
        preds = model.predict(X_test)
        
        # AUC por clase
        auc_scores = []
        fpr = dict()
        tpr = dict()
        thresholds = dict()

        for i in range(y_test.shape[1]):
            auc = roc_auc_score(y_test[:, i], preds[:, i])
            auc_scores.append(auc)
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], preds[:, i])

        mean_auc = np.mean(auc_scores)
        
        # F1 y ACC
        f1 = f1_score(y_test, np.round(preds), average='micro')
        acc = accuracy_score(y_test, np.round(preds))
        
        # Guardar resultados
        fold_results.append({
            "predictions": preds,
            "history": history,
            "auc_scores": auc_scores,
            "fpr": fpr,
            "tpr": tpr,
            "mean_auc": mean_auc,
            "f1": f1,
            "acc": acc
        })

    for fold_idx, result in enumerate(fold_results):
        print(f"Graficando resultados del fold {fold_idx+1}...")

        preds = result["predictions"]
        history = result["history"]
        auc_scores = result["auc_scores"]
        fpr = result["fpr"]
        tpr = result["tpr"]

        ## 1. Curva ROC
        plt.figure(figsize=(6, 5))
        for i in range(y_test.shape[1]):
            plt.plot(fpr[i], tpr[i], label=f'Class {encoder.categories_[0][i]} (AUC={auc_scores[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve Fold {fold_idx+1}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid()
        plt.show()

        ## 2. Loss & Accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve Fold {fold_idx+1}')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curve Fold {fold_idx+1}')
        plt.legend()
        plt.grid()
        plt.show()

        ## 3. Correctos/Incorrectos por paciente
        df = df.copy()
        df["pred"] = [encoder.categories_[0][i] for i in preds.argmax(axis=1)]
        df["correct"] = df["pred"] == df["class"]
        df_grouped = df.groupby("patient_name")["correct"].value_counts().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 5))
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
        ax.set_title(f"Aciertos y errores por paciente - Fold {fold_idx+1}")
        handle_azul = mlines.Line2D([], [], marker='o', color='blue', label='Clase Real C', markersize=10)
        handle_negro = mlines.Line2D([], [], marker='o', color='black', label='Clase Real MCA', markersize=10)
        handle_correcto = mlines.Line2D([], [], color='green', lw=4, label='Correcto')
        handle_incorrecto = mlines.Line2D([], [], color='red', lw=4, label='Incorrecto')
        ax.legend(handles=[handle_azul, handle_negro, handle_correcto, handle_incorrecto])

        plt.show()