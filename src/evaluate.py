from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def avaliar_modelo(modelo, X_teste, y_teste, nome_modelo: str = "Modelo"):
    """
    Realiza predições, imprime o classification report e plota a matriz de confusão.
    """
    y_pred = modelo.predict(X_teste)
    
    print("\n" + "="*60)
    print(f"RELATÓRIO DE CLASSIFICAÇÃO - {nome_modelo.upper()}")
    print("="*60)
    print(classification_report(y_teste, y_pred))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_pred, cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax
    )
    plt.title(f"Matriz de Confusão - {nome_modelo}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()