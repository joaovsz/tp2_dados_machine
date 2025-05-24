from base import BaseProcessor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

class Questao6:
    def __init__(self):
        self.processor = BaseProcessor()

    def executar(self):
        X, y = self.processor.get_cancer_data()

        model = LogisticRegression(max_iter=10000)
        rfe = RFE(model, n_features_to_select=5)
        rfe.fit(X, y)

        print("\nFeatures selecionadas (Wrapper):")
        print(np.where(rfe.support_)[0])

        selected_features = np.where(rfe.support_)[0]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(selected_features)), selected_features, color='blue')
        plt.xlabel('Índice da Feature')
        plt.ylabel('Valor')
        plt.title('Features Selecionadas (Wrapper)')
        plt.savefig('Ex6_features_selecionadas.png')
        plt.close()
        print("Gráfico salvo como 'Ex6_features_selecionadas.png'")

if __name__ == "__main__":
    questao = Questao6()
    questao.executar()
