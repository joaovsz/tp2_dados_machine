from base import BaseProcessor
import numpy as np

class Questao5:
    def __init__(self):
        self.processor = BaseProcessor()

    def filtrar_features(self):
        X, y = self.processor.get_cancer_data()

        correlations = self.processor.calculate_correlation(X, y)
        selected_features = np.argsort(np.abs(correlations))[-5:]  

        print("\nFeatures selecionadas (filtragem):")
        print(selected_features)

if __name__ == "__main__":
    questao = Questao5()
    questao.filtrar_features()
