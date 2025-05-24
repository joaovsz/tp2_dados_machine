import numpy as np
from base import BaseProcessor

class Questao2:
    def __init__(self):
        self.processor = BaseProcessor()

    def comparar_normalizacoes(self):
        X, _ = self.processor.get_iris_data()

        X_manual_scaled = self.processor.manual_standard_scaler(X)

        X_sklearn_scaled = self.processor.standard_scale(X)

        print("\nDados escalonados manualmente:")
        print(X_manual_scaled[:5])
        print("\nDiferença entre os métodos:")
        print(np.abs(X_manual_scaled - X_sklearn_scaled).max())

if __name__ == "__main__":
    questao = Questao2()
    questao.comparar_normalizacoes()
