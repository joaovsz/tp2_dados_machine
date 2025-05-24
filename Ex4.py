from base import BaseProcessor
import matplotlib.pyplot as plt

class Questao4:
    def __init__(self):
        self.processor = BaseProcessor()

    def executar(self):
        X, _ = self.processor.get_iris_data()

        # Regularização L1
        X_l1_normalized = self.processor.normalize_data(X, norm_type='l1')

        print("\nDados regularizados com norma L1:")
        print(X_l1_normalized[:5])

        # Verificando soma das features
        print("\nSoma das features normalizadas (L1):")
        print(X_l1_normalized.sum(axis=1)[:5])

        # Gerar gráfico dos dados regularizados (L1)
        plt.figure(figsize=(10, 6))
        plt.boxplot(X_l1_normalized, vert=False, patch_artist=True)
        plt.title('Dados Regularizados (Norma L1)')
        plt.xlabel('Valor Regularizado')
        plt.ylabel('Features')
        plt.savefig('Ex4_dados_regularizados_L1.png')
        plt.close()
        print("Gráfico salvo como 'Ex4_dados_regularizados_L1.png'")

if __name__ == "__main__":
    questao = Questao4()
    questao.executar()
