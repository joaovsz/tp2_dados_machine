from base import BaseProcessor
import matplotlib.pyplot as plt

class Questao3:
    def __init__(self):
        self.processor = BaseProcessor()

    def regularizar_l2(self):
        X, _ = self.processor.get_iris_data()

        X_l2_normalized = self.processor.normalize_data(X, norm_type='l2')

        print("\nDados regularizados com norma L2:")
        print(X_l2_normalized[:5])

        plt.figure(figsize=(10, 6))
        plt.boxplot(X_l2_normalized, vert=False, patch_artist=True)
        plt.title('Dados Regularizados (Norma L2)')
        plt.xlabel('Valor Regularizado')
        plt.ylabel('Features')
        plt.savefig('Ex3_dados_regularizados_L2.png')
        plt.close()
        print("Gráfico salvo como 'Ex3_dados_regularizados_L2.png'")

if __name__ == "__main__":
    questao = Questao3()
    questao.regularizar_l2()
