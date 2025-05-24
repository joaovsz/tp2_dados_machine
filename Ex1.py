from base import BaseProcessor
import matplotlib.pyplot as plt

class Questao1:
    def __init__(self):
        self.processor = BaseProcessor()

    def executar(self):
        X, _ = self.processor.get_iris_data()

        print("Dados originais:")
        print(X[:5])

        X_scaled = self.processor.standard_scale(X)

        print("\nDados normalizados:")
        print(X_scaled[:5])
        print("\nMédia das features normalizadas:", X_scaled.mean(axis=0))
        print("Desvio padrão das features normalizadas:", X_scaled.std(axis=0))

        plt.figure(figsize=(10, 6))
        plt.boxplot(X_scaled, vert=False, patch_artist=True)
        plt.title('Dados Normalizados (StandardScaler)')
        plt.xlabel('Valor Normalizado')
        plt.ylabel('Features')
        plt.savefig('Ex1_dados_normalizados.png')
        plt.close()
        print("Gráfico salvo como 'Ex1_dados_normalizados.png'")

if __name__ == "__main__":
    questao = Questao1()
    questao.executar()
