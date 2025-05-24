import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

class Questao10:
    def executar(self):
        texto = "A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência."
        stop_words = set(stopwords.words('portuguese'))
        palavras = word_tokenize(texto, language='portuguese')
        palavras_filtradas = [palavra for palavra in palavras if palavra.lower() not in stop_words]

        print("\nTexto sem stopwords:")
        print(" ".join(palavras_filtradas))

if __name__ == "__main__":
    questao = Questao10()
    questao.executar()
