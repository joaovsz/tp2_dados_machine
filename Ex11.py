from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class Questao11:
    def aplicar_stemming(self):
        texto = "A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência."
        stemmer = PorterStemmer()
        palavras = word_tokenize(texto)
        palavras_stem = [stemmer.stem(palavra) for palavra in palavras]

        print("\nTexto com stemming:")
        print(" ".join(palavras_stem))

if __name__ == "__main__":
    questao = Questao11()
    questao.aplicar_stemming()
