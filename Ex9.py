from sklearn.feature_extraction.text import CountVectorizer

class Questao9:
    def gerar_ngrams(self):
        texto = "A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência."
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        ngrams = vectorizer.fit_transform([texto])

        print("\nNúmero de unigrams:", len([word for word in vectorizer.get_feature_names_out() if len(word.split()) == 1]))
        print("Número de bigrams:", len([word for word in vectorizer.get_feature_names_out() if len(word.split()) == 2]))
        print("Número de trigrams:", len([word for word in vectorizer.get_feature_names_out() if len(word.split()) == 3]))

if __name__ == "__main__":
    questao = Questao9()
    questao.gerar_ngrams()
