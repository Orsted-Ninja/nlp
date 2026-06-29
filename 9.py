import math
from collections import Counter

def calculate_tf_idf(document):
    document_text = " ".join(document)
    words = document_text.split()
    word_freq = Counter(words)
    total_words = len(words)
    tf = {word: word_freq[word] / total_words for word in word_freq}
    num_documents = len(document)
    idf = {}
    for word in word_freq:
        num_docs_with_word = sum(1 for doc in document if word in doc.split())
        if num_docs_with_word > 0:
            idf[word] = math.log(num_documents / num_docs_with_word)
        else:
            idf[word] = 0
    tf_idf = {word: tf[word] * idf[word] for word in word_freq}
    return tf_idf

if __name__ == "__main__":
    document = ["python is program",
                "machine is fun fun fun",
                "python is machine"
               ]
    tf_idf_values = calculate_tf_idf(document)
    print("TF-IDF values of each word in the document:")
    for word, tf_idf in tf_idf_values.items():
        print(f"{word}: {tf_idf:.4f}")
