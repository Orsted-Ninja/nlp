import numpy as np

class HMM_POS_Tagger:
    def __init__(self):
        self.states = {}
        self.observations = {}
        self.start_prob = {}
        self.transition_prob = {}
        self.emission_prob = {}

    def train(self, tagged_corpus):
        for sentence in tagged_corpus:
            for word, pos_tag in sentence:
                if pos_tag not in self.states:
                    self.states[pos_tag] = len(self.states)
                if word not in self.observations:
                    self.observations[word] = len(self.observations)

        num_states = len(self.states)
        num_observations = len(self.observations)

        self.transition_prob = np.zeros((num_states, num_states))
        self.emission_prob = np.zeros((num_states, num_observations))
        for sentence in tagged_corpus:
            prev_tag = None
            for word, pos_tag in sentence:
                pos_tag_idx = self.states[pos_tag]
                word_idx = self.observations[word]
                if prev_tag is None:
                    self.start_prob[pos_tag_idx] = self.start_prob.get(pos_tag_idx, 0) + 1
                else:
                    self.transition_prob[self.states[prev_tag]][pos_tag_idx] += 1
                self.emission_prob[pos_tag_idx][word_idx] += 1
                prev_tag = pos_tag

        self.start_prob = {k: v / len(tagged_corpus) for k, v in self.start_prob.items()}

        row_sums = self.transition_prob.sum(axis=1, keepdims=True)
        self.transition_prob = np.divide(self.transition_prob, row_sums, where=row_sums != 0)

        row_sums = self.emission_prob.sum(axis=1, keepdims=True)
        self.emission_prob = np.divide(self.emission_prob, row_sums, where=row_sums != 0)

    def predict(self, sentence):
        if not sentence:
            return []

        T = len(sentence)
        N = len(self.states)
        delta = np.zeros((T,N))
        psi = np.zeros((T,N), dtype=int)
        best_path = np.zeros(T, dtype=int)

        for j in range(N):
            word_idx = self.observations.get(sentence[0], None)
            if word_idx is None:
                delta[0,j] = 0
            else:
                delta[0,j] = self.start_prob.get(j,0) * self.emission_prob[j, word_idx]

        for t in range(1, T):
            for j in range(N):
                word_idx = self.observations.get(sentence[t], None)
                if word_idx is None:
                    delta[t,j] = 0
                else:
                    delta[t,j] = np.max(delta[t-1] * self.transition_prob[:,j]) * self.emission_prob[j, word_idx]
                    psi[t,j] = np.argmax(delta[t-1] * self.transition_prob[:,j])

        best_path[T-1] = np.argmax(delta[T-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t+1]]
        predicted_tags = [list(self.states.keys())[i] for i in best_path]
        return list(zip(sentence, predicted_tags))

if __name__ == "__main__":
    tagged_corpus = [[("The","DET"), ("cat","NOUN"), ("runs","VERB")],
                     [("A","DET"), ("dog","NOUN"), ("barks","VERB")]]
    test_sentence = ["The", "dog", "runs"]
    hmm_tagger = HMM_POS_Tagger()
    hmm_tagger.train(tagged_corpus)
    predicted_tags = hmm_tagger.predict(test_sentence)
    print(predicted_tags)

