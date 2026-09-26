import re
import random
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self.analyze_bigrams(text)

    def simple_tokenizer(self, text, frequency_threshold=5):
        tokens = re.findall(r"\b\w+\b", text.lower())

        if not frequency_threshold:
            return tokens

        word_counts = Counter(tokens)

        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]

        return filtered_tokens

    def analyze_bigrams(self, text, frequency_threshold=None):
        words = self.simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, num_words=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break

            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)