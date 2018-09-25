from nltk import pos_tag
from nltk.tokenize import word_tokenize
from math import sqrt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn

# Implementation of techniques described here: https://arxiv.org/pdf/1802.05667.pdf

benchmark_similarity = 0.8025
def commonWords(s1, s2):
    """
    Calculates the number of words shared between two sentences, excluding stop words.
    Then divides by the length of the longer sentence to normalize and returns the result.
    """
    total = 0
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)
    tokens1 = [word for word in tokens1 if word not in stopwords.words('english')]
    tokens2 = [word for word in tokens2 if word not in stopwords.words('english')]
    for word in tokens1:
        if word in tokens2:
            total += 1
    total /= max(len(tokens1), len(tokens2))
    return total

def wordOrderSim(s1, s2):
    """
    Determines the similarity in word order between the two sentences. 
    Creates two vectors
        the first sentence's word order
        the order of appearance for the words in s1 that occur in s2
    Returns the magnitude of the difference between the vectors divided by the product
    """
    s1 = word_tokenize(s1)
    s2 = word_tokenize(s2)
    length = max(len(s1), len(s2))
    wos = 0
    v1 = range(1, length+1)
    v2 = []
    for i in range(len(s2)):
        try:
            v2.append(s1.index(s2[i]))
        except ValueError:
            v2.append(i)
    v2 = [i+1 for i in v2]
    v1xv2 = [a*b for a,b in zip(v1,v2)]
    v1_v2 = [a-b for a,b in zip(v1,v2)]
    wos = magnitude(v1_v2) / magnitude(v1xv2)
    return wos
def simVectors(pos1, pos2):
    """
    Create the similarity vectors for two sentences.
    Calculate the max similarity for each word in pos1.
    This is done by generating the definitions for pos1 and pos2 from WordNet 
    and calculating the path similarities between them
    (Note: only words with matching parts of speech can be compared).
    The resulting vector will be a list of floats.
    """
    global benchmark_similarity
    sim_vec = []
    for word1 in pos1:
        max_sim = 0
        for word2 in pos2:
            if max_sim >= benchmark_similarity:
                break
            # matching parts of speech
            if word1[1] == word2[1]:
                def1 = wn.synsets(word1[0])
                def2 = wn.synsets(word2[0])
                for d1 in def1:
                    if max_sim >= benchmark_similarity:
                        break
                    for d2 in def2:
                        if d1.pos != d2.pos:
                            continue
                        sim = d1.path_similarity(d2)
                        if sim and sim > max_sim:
                            max_sim = sim
                        if max_sim >= benchmark_similarity:
                            break
        sim_vec.append(max_sim)
    return sim_vec
def magnitude(vector):
    """
    v1 = [x_1, x_2 ..., x_n]\n
    total = sqrt(x_1^2 + x_2^2...+x_n^2)
    """
    total = 0
    for num in vector:
        total += (num * num)
    total = sqrt(total)
    return total
def wordSim(t1, t2):
    """
    Tag each token's part of speech
    Compare shortest path distance between each of sentence's words
        with multiple definitions choose max
    """
    pos1 = pos_tag(t1)
    pos2 = pos_tag(t2)
    sim_vec1 = simVectors(pos1, pos2)
    sim_vec2 = simVectors(pos2, pos1)
    return sim_vec1, sim_vec2
def semanticSim(s1, s2):
    """
    Tokenize strings.
    Eliminate words like a, is, to, that, you, such, as, or.
    Calculate word similarity for all words in each sentence.
    For each sentence compare their word similarities to a benchmark 
        increment corresponding variable for all above benchmark
        sum s1, s2 above benchmark tracker and divide by constant
    Multiply magnitude of each sentences similarity vector
        divide by vector length/2 if no sims above benchmark
        divide by above_bench otherwise
    """
    global benchmark_similarity
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)
    tokens1 = [word for word in tokens1 if word not in stopwords.words('english')]
    tokens2 = [word for word in tokens2 if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens1 = [stemmer.stem(t) for t in tokens1]
    tokens2 = [stemmer.stem(t) for t in tokens2]
    sim_vec1, sim_vec2 = wordSim(tokens1, tokens2)
    sim_mag1 = magnitude(sim_vec1) 
    sim_mag2 = magnitude(sim_vec2)
    sim_mag = sim_mag1 * sim_mag2
    sim_constant = 1.8
    above_bench1, above_bench2 = 0, 0
    for sim in sim_vec1:
        if sim > benchmark_similarity:
            above_bench1 += 1
    for sim in sim_vec2:
        if sim > benchmark_similarity:
            above_bench2 += 1
    above_bench = above_bench1 + above_bench2
    above_bench /= sim_constant
    if above_bench > 0:
        similarity = sim_mag / above_bench
    else:
        m = max(len(tokens1), len(tokens2)) / 2
        similarity = sim_mag / m
    return similarity
def sentimentAnalysis(data, new_tc, semantic_weight=1.5, order_weight=1.5, common_weight=1.1, normalize_weight=2.5, num_entries=3):
    """
    Combine word order similarity, semantic similarity, and common words to determine overall similarity of sentences.
    Iterates through a list of data to find the most similar entries to new_tc.
    Returns a list of tuples where each entry is thought to be above benchmark in similarity to new_tc.
    """
    global benchmark_similarity
    most_similar = []
    for entry in data:
        sim = (semanticSim(entry, new_tc) * semantic_weight)
        sim += (wordOrderSim(entry, new_tc) * order_weight)
        sim += (commonWords(entry, new_tc) * common_weight)
        sim /= normalize_weight
        if sim > benchmark_similarity:
            most_similar.append((entry, sim))
        elif len(most_similar) >= num_entries:
            break
    most_similar.sort(key=lambda tup: tup[1], reverse=True)
    return most_similar
if __name__ == '__main__':
    s1 = 'An asylum is a psychiatric hospital.'
    s2 = 'If you describe a place or situation as a madhouse, you mean that it is full of confusion and noise.'
    sim = semanticSim(s1, s2)
    wos = wordOrderSim(s1, s2)
    print(sim)