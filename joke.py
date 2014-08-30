#! /usr/bin/env python
# encoding: utf-8
# implementation of "Unsupervised joke generation from big data" (ACL 2013)
# original paper proposes the system that generates 'I like my X like I like my Y, Z' style joke.

from ngram import Ngrams
import wordnet

from argparse import ArgumentParser
from collections import defaultdict
import pickle, random

class JokeCreator:
    def __init__(self, init):
        # list up the word and compute p(x, z), f(z), sense(z), p(z|x), sqrt(sum_z(p(z|x) ** 2))
        if type(init) == str:
            obj_path = init
            with open(obj_path) as r:
                dump = pickle.load(r)
                data_names = ['nouns', 'adjs', 'noun2i', 'adj2i', 'sense', 'adj_freq', 'na_freq', 'cond_prob', 'norm_memo']
                for data_name in data_names:
                    setattr(self, data_name, getattr(dump, data_name))
            return
        else:
            bigram = init

        self.nouns = set()
        self.adjs = set()

        for words, count in bigram.counts():
            for word in words:
                wnwords = wordnet.getWords(word)
                if wnwords:
                    poses = [wnword.pos for wnword in wnwords]
                    if u'n' in poses:
                        # this word has noun usage.
                        self.nouns.add(word)
                    if u'a' in poses:
                        # this word has adjective usage.
                        self.adjs.add(word)

        self.nouns = list(self.nouns)
        self.adjs = list(self.adjs)
        self.noun2i = dict()
        self.adj2i = dict()
        for i, noun in enumerate(self.nouns):
            self.noun2i[noun] = i
        for i, adj in enumerate(self.adjs):
            self.adj2i[adj] = i

        self.sense = []
        for adj in self.adjs:
            wnword = wordnet.getWords(adj)[0]
            self.sense.append(len(wordnet.getSenses(wnword)))

        self.adj_freq = [0] * len(self.adjs)
        noun_freq = [0] * len(self.nouns)
        for words, count in bigram.counts():
            for word in words:
                if word in self.noun2i:
                    noun_freq[self.noun2i[word]] += count
                if word in self.adj2i:
                    self.adj_freq[self.adj2i[word]] += count

        self.na_freq = defaultdict(lambda :defaultdict(int))
        for i, x in enumerate(self.nouns):
            for j, z in enumerate(self.adjs):
                b_count = bigram.count(x, z) + bigram.count(z, x)
                if b_count:
                    self.na_freq[i][j] = b_count

        self.cond_prob = defaultdict(lambda :defaultdict(int))
        self.norm_memo = defaultdict(float)
        for i, x in enumerate(self.nouns):
            for j, z in enumerate(self.adjs):
                if self.na_freq[i][j] > 0:
                    self.cond_prob[i][j] = float(self.na_freq[i][j]) / noun_freq[i]
                    self.norm_memo[i] += self.cond_prob[i][j]
            self.norm_memo[i] = self.norm_memo[i] ** 0.5
                
    def dump(self, dest_path):
        # dump this class with pickle
        with open(dest_path, 'w') as w:
            pickle.dump(self, w)

    def na_sim(self, xi, zi):
        return self.na_freq[xi][zi]

    def word_freq_score(self, zi):
        return 1. / self.adj_freq[zi]

    def sense_score(self, zi):
        return self.sense[zi]

    def nn_distance(self, xi, yi):
        prod = sum(self.cond_prob[xi][zi] * self.cond_prob[yi][zi] for zi in xrange(len(self.adjs)))
        if prod == 0: return 1
        return self.norm_memo[xi] * self.norm_memo[yi] / prod

    def evaluate(self, xi, yi, zi):
        return self.na_sim(xi, zi) * self.na_sim(yi, zi) * self.word_freq_score(zi) * self.sense_score(zi) * self.nn_distance(xi, yi)

    def get_z(self, xi, yi):
        nn_dist = self.nn_distance(xi, yi)
        def partial_evaluate(zi):
            return self.na_sim(xi, zi) * self.na_sim(yi, zi) * self.word_freq_score(zi) * self.sense_score(zi) * nn_dist
        ret = max((partial_evaluate(zi), (xi, yi, zi)) for zi in xrange(len(self.adjs)))
        return ret

    def generate(self, word_x):
        if word_x not in self.noun2i:
            print u'{word_x} is not appeared in the corpus. sorry.'.format(**vars())
            return word_x, word_x, word_x
        xi = self.noun2i[word_x]
        cand_ys = random.sample(xrange(len(self.nouns)), min(500, len(self.nouns)))
        score, (xi, yi, zi) = max(self.get_z(xi, yi) for yi in cand_ys)
        return self.nouns[xi], self.nouns[yi], self.adjs[zi]

def interact(jc):
    while True:
        print u'お題をどうぞ >',
        word_x = raw_input().decode('UTF-8')
        print u'{word_x}とかけまして、'.format(**vars()),
        word_x, word_y, word_z = jc.generate(word_x)
        print u'{word_y}と解きます。その心は、どちらも{word_z}。'.format(**vars())

def train(args):
    corpus_paths = args.corpus if type(args.corpus) == list else [args.corpus]
    ngrams = Ngrams(corpus_paths)
    print 'ngram ok'
    jc = JokeCreator(ngrams)
    print 'joke creator ok'
    print jc.evaluate(0, 1, 2)
    jc.dump('model.dat')

def generate(args):
    model_path = args.model
    jc = JokeCreator(model_path)
    interact(jc)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--corpus', nargs='*', help = 'path to N-gram count list.')
    parser.add_argument('--model', help = 'path to pre-learned model.')
    args = parser.parse_args()

    if args.model:
        generate(args)
    elif args.corpus:
        train(args)
    else:
        parser.print_help()

    

