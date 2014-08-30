#! /usr/bin/env python
# encoding: utf-8

from collections import defaultdict

class Ngrams:
    def __init__(self, source_path_list):
        self.cnt = defaultdict(int)
        for source_path in source_path_list:
            with open(source_path) as r:
                for line in r:
                    words, count = line.strip().split('\t')
                    tup = tuple(word.decode('UTF-8') for word in words.split(' '))
                    count = int(count)
                    self.cnt[tup] = count

    def count(self, *ngram):
        return self.cnt[ngram]

    def vocabs(self):
        return self.cnt.keys()

    def counts(self):
        return self.cnt.items()

if __name__ == '__main__':
    g = Ngrams(['/Users/amylase/corpus/google-ngram-jp/2gms/2gm-0008'])
    for words, count in g.counts():
        if count >= 1000: 
            print '{0}\t{1}'.format(' '.join(words).encode('UTF-8'), count)
