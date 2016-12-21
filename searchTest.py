import sqlite3
import numpy as np
from lib import vectorize
from lib import text_searcher

vec = vectorize.Vectorizer("char1.txt", "label.txt")
searcher = text_searcher.TextSearcher("corpus.sqlite")

termsFile = open("training_terms.tsv", "r", encoding="utf-8", newline="\n").read().strip().split("\n")

training_terms = []
for row in termsFile:
    r = row.split("\t")
    x = r[0]
    y = r[1].split(" ")
    training_terms.append((x, y))

print(training_terms)

query = ""

for terms in training_terms:
    query = terms[0]
    print(query)
    n = 0
    for doc in searcher.genDocs(query):
        n += 1
    print(n)
