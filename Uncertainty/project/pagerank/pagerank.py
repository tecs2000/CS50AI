import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # if page does not points to others pages
    if len(corpus[page]) == 0:
        links = {i: 1/len(corpus) for i in list(corpus)}
    else:
        random = (1 - damping_factor)/len(corpus)
        aux = (damping_factor/len(corpus[page])) + random

        links = {i: aux for i in corpus[page]}

        # chance to any page to be chosen randomly
        for key in set(corpus)-set(links):
            links[key] = random

    return links


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create a dict which maps every key in corpus to a 0 frequency
    pageRank = {i: 0 for i in list(corpus)}

    current = random.choice(list(corpus))
    pageRank[current] += 1

    for i in range(1, n):
        if np.random.uniform() <= damping_factor and corpus[current]:
            prox = np.random.choice(list(corpus[current]))
        else:
            prox = np.random.choice(list(corpus))

        # aux = transition_model(corpus, current, damping_factor)
        # prox = np.random.choice(list(aux), p=list(aux.values()))

        pageRank[prox] += 1
        current = prox

    for key, value in pageRank.items():
        pageRank[key] = value/n

    return pageRank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Search for a page wich does not point to others
    for key in list(corpus):
        if not corpus[key]:
            corpus[key] = list(corpus)

    currentState = {i: 1/len(corpus) for i in list(corpus)}

    while 1:
        proxState = {i: None for i in list(corpus)}

        for page in currentState.keys():
            ranking = PR(page, corpus, currentState, damping_factor)
            proxState[page] = ranking

        diff = False
        for key in proxState.keys():
            if abs(currentState[key]-proxState[key]) > 0.001:
                diff = True
                break

        if not diff:
            return proxState

        currentState = proxState


def incoming(corpus, currentPage):
    incomingLinks = []

    for key in corpus.keys():
        if currentPage in corpus[key]:
            incomingLinks.append(key)

    return incomingLinks


def PR(current, corpus, currentState, damping_factor):
    inLinks = incoming(corpus, current)

    part1 = (1-damping_factor)/len(corpus)

    part2 = 0

    for link in inLinks:
        part2 += currentState[link] / len(corpus[link])

    part2 *= damping_factor

    return part1 + part2


if __name__ == "__main__":
    main()
