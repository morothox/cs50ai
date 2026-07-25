import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> N V
NP -> N | Det N 
AN -> Adj N 
PAn -> P Adj N
S -> N V NP 
S -> N V NP P N 
S -> N V PAN D N V 
S -> N V Det AN
S -> N V P N 
S -> N Adv V Det N Conj N V P NP Adv 
S -> N V Adv Det V Det N 

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    word = sentence.lower()
    res = nltk.word_tokenize(word)
    for words in res.copy():
        alpha = False
        for char in words:
            if char.isalpha():
                alpha = True
                break
        if not alpha:
            res.remove(words)
    return res


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    has_child_np = False
    res = []
    tree.subtrees()
    for s in tree.subtrees():
        if s.label() != "NP":
            continue
        has_child_np = False
        for sub in s.subtrees():
            if sub != s and sub.label() == "NP":
                has_child_np = True
                break

        if not has_child_np:
            res.append(s)
    return res


if __name__ == "__main__":
    main()
