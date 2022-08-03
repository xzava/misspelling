import numpy as np
import numba as nb

import random
from functools import reduce
import string
import math 

from typing import List

select_index = lambda s: random.randint(0, len(s)-1) if len(s) else 0

'''

random_nearist_letter(l)    --> Pick a random letter near given letter, using keyboard layout. Ie fat fingers.
random_letter()             --> Pick a random printable, with a preference for letters.
insertion(l, index=None)    --> Insert a random letter
deletion(l, index=None)     --> Delete a random letter
substitution(l, index=None) --> Change a random letter
transposition(l)            --> Swap two adjacent letters
create_misspelling(s, edit_distance=3, _force=False, _weights=None)  --> 61154: Create mispellings based on levenshtein edit distance 
levenshtein(seq1, seq2)                 -->     Compute the Levenshtein distance between two given strings (seq1 and seq2) deletion, insertion, substitution
damerau_levenshtein_distance(s1, s2)    -->   Compute the Damerau-Levenshtein distance between two given strings (s1 and s2) deletion, insertion, substitution, transposition
LCSLength(X, Y, m, n)                   -->    Function to find the length of the longest common subsequence of sequences `X[0…m-1]` and `Y[0…n-1]`
show_LCSLength(X, Y, normalise=True)    -->   Caller for Largest Common Subsequence ie LCSLength()
hamming_distance(string1, string2)

'''

def random_nearist_letter(letter: str):
    """ Pick a random letter near given letter, using keyboard layout. Ie fat fingers.

        Note: will return a random printable letter if given letter is not on the keyboard.
    """
    population = keyboard_surrounding_keys.get(letter.lower())
    if population is None:
        print(f"@random_nearist_letter() given letter: '{letter}' is not in lookup, add it.")
        return random_letter()
    return random.choices(population)[0]


def random_letter(*args):
    """ Pick a random printable, with a preference for letters.
    """
    population = string.printable[:-5]
    # This frequency is related to (digits), (lowercase), (uppercase), (punctuation)
    # frequency = ([1]*10) + ([4] * 26) + ([3] * 26) + ([1] * 33)
    frequency = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    letter = random.choices(population, frequency)[0]
    return letter 


def pick_letter(letter: str, _weights=(2,1)):
    """ Pick the letter, use the function 'random_nearist_letter' more often than 'random_letter'

        ## LOGIC - do 'a' function more than b function
        # a = lambda: "a"
        # b = lambda: "b"
        # random.choices([a,b], (2,1))[0]()
    """
    return random.choices([random_nearist_letter, random_letter], _weights)[0](letter)

# based on https://en.wikipedia.org/wiki/Levenshtein_distance
def insertion(l: List[str], index=None):
    """ Insert a random letter
        where 'l' is a list of chars
    """
    if l:
        # l.insert(select_index(l), random_letter())
        l.insert(select_index(l), pick_letter(l[select_index(l)]))
    return l 

# based on https://en.wikipedia.org/wiki/Levenshtein_distance
def deletion(l: List[str], index=None):
    """ Delete a random letter
        where 'l' is a list of chars
    """
    if l:
        l.pop(select_index(l))
    return l

# based on https://en.wikipedia.org/wiki/Levenshtein_distance
def substitution(l: List[str], index=None):
    """ Change a random letter
        where 'l' is a list of chars
    """
    if l:
        # l[select_index(l)] = random_letter()
        l[select_index(l)] = pick_letter(l[select_index(l)])
    return l

# based on https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
def transposition(l: List[str]):
    """ Swap two adjacent letters
        where 'l' is a list of chars
    """
    # ignore strings 0 and 1 length
    if l and len(l) > 1:
        index = select_index(l) 
        front = index + 1
        back = index - 1

        if (len(l) -1 ) == index:
            # Target is at the end of the list
            index_2 = back
        else:
            # default to next index.
            index_2 = front

        temp = l[::1][index]
        l[index] = l[index_2]
        l[index_2] = temp
    return l


def create_misspelling(s: str, edit_distance:int=3, _force:bool=False, _weights:bool=None):
    """ Create misspellings based on levenshtein edit distance 

    Take a string and add 'insertion', 'deletion', 'substitution'. 
    To create a new string with a edit distance less then 3
    
    POSSIPLE EDITS:
    insertion:      cot → coat
    deletion:       coat → cot
    substitution:   coat → cost
    transposition:  cost → cots
    
    ARGS:
    s:              String to do the operations on
    edit_distance:  Max edit distance to use, any number below this.
    _force:         Use edit distance with out any changes
    _weights:       Optional frequency to apply the 4 functions [insertion, deletion, substitution, transposition]
                    ie _weights=[4,1,1,1] --> Do 'insertion' 4/7 of the time and one of the others 3/7 of the time.

    EXMAPLES:
            >>> random.seed(100)
            >>> create_misspelling("hello", 3) #> 'he7l'
            >>> create_misspelling('')  #> ''
            >>> create_misspelling(None)  #> ''
            >>> create_misspelling("random house", 3) #> 'r*ndom hdus'
            >>> create_misspelling("Software Dev", 3) #> 'SofQtwrea Dev'
            >>> create_misspelling("Office worker", 3) #> 'Offtice wor(ker'

   Based Roughly on the levenshtein distance 
    
    """
    if not s:
        return "" 

    if _weights is None:
        _weights = [4,4,4,1]
        # _weights = [1,1,1,1]
        # _weights = [1,1,1,0]
    
    options = [insertion, deletion, substitution, transposition]
    
    "How many edits total"
    if _force is True:
        edits = edit_distance
    else:
        # This changes the edit distance to make sure the max edit distance is not more than 30% of the length
        edit_distance = min(math.ceil(len(s) * 0.3), edit_distance)

        ## bias towards LARGE number of mistakes
        edits = random.choices(range(1, edit_distance + 1), range(1, edit_distance + 1), k=1)[0]
        ## bias towards SMALL number of mistakes
        edits = random.choices(range(1, edit_distance + 1), reversed(range(1, edit_distance + 1)), k=1)[0]
    
    print("EDITS: ", edits) #> EDITS:  2
    
    "What type of edit(s) to do"
    funcs = random.choices(options, _weights, k=edits)
    print("FUNCS: ", [e.__name__ for e in funcs]) #> FUNCS:  [substitution, deletion]

    "Apply each function in series to the input string"
    result = reduce(lambda v, f: f(v), funcs, list(s))
    return ''.join(result)




@nb.njit
def levenshtein(seq1, seq2):
    """
    Compute the Levenshtein distance between two given
    strings (seq1 and seq2) deletion, insertion, substitution
    
    Different from levenshtein because of does not check for transposition
    """
    # https://en.wikipedia.org/wiki/Levenshtein_distance
    # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print(matrix)
    return (matrix[size_x - 1, size_y - 1]) 




# @nb.njit
# def damerau_levenshtein_distance(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1+1, lenstr2+1))
#     for i in range(-1,lenstr1+1):
#         matrix[i,0] = i
#     for j in range(-1,lenstr2+1):
#         matrix[0,j] = j
#     # print(matrix)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             matrix[i,j] = min(
#                            matrix[i-1,j] + 1, # deletion
#                            matrix[i,j-1] + 1, # insertion
#                            matrix[i-1,j-1] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 matrix[i,j] = min (matrix[i,j], matrix[i-2,j-2] + cost) # transposition

#     return matrix[lenstr1-1,lenstr2-1]


@nb.njit
def damerau_levenshtein_distance(seq1, seq2):
    """ Compute the damerau_levenshtein distance between two given
    strings (seq1 and seq2) deletion, insertion, substitution
    
    Different from levenshtein because of does not check for transposition
    """
    # https://en.wikipedia.org/wiki/Levenshtein_distance
    # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    # print(matrix)
    for x in range(1, size_x):
        for y in range(1, size_y):
            cost = int(not (seq1[x-1] == seq2[y-1]))
            matrix[x,y] = min(
                matrix[x-1, y] + 1,
                matrix[x-1, y-1] + cost,
                matrix[x, y-1] + 1
            )
            if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
                matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + cost) #, matrix[x-2,y-2] + 0) # transposition 
            # if seq1[x-1] == seq2[y-1]:
            #     matrix[x,y] = min(
            #         matrix[x-1, y] + 1,
            #         matrix[x-1, y-1],
            #         matrix[x, y-1] + 1
            #     )
            #     if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
            #         matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 0) #, matrix[x-2,y-2] + 0) # transposition
            # else:
            #     matrix[x,y] = min(
            #         matrix[x-1,y] + 1,
            #         matrix[x-1,y-1] + 1,
            #         matrix[x,y-1] + 1
            #     )
            #     if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
            #         matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 1) #, matrix[x-2,y-2] + 0) # transposition
    # print(matrix)
    return (matrix[size_x - 1, size_y - 1])

@nb.njit
def hamming_distance(string1, string2):
	"""
		https://en.wikipedia.org/wiki/Hamming_distance

		"karolin" and "kathrin" is 3.
		"karolin" and "kerstin" is 3.
		"kathrin" and "kerstin" is 4.
		0000 and 1111 is 4.
		2173896 and 2233796 is 3.

		>>> hamming_distance("karolin","kathrin")
		3
	"""
    dist_counter = 0
    for n in range(len(string1)):
            if string1[n] != string2[n]:
                    dist_counter += 1
    return dist_counter

@nb.njit
def LCSLength(X, Y, m, n):
    """ Function to find the length of the longest common subsequence of
        sequences `X[0…m-1]` and `Y[0…n-1]`

    >>> X = 'ABCBDAB'
    >>> Y = 'BDCABA'
    >>> LCSLength(X, Y, len(X), len(Y)))
    4

    """
    # return if the end of either sequence is reached
    if m == 0 or n == 0:
        return 0
    # if the last character of `X` and `Y` matches
    if X[m - 1] == Y[n - 1]:
        return LCSLength(X, Y, m - 1, n - 1) + 1
 
    # otherwise, if the last character of `X` and `Y` does not match
    return max(LCSLength(X, Y, m, n - 1), LCSLength(X, Y, m - 1, n))


@nb.njit
def show_LCSLength(X, Y, normalise=True):
    """ Caller for Largest Common Subsequence ie LCSLength()

        https://en.wikipedia.org/wiki/Longest_common_subsequence_problem

        Return is normalized between 0 to 1 
        of what percent of string X is substring Y
    """
    m = len(X)
    n = len(Y)
    result = LCSLength(X, Y, m=m, n=n)
    if normalise:
        ## normalised option one
        # length = max(len(X), len(Y))
        # output = (length - result) / length
        ## normalised option two
        output = result / (m + n)
    else:
        output = result
    return output
    



# List of keys near each key on a keyboard, where the first index is it's self.
keyboard_surrounding_keys = {
    # Number line
    "`": "`1a",
    "1": "1`2wq",
    "2": "213ewq",
    "3": "324rew",
    "4": "435tre",
    "5": "546ytr",
    "6": "657uyt",
    "7": "768iuy",
    "8": "879oiu",
    "9": "980poi",
    "0": "09-[po",
    "-": "-0=][p",
    "=": "=-\][",
    # First line
    "q": "q `12wsa",
    "w": "wq123edsa",
    "e": "ew234rfds",
    "r": "re345tgfd",
    "t": "tr456yhgf",
    "y": "yt567ujhg",
    "u": "uy678ikjh",
    "i": "iu789olkj",
    "o": "oi890p;lk",
    "p": "po90-[';l",
    "[": "[p0-=]';",
    "]": "][-=\'",
    # Second line
    "a": "aqwsxz",
    "s": "saqwedcxz",
    "d": "dswerfvcx",
    "f": "fdertgbvc",
    "g": "gfrtyhnbv",
    "h": "hgtyujmnb",
    "j": "jhyuik,mn",
    "k": "kjuiol.,m",
    "l": "lkiop;/.,",
    ";": ";lop['/.",
    "'": "';p[]/.",
    # Thrid Line
    "z": "zasx",
    "x": "xzasdc ",
    "c": "cxsdfv ",
    "v": "vcdfgb ",
    "b": "bvfghn ",
    "n": "nbghjm ",
    "m": "mnhjk, ",
    ",": ",mjkl. ",
    ".": ".,kl;/",
    "/": "/.l;'",
    # Shift mapping
    '!': '!`2wq',
    '@': '@13ewq',
    '#': '#24rew',
    '$': '$35tre',
    '%': '%46ytr',
    '^': '^57uyt',
    '&': '&68iuy',
    '*': '*79oiu',
    '(': '(80poi',
    ')': ')9-[po',
    '_': '_0=][p',
    '+': '+-\\][',
    '{': "{p0-=]';",
    '}': "}[-='",
    ':': ":lop['/.",
    '"': '";p[]/.',
    '<': '<mjkl. ',
    '>': '>,kl;/',
    '?': "?.l;'",
    '~': "~1q"
}


# shift_mapping = {
#     '`': '~',
#     '1': '!',
#     '2': '@',
#     '3': '#',
#     '4': '$',
#     '5': '%',
#     '6': '^',
#     '7': '&',
#     '8': '*',
#     '9': '(',
#     '0': ')',
#     '-': '_',
#     '=': '+',
#     '[': '{',
#     ']': '}',
#     '\\': '|',
#     ';': ':',
#     "'": '"',
#     ',': '<',
#     '.': '>',
#     '/': '?'
# }
 
# X = 'ABCBDAB'
# Y = 'BDCABA'

# print('The length of the LCS is', LCSLength(X, Y, len(X), len(Y)))


def test_create_misspelling():
    """ TEST: 

    # python3 -m pytest create_misspelling.py
    """

    words = ["Software Dev", "House wife", "Fisherman", "Doctor", "Supermarket", "Teacher"]
    edits = 3

    for word in words:
        bad_word_levenshtein = create_misspelling(word, edits, _force=True, _weights=[1,1,1,0])
        bad_word = create_misspelling(word, edits, _force=True)
        print(bad_word)
        print(levenshtein(word, bad_word_levenshtein))
        print(damerau_levenshtein_distance(word, bad_word))

        assert levenshtein(word, bad_word_levenshtein) <= edits
        assert damerau_levenshtein_distance(word, bad_word) <= edits



test_words = [
    "sausage",
    "blubber",
    "pencil",
    "cloud",
    "moon",
    "water",
    "computer",
    "school",
    "network",
    "hammer",
    "walking",
    "violently",
    "mediocre",
    "literature",
    "chair",
    "two",
    "window",
    "cords",
    "musical",
    "zebra",
    "xylophone",
    "penguin",
    "home",
    "dog",
    "final",
    "ink",
    "teacher",
    "fun",
    "website",
    "banana",
    "uncle",
    "softly",
    "mega",
    "ten",
    "awesome",
    "attach",
    "blue",
    "Internet",
    "bottle",
    "tight",
    "zone",
    "tomato",
    "prison",
    "hydro",
    "cleaning",
    "television",
    "send",
    "frog",
    "cup",
    "book",
    "zooming",
    "falling",
    "evil",
    "gamer",
    "lid",
    "juice",
    "monster",
    "captain",
    "bonding",
    "loudly",
    "thudding",
    "guitar",
    "shaving",
    "hair",
    "soccer",
    "water",
    "racket",
    "table",
    "late",
    "media",
    "desktop",
    "flipper",
    "club",
    "flying",
    "smooth",
    "monster",
    "purple",
    "guardian",
    "bold",
    "hypersonic",
    "presentation",
    "world",
    "national",
    "comment",
    "element",
    "magic",
    "lion",
    "sand",
    "crust",
    "toast",
    "jam",
    "hunter",
    "forest",
    "foraging",
    "silently",
    "awesomeness",
    "joshing",
    "pony",
]

def more_tests():

    from create_misspelling import damerau_levenshtein_distance
    from create_misspelling import create_misspelling
    from create_misspelling import test_words
    from create_misspelling import levenshtein

    word = "hunter"
    mispelt = create_misspelling(word, _weights=[1,1,1,1])
    print(mispelt)
    damerau_levenshtein_distance(word, mispelt)

    # create_misspelling(word, edits, _force=True, _weights=[1,1,1,0])

    # test_words


    # min(len(word) // 3, 3)
def more_tests():

    word = random.choices(test_words)[0]
    print(word)
    mispelt = create_misspelling(word, _weights=[1,1,1,1])
    print(mispelt)


    x = [(normalise(damerau_levenshtein_distance(e, mispelt), max(len(e), len(mispelt))), e) for e in test_words]
    xx = [(normalise(levenshtein(e, mispelt), max(len(e), len(mispelt))), e) for e in test_words]


    sorted(x, key=lambda a: a[0])[-10:]
    sorted(xx, key=lambda a: a[0])[-10:]


    word = random.choices(test_words)[0]
    print(word)
    mispelt = create_misspelling(word, _weights=[1,1,1,1])
    print(mispelt)


    x = [(damerau_levenshtein_distance(e, mispelt), e) for e in test_words]
    xx = [(levenshtein(e, mispelt), e) for e in test_words]

    sorted(x, key=lambda a: a[0])[:10]
    sorted(xx, key=lambda a: a[0])[:10]



# @nb.njit
def normalise(distance, max_length):
    # distance = float(matrix[l2][l1])
    # max_length = s1 if s1 >= s2 else s2
    # result = 1.0-distance/max(s1,s2)
    result = 1.0-distance/max_length
    # result = 1.0-np.divide(distance, max_length)
    return result


# https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
# https://datascience.stackexchange.com/questions/60019/damerau-levenshtein-edit-distance-in-python

# """
# My damerau_levenshtein_distance formula is return seemingly the wrong result, where (2.0, 'ten'), (2.0, 'cup') for inputs like 'soxcer' (soccer)

# Looks like a mistake.

# """

# def damerau_levenshtein_distance2(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1+1, lenstr2+1))
#     for i in range(-1,lenstr1+1):
#         matrix[i,0] = i
#     for j in range(-1,lenstr2+1):
#         matrix[0,j] = j
#     print(matrix)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             matrix[i,j] = min(
#                            matrix[i-1,j] + 1, # deletion
#                            matrix[i,j-1] + 1, # insertion
#                            matrix[i-1,j-1] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 matrix[i,j] = min (matrix[i,j], matrix[i-2,j-2] + cost) # transposition
#     print(matrix)
#     return matrix[lenstr1-1,lenstr2-1]


# def levenshtein(seq1, seq2):
#     """
#     Compute the Levenshtein distance between two given
#     strings (seq1 and seq2) deletion, insertion, substitution
    
#     Different from levenshtein because of does not check for transposition
#     """
#     # https://en.wikipedia.org/wiki/Levenshtein_distance
#     # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
#     size_x = len(seq1) + 1
#     size_y = len(seq2) + 1
#     matrix = np.zeros ((size_x, size_y))
#     for x in range(size_x):
#         matrix[x, 0] = x
#     for y in range(size_y):
#         matrix[0, y] = y
#     print(matrix)
#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             if seq1[x-1] == seq2[y-1]:
#                 matrix[x,y] = min(
#                     matrix[x-1, y] + 1,
#                     matrix[x-1, y-1],
#                     matrix[x, y-1] + 1
#                 )
#                 if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
#                     matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 0) #, matrix[x-2,y-2] + 0) # transposition
#             else:
#                 matrix[x,y] = min(
#                     matrix[x-1,y] + 1,
#                     matrix[x-1,y-1] + 1,
#                     matrix[x,y-1] + 1
#                 )
#                 if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
#                     matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 1) #, matrix[x-2,y-2] + 0) # transposition
#     print(matrix)
#     return (matrix[size_x - 1, size_y - 1])


# levenshtein('ten', 'soxcer') 5
# levenshtein('soccer', 'soxcer') 1
# levenshtein('soccer', 'soccre') 1
# levenshtein('soccer', 'osccre') 2
# levenshtein('soccer', 'soccceer') 2 
# levenshtein('soccer', 's') 5 
    
# @nb.njit
# def levenshtein(seq1, seq2):
#     """
#     Compute the Levenshtein distance between two given
#     strings (seq1 and seq2) deletion, insertion, substitution
    
#     Different from levenshtein because of does not check for transposition
#     """
#     # https://en.wikipedia.org/wiki/Levenshtein_distance
#     # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
#     size_x = len(seq1) + 1
#     size_y = len(seq2) + 1
#     matrix = np.zeros ((size_x, size_y))
#     for x in range(size_x):
#         matrix[x, 0] = x
#     for y in range(size_y):
#         matrix[0, y] = y
#     print(matrix)
#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             cost = int(not (seq1[x-1] == seq2[y-1]))
#             matrix[x,y] = min(
#                 matrix[x-1, y] + 1,
#                 matrix[x-1, y-1] + cost,
#                 matrix[x, y-1] + 1
#             )
#             if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
#                 matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + cost) #, matrix[x-2,y-2] + 0) # transposition 
#             # if seq1[x-1] == seq2[y-1]:
#             #     matrix[x,y] = min(
#             #         matrix[x-1, y] + 1,
#             #         matrix[x-1, y-1],
#             #         matrix[x, y-1] + 1
#             #     )
#             #     if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
#             #         matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 0) #, matrix[x-2,y-2] + 0) # transposition
#             # else:
#             #     matrix[x,y] = min(
#             #         matrix[x-1,y] + 1,
#             #         matrix[x-1,y-1] + 1,
#             #         matrix[x,y-1] + 1
#             #     )
#             #     if x and y and seq1[x-1]==seq2[y-2] and seq1[x-2] == seq2[y-1]:
#             #         matrix[x,y] = min(matrix[x,y], matrix[x-2,y-2] + 1) #, matrix[x-2,y-2] + 0) # transposition
#     print(matrix)
#     return (matrix[size_x - 1, size_y - 1])


# def damerau_levenshtein_distance2(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1+1, lenstr2+1))
#     for i in range(-1,lenstr1+1):
#         matrix[i,0] = i
#     for j in range(-1,lenstr2+1):
#         matrix[0,j] = j
#     print(matrix)
#     for i in range(-1, lenstr1):
#         for j in range(-1, lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             matrix[i,j] = min(
#                            matrix[i-1,j] + 1, # deletion
#                            matrix[i,j-1] + 1, # insertion
#                            matrix[i-1,j-1] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 matrix[i,j] = min (matrix[i,j], matrix[i-2,j-2] + cost) # transposition
#     print(matrix)
#     return matrix[lenstr1-1,lenstr2-1]
# damerau_levenshtein_distance2('ten', 'soxcer')
# x = [(damerau_levenshtein_distance2(e, mispelt), e) for e in test_words]
# sorted(x, key=lambda a: a[0])[:10]


# def damerau_levenshtein_distance2(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1, lenstr2))
#     # for i in range(-1,lenstr1+1): matrix[i,-1] = i
#     # for j in range(-1,lenstr2+1): matrix[-1,j] = j
#     print(matrix)
#     # for i in range(-1,lenstr1+1):
#     #     d[(i,-1)] = i+1
#     # for j in range(-1,lenstr2+1):
#     #     d[(-1,j)] = j+1
#     # print(d)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             print(s1[i], s2[j])
#             if s1[i] == s2[j]:
#                 cost = 0
#                 matrix[(i,j)] = cost
#             else:
#                 cost = 1
#                 matrix[(i,j)] = cost
#             # matrix[(i,j)] = min(
#             #                matrix[(i-1,j)] + 1, # deletion
#             #                matrix[(i,j-1)] + 1, # insertion
#             #                matrix[(i-1,j-1)] + cost, # substitution
#             #                   )
#             # if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#             #     matrix[i,j] = min(matrix[i,j], matrix[i-2,j-2] + cost) # transposition
#     print(matrix)
#     print(matrix[(lenstr1-1,lenstr2-1)])
#     return matrix[lenstr1-1,lenstr2-1]


# damerau_levenshtein_distance2('ten', 'soxcer')
# damerau_levenshtein_distance2('sen', 'soxcer')
# damerau_levenshtein_distance2('soccer', 'soxcer')


# def damerau_levenshtein_distance2(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1+1, lenstr2+1))
#     for i in range(-1,lenstr1+1): matrix[i,-1] = i
#     for j in range(-1,lenstr2+1): matrix[-1,j] = j
#     print(matrix)
#     # for i in range(-1,lenstr1+1):
#     #     d[(i,-1)] = i+1
#     # for j in range(-1,lenstr2+1):
#     #     d[(-1,j)] = j+1
#     # print(d)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             matrix[i,j] = min(
#                            matrix[i-1,j] + 1, # deletion
#                            matrix[i,j-1] + 1, # insertion
#                            matrix[i-1,j-1] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 matrix[i,j] = min (matrix[i,j], matrix[i-2,j-2] + cost) # transposition
#     print(matrix)
#     return matrix[lenstr1-1,lenstr2-1]


# damerau_levenshtein_distance2('ten', 'soxcer')
# x = [(damerau_levenshtein_distance2(e, mispelt), e) for e in test_words]
# sorted(x, key=lambda a: a[0])[:10]


# def damerau_levenshtein_distance2(s1, s2):
#     """ Compute the Damerau-Levenshtein distance between two given
#         strings (s1 and s2) deletion, insertion, substitution, transposition
    
#         Different from levenshtein because of added transposition check
#     """
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     matrix = np.zeros((lenstr1+1, lenstr2+1))
#     for i in range(-1,lenstr1+1):
#         matrix[i,0] = i
#     for j in range(-1,lenstr2+1):
#         matrix[0,j] = j
#     print(matrix)
#     for i in range(lenstr1):
#         i = i + 1
#         for j in range(lenstr2):
#             j = j + 1
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             matrix[i,j] = min(
#                            matrix[i,j] + 1, # deletion
#                            matrix[i,j] + 1, # insertion
#                            matrix[i,j] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j] and s1[i] == s2[j]:
#                 matrix[i,j] = min (matrix[i,j], matrix[i-1,j-1] + cost) # transposition
#     print(matrix)
#     return matrix[lenstr1-1,lenstr2-1]


# def damerau_levenshtein3(s1, s2):
#     l1 = len(s1)
#     l2 = len(s2)
#     matrix = [list(range(l1 + 1))] * (l2 + 1)
#     for zz in list(range(l2 + 1)):
#       matrix[zz] = list(range(zz,zz + l1 + 1))
#     print(matrix)
#     for zz in list(range(0,l2)):
#       for sz in list(range(0,l1)):
#         if s1[sz] == s2[zz]:
#           matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
#         else:
#           matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
#     print(matrix)
#     distance = float(matrix[l2][l1])
#     result = 1.0-distance/max(l1,l2)
#     return result

# # @nb.njit
# # Correct, but uses a dict, so can't njit.
# def damerau_levenshtein_distance4(s1, s2):
#     d = {}
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     for i in range(-1,lenstr1+1):
#         d[(i,-1)] = i+1
#     for j in range(-1,lenstr2+1):
#         d[(-1,j)] = j+1
#     print(d)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             d[(i,j)] = min(
#                            d[(i-1,j)] + 1, # deletion
#                            d[(i,j-1)] + 1, # insertion
#                            d[(i-1,j-1)] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
#     print(d)
#     return d[lenstr1-1,lenstr2-1]


# def damerau_levenshtein_distance5(s1, s2):
#     d = {}
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     for i in range(-1,lenstr1+1):
#         d[(i,-1)] = i+1
#     for j in range(-1,lenstr2+1):
#         d[(-1,j)] = j+1
#     print(d)
#     for i in range(lenstr1):
#         for j in range(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             d[(i,j)] = min(
#                            d[(i-1,j)] + 1, # deletion
#                            d[(i,j-1)] + 1, # insertion
#                            d[(i-1,j-1)] + cost, # substitution
#                           )
#             if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
#                 d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
#     print(d)
#     return d[lenstr1-1,lenstr2-1]

# """
# >>> damerau_levenshtein_distance4('soccer', 'soxcer')
# >>> damerau_levenshtein3('soccer', 'soxcer')
# >>> damerau_levenshtein_distance2('soccer', 'soxcer')
# [[0. 1. 2. 3. 4. 5. 6.]
#  [1. 0. 0. 0. 0. 0. 0.]
#  [2. 0. 0. 0. 0. 0. 0.]
#  [3. 0. 0. 0. 0. 0. 0.]
#  [4. 0. 0. 0. 0. 0. 0.]
#  [5. 0. 0. 0. 0. 0. 0.]
#  [6. 0. 0. 0. 0. 0. 0.]]
# [[0. 1. 2. 3. 4. 5. 6.]
#  [1. 0. 1. 1. 1. 1. 1.]
#  [2. 1. 0. 1. 2. 2. 2.]
#  [3. 2. 1. 1. 1. 2. 2.]
#  [4. 3. 2. 1. 0. 2. 2.]
#  [5. 3. 3. 2. 2. 1. 1.]
#  [6. 2. 4. 3. 2. 1. 0.]]
# 1.0
# >>> damerau_levenshtein_distance2('ten', 'soxcer')
# [[0. 1. 2. 3. 4. 5. 6.]
#  [1. 0. 0. 0. 0. 0. 0.]
#  [2. 0. 0. 0. 0. 0. 0.]
#  [3. 0. 0. 0. 0. 0. 0.]]
# 2.0
# """
