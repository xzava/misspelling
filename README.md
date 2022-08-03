# Misspelling

Python library to generate likely misspellings of words using the [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)

The `damerau_levenshtein_distance` can also be used to correct the misspelling.

`damerau_levenshtein_distance` and `levenshtein_distance` are both super fast as they are jit compiled with `numba`

The idea was to correct Speech to Text and OCR output.

- Makes a good fuzzy finder.

- Factors in keyboard layout to make physically close keys appear more often, simulating fat fingers.

## create_misspelling
```python
>>> from create_misspelling import create_misspelling
>>> create_misspelling("flying")
FUNCS:  ['transposition']
"fyling"

>>> create_misspelling("smooth")
FUNCS:  ['insertion']
"psmooth"

>>> create_misspelling("monster")
FUNCS:  ['substitution']
"mons6er"

>>> create_misspelling("purple")
FUNCS:  ['deletion']
"urple"

>>> create_misspelling("guardian")
FUNCS:  ['substitution', 'deletion', 'transposition']
"gaurdia"

>>> create_misspelling("bold")
FUNCS:  ['substitution', 'insertion']
"holud"

>>> create_misspelling("hypersonic")
FUNCS:  ['deletion']
"hyersonic"

>>> create_misspelling("presentation")
FUNCS:  ['deletion']
"pesentation"

>>> create_misspelling("world")
FUNCS:  ['substitution']
"woeld"

>>> create_misspelling("national")
FUNCS:  ['insertion', 'deletion']
"naiFonal"

>>> create_misspelling("comment")
FUNCS:  ['deletion']
"commet"

>>> create_misspelling("element")
FUNCS:  ['transposition', 'substitution']
"flemetn"

>>> create_misspelling("magic")
FUNCS:  ['insertion']
"maqgic"

>>> create_misspelling("lion")
FUNCS:  ['deletion', 'insertion']
"Qlin"

>>> create_misspelling("sand")
FUNCS:  ['insertion', 'deletion']
"snad"

>>> create_misspelling("crust")
FUNCS:  ['substitution', 'substitution']
"crbft"

>>> create_misspelling("toast")
FUNCS:  ['substitution']
"toayt"

>>> create_misspelling("jam")
FUNCS:  ['deletion']
"am"

>>> create_misspelling("hunter")
FUNCS:  ['insertion']
"jhunter"

>>> create_misspelling("forest")
FUNCS:  ['deletion']
"frest"

>>> create_misspelling("foraging")
FUNCS:  ['transposition']
"forgaing"

>>> create_misspelling("silently")
FUNCS:  ['deletion', 'transposition']
"siletny"

>>> create_misspelling("awesomeness")
FUNCS:  ['deletion', 'deletion']
"awesoenes"

>>> create_misspelling("joshing")
FUNCS:  ['deletion', 'transposition']
"johign"

>>> create_misspelling("pony")
FUNCS:  ['insertion']
"pohny"
```

## Correct spelling

```python
from create_misspelling import damerau_levenshtein_distance
from create_misspelling import create_misspelling
from create_misspelling import test_words
from create_misspelling import levenshtein
import random


>>> word = random.choices(test_words)[0]
>>> print(word)
mediocre
>>> mispelt = create_misspelling(word, _weights=[1,1,1,1])
EDITS:  2
FUNCS:  ['insertion', 'deletion']
>>> print(mispelt)
meiocrQe
>>> 
>>> x = [(damerau_levenshtein_distance(e, mispelt), e) for e in test_words]
>>> xx = [(levenshtein(e, mispelt), e) for e in test_words]
>>> 
>>> sorted(x, key=lambda a: a[0])[:10]
[(2.0, 'mediocre'), (5.0, 'chair'), (5.0, 'juice'), (6.0, 'pencil'), (6.0, 'moon'), (6.0, 'network'), (6.0, 'cords'), (6.0, 'musical'), (6.0, 'zebra'), (6.0, 'home')]
>>> sorted(xx, key=lambda a: a[0])[:10]
[(2.0, 'mediocre'), (5.0, 'juice'), (6.0, 'pencil'), (6.0, 'moon'), (6.0, 'network'), (6.0, 'cords'), (6.0, 'musical'), (6.0, 'zebra'), (6.0, 'home'), (6.0, 'teacher')]

```

`damerau_levenshtein_distance` is often better as it factors in transposition, which is a common typing mistake.


Todo: Might be able to vectorise the spell check and apply `damerau_levenshtein_distance` to a numpy array.

Or something like this


```python
@nb.njit
def spell_check(dictionary):
    ...

```

Todo: Use the built in linux `words` located in `/usr/share/dict/words` or `/usr/dict/words`

Note: https://www.geeksforgeeks.org/how-to-read-specific-lines-from-a-file-in-python/


