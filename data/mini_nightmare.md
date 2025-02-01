## Mini Nightmare
This is a selection of human generated buggy code that give conventional code repairing system some mini panic attack. 

### config
Dynamic attribute access and management.

Hint:
1. `_cache = {}` needs to be instance-level, not shared;
2. Clear cache when updating values in set()
```
    if name in self._cache:
        del self._cache[name]
```

### counter
Race condition in multi threading.

Hint:
Put everything in `increment()` in a `threading.Lock()`.

### grader
Complex data structure.

Hint:
Follow instruction that "if a stuent has incomplete grades, they are not considered."

### pandas_dataframe
Unknown data structure.

Hint:
There's no `Price` column, the correct key is `fare`.

### patcher
Boundary check.

Hint:
Line numbers are 1-based. Handle cases where `head <= 0 or tail <= 0` or `head > tail`.

### purr
Buggy condition.

Hint:
```
     if self.hunger > 10:
         ...
     elif self.hunger > 20:
         ...
```

### pytorch
Sequence of bugs.

Hint:
1. L21: `=` vs `==`;
2. L66: `cat` needs to happen on `dim=1`;
3. L106: the right key is `attention_mask`;
4. L98: wrong shape, see `self.max_length`.

### scienfitic calculator
String split.

Hint:
L6: `action_str` vs `action`

### shopping_cart
Logic bug.

Hint:
`apply_discount()` does not apply discount to future items.

### sum_tree
Loop in data structure

Hint:
Catch loop in tree traversal.

### tomorrow_date
Condition coverage.

Hint:
`Feb 29 + 1` in leap years.
