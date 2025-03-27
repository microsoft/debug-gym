## Mini Nightmare
This is a selection of human generated buggy code that give conventional code repairing system some mini panic attack. 

### 1.config
Dynamic attribute access and management.

Hint:

- `_cache = {}` needs to be instance-level, not shared;
- Clear cache when updating values in set()
```
    if name in self._cache:
        del self._cache[name]
```

### 2.counter
Race condition in multi threading.

Hint:

Put everything in `increment()` in a `threading.Lock()`.

### 3.grader
Complex data structure.

Hint:

Follow instruction that "if a stuent has incomplete grades, they are not considered."

### 4.pandas_dataframe
Unknown data structure.

Hint:

There's no `Price` column, the correct key is `fare`.

### 5.patcher
Boundary check.

Hint:

Line numbers are 1-based. Handle cases where `head <= 0 or tail <= 0` or `head > tail`.

### 6.purr
Buggy condition.

Hint:
```
     if self.hunger > 10:
         ...
     elif self.hunger > 20:
         ...
```

### 7.scienfitic calculator
String split.

Hint:

L6: `action_str` vs `action`

### 8.shopping_cart
Logic bug.

Hint:

`apply_discount()` does not apply discount to future items.

### 9.sum_tree
Loop in data structure

Hint:

Catch loop in tree traversal.

### 10.tomorrow_date
Condition coverage.

Hint:

`Feb 29 + 1` in leap years.
