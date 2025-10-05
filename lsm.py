from collections import defaultdict

def normalize_value(val):

    # string to int or float
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val)

    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass

    return s


def pretty_print_index_stats(stats):
    # info returned by debug_index_stats()
    for col, info in stats.items():
        print(f"Index '{col}':")
        print(f"  column      : {info.get('column')}")
        print(f"  levels      : {info.get('levels')}")
        print(f"  level_counts: {info.get('level_counts')}")
        print(f"  level_caps  : {info.get('level_caps')}")
    print()

def pretty_print_agg(agg):
    print('Aggregates:', ', '.join(f"{k}: {v}" for k, v in agg.items()), '\n')

class FactTable:
    
    def __init__(self, schema, pk = 'ID'):

        self.schema = list(schema)

        if pk not in self.schema:
            raise ValueError(f"ERROR: Primary key '{pk}' not in schema {self.schema}")
        self.pk = pk
        self.rows = {}

    def insert_row(self, row):

        if self.pk not in row:
            raise KeyError(f"ERROR: Row doesn't contain primary key '{self.pk}'")
        key = row[self.pk]

        if key in self.rows:
            raise KeyError(f"ERROR: Duplicate primary key '{key}'")

        normalized = {c: normalize_value(row.get(c)) for c in self.schema}
        self.rows[key] = normalized

    def delete_row(self, key):

        # won't get returned during lookup
        if key in self.rows:
            del self.rows[key]

    def get_row(self, key):
        return self.rows.get(key)

    def scan(self):
        return list(self.rows.values())

class LSMIndex:

    def __init__(self, column, deleted_ids = None, l0_capacity = 1000, growth = 3, max_rows = 13000):
        self.column = column

        # 1 level = a dictionary mapping value -> set of PKs
        self.levels = []

        # counts of PK entries per level (to know when to merge)
        self.level_counts = []

        # keep a ref to the tombstone set (if exists)
        self.deleted_ids = deleted_ids if deleted_ids is not None else set()

        # compute capacity per level, must not exceed max_rows
        caps = []
        cap = l0_capacity
        # max_rows * 2 =  heuristic to ensure enough levels are created
        while sum(caps) < max_rows * 2:
            caps.append(cap)
            cap *= growth
        self.level_caps = caps

        # init data structure and counts for every level
        for _ in caps:
            self.levels.append(defaultdict(set))
            self.level_counts.append(0)

    # current cnt of PK entries in given level
    def _level_count(self, lvl):
        return self.level_counts[lvl]

    # insert mapping val -> row_ID into level
    def insert(self, value, pk):
        val = normalize_value(value)

        # only inserting if unique
        if pk not in self.levels[0][val]:
            self.levels[0][val].add(pk)
            self.level_counts[0] += 1

        # check capacity - merge might be required
        if self.level_counts[0] > self.level_caps[0]:
            self._merge_level(0)

    # remove mapping of val->row from each level in which it appears
    def delete(self, value, pk):
        val = normalize_value(value)

        for lvl in range(len(self.levels)):
            s = self.levels[lvl].get(val)

            # if set containing pk exists, remove it and update cnt
            if s and pk in s:
                s.remove(pk)
                self.level_counts[lvl] -= 1
                if len(s) == 0:
                    # empty value mapping, can be removed
                    del self.levels[lvl][val]

    # merge level with level + 1, can be recursive if level + 1 overflows
    def _merge_level(self, lvl):

        # merging for every value (union sets of row IDs)
        
        # if required level doesn't exist it needs to be created
        if lvl + 1 >= len(self.levels):
            self.levels.append(defaultdict(set))
            self.level_counts.append(0)
            # extend caps accordingly
            self.level_caps.append(self.level_caps[-1] * 3)

        src = self.levels[lvl]
        dst = self.levels[lvl + 1]

        # merge = for each value from source, create union of PKs and put into destination
        for val, ids in list(src.items()):

            # leaving out PKs that were logically deleted to avoid resurrecting deleted rows
            if self.deleted_ids:
                valid_ids = {i for i in ids if i not in self.deleted_ids}
            else:
                valid_ids = set(ids)

            # if there are no valid IDs for this value, skip adding
            if not valid_ids:
                continue

            # number of PKs present before unioning (used for updating counts)
            before = len(dst.get(val, set()))

            # union valid IDs into destination level's set for this value
            dst[val].update(valid_ids)

            # ow many new PKs were added, update destination level count
            added = len(dst[val]) - before
            self.level_counts[lvl + 1] += added

        # clear source level
        self.level_counts[lvl] = 0
        self.levels[lvl].clear()

        # recursive merge if destination exceeded capacity
        if self.level_counts[lvl + 1] > self.level_caps[lvl + 1]:
            self._merge_level(lvl + 1)

    # lookup across all lvls for a value (return union of PKs)
    def lookup_all_levels(self, value):
        val = normalize_value(value)
        result = set()

        # IDs from every level set for this value
        for lvl in range(len(self.levels)):
            s = self.levels[lvl].get(val)
            if s:
                result.update(s)

        # exclude tombstone IDs
        if self.deleted_ids:
            result = {i for i in result if i not in self.deleted_ids}
        return result

    def debug_stats(self):
        return {
            "column": self.column,
            "levels": len(self.levels),
            "level_counts": list(self.level_counts),
            "level_caps": list(self.level_caps),
        }

class LSMSystem:

    def __init__(self, schema, indexed_cols, pk = "ID"):
        self.deleted_ids = set()

        self.table = FactTable(schema, pk=pk)

        # LSMIndex instance for each column listed as indexed
        self.indexes = {}
        self.indexed_cols = set(indexed_cols)

        for col in indexed_cols:
            if col not in schema:
                raise ValueError(f"Indexed column '{col}' not in schema")

            # pass deleted_ids so index merges filter tombstoned PKs
            self.indexes[col] = LSMIndex(col, deleted_ids = self.deleted_ids)

    def insert(self, row):

        pk = row.get(self.table.pk)
        if pk is None:
            raise KeyError("Row must contain primary key")

        # If the PK was deleted (is now in tombstone) remove the tombstone
        if pk in self.deleted_ids:
            self.deleted_ids.remove(pk)

        # can now insert the normalized row into in-memory table
        self.table.insert_row(row)

        # for each indexed column extract the column value and insert the mapping into that LSMIndex
        for col in self.indexed_cols:
            self.indexes[col].insert(row.get(col), pk)

    # logical deletion of row based on PK (removing from table, tombstoning)
    def delete(self, pk):

        row = self.table.get_row(pk)
        if not row:
            return
        # remove row from physical table storage
        self.table.delete_row(pk)

        # record PK as logically deleted
        self.deleted_ids.add(pk)

        # try to remove PK from per-column index maps for known values
        # => best-effort cleanup, keeps index levels small
        for col in self.indexed_cols:
            self.indexes[col].delete(row.get(col), pk)

    # return set of matching PKs for column
    def _eval_indexed_condition(self, col, value):
        idx = self.indexes.get(col)
        if not idx:
            return set()
        ids = idx.lookup_all_levels(value)

        # exclude deleted
        ids = {i for i in ids if i in self.table.rows}
        return ids

    # evaluate predicates (equality only)
    def _full_scan_condition(self, conditions, logic_ops):
        # conditions: list of tuples (column, '=', value)
        # logic_ops length = len(conditions)-1, combos left to right
        def row_satisfies(row, cond):
            col, op, val = cond
            if op != '=':
                raise NotImplementedError("ERROR: Only equality is supported for a full scan")
            return normalize_value(row.get(col)) == normalize_value(val)

        result_ids = None
        # evaluate left to right
        for i, cond in enumerate(conditions):
            matching = {pk for pk, r in self.table.rows.items() if row_satisfies(r, cond)}
            if result_ids is None:
                result_ids = matching
            else:
                op = logic_ops[i - 1].upper()
                if op == 'AND':
                    result_ids = result_ids.intersection(matching)
                elif op == 'OR':
                    result_ids = result_ids.union(matching)
                else:
                    raise ValueError(f"ERROR: Unknown logic op '{op}'")
        if result_ids is None:
            return set()

        # exclude deleted rows
        return {i for i in result_ids if i not in self.deleted_ids and i in self.table.rows}

    def search(self, conditions, logic_ops, use_index = True):
        if use_index:

            indexed_preds = []
            nonindexed_preds = []

            for cond in conditions:
                col, op, val = cond
                if col in self.indexed_cols and op == '=':
                    indexed_preds.append(cond)
                else:
                    nonindexed_preds.append(cond)

            # if there are indexed predicates, evaluate them first (get candidate IDs)
            candidate_ids = None
            if indexed_preds:

                sets = [self._eval_indexed_condition(col, val) for col, _, val in indexed_preds]
                idx_positions = [i for i, c in enumerate(conditions) if c in indexed_preds]

                # get logic ops between indexed predicates
                sub_ops = []
                for i in range(len(idx_positions) - 1):
                    pos = idx_positions[i]
                    sub_ops.append(logic_ops[pos])

                # combine left-to-right
                candidate_ids = sets[0]
                for j, s in enumerate(sets[1:]):
                    op = sub_ops[j].upper()
                    if op == 'AND':
                        candidate_ids = candidate_ids.intersection(s)
                    elif op == 'OR':
                        candidate_ids = candidate_ids.union(s)
                    else:
                        raise ValueError("ERROR: Unsupported logic op " + op)
            else:
                # no indexed predicates, candidate set is all rows
                candidate_ids = set(self.table.rows.keys())

            # narrow candidate_ids by evaluating all non-indexed predicates (checking rows)
            if nonindexed_preds:
                filtered = set()
                for pk in candidate_ids:
                    row = self.table.get_row(pk)
                    if not row:
                        continue
                    ok = True
                    # must evaluate full bool expr across all conditions; eval all conditions directly using conditions+logic_ops
                    # check only nonindexed ones, assume indexed already matched
                    for cond in nonindexed_preds:
                        col, op, val = cond
                        if op != '=':
                            raise NotImplementedError('ERROR: Only equality supported')
                        if normalize_value(row.get(col)) != normalize_value(val):
                            ok = False
                            break
                    if ok:
                        filtered.add(pk)
                return {i for i in filtered if i not in self.deleted_ids}
            else:
                # no nonindexed conditions => candidate_ids is final
                return {i for i in candidate_ids if i not in self.deleted_ids}
        else:
            # full table scan
            return self._full_scan_condition(conditions, logic_ops)

    def aggregate(self, pks, agg_specs):
        rows = [self.table.get_row(pk) for pk in pks if self.table.get_row(pk) is not None]
        results = {}
        for fn, col in agg_specs:
            key = fn if (not col) else f"{fn}_{col}"
            if fn == 'count':
                results[key] = len(rows)
                continue
            values = [r.get(col) for r in rows if r.get(col) is not None]
            if not values:
                results[key] = None
                continue
            if fn == 'sum':
                results[key] = sum(values)
            elif fn == 'min':
                results[key] = min(values)
            elif fn == 'max':
                results[key] = max(values)
            elif fn == 'avg':
                results[key] = sum(values) / len(values)
            else:
                raise ValueError(f"ERROR: Unknown aggregate {fn}")
        return results

    def debug_index_stats(self):
        return {col: idx.debug_stats() for col, idx in self.indexes.items()}


def parse_conditions(cond_str):
    tokens = cond_str.replace('(', ' ( ').replace(')', ' ) ').split()
    # parse left to right expecting pattern: COL OP VAL {LOGIC COL OP VAL}
    conditions  = []
    logic_ops = []
    i = 0
    while i < len(tokens):
        # skip opening parenthesis tokens
        if tokens[i] == '(':
            i += 1
            continue
        col = tokens[i]
        op = tokens[i + 1]
        val_token = tokens[i + 2]
        # handle quoted strings starting with ' or "
        if val_token.startswith("'") or val_token.startswith('"'):
            if val_token.endswith("'") or val_token.endswith('"'):
                val = val_token.strip("'\"")
                i += 3
            else:
                # join until closing quote
                j = i + 3
                accum = [val_token.strip("'\"")]
                while j < len(tokens) and not tokens[j].endswith("'") and not tokens[j].endswith('"'):
                    accum.append(tokens[j])
                    j += 1
                if j < len(tokens):
                    accum.append(tokens[j].strip("'\""))
                    i = j + 1
                else:
                    raise ValueError("ERROR: Unterminated string in condition")
                val = ' '.join(accum)
        else:
            val = val_token
            i += 3
        conditions.append((col, op, normalize_value(val)))
        # next token may be logic op
        if i < len(tokens) and tokens[i].upper() in ("AND", "OR"):
                logic_ops.append(tokens[i].upper())
                i += 1

    return conditions, logic_ops

schema = ['ID', 'D1', 'D2', 'Fact1', 'Fact2']
indexed = ['D1', 'D2']
sys = LSMSystem(schema, indexed_cols=indexed, pk='ID')

for i in range(1, 51):
    row = {
        'ID': i,
        'D1': f'key{(i % 5)}',
        'D2': i % 3,
        'Fact1': i * 10,
        'Fact2': i * 100,
    }
    sys.insert(row)

print('Index stats after inserts are:')
pretty_print_index_stats(sys.debug_index_stats())

conds, ops = parse_conditions("D1 = 'key1' AND D2 = 2")

res = sys.search(conds, ops, use_index=True)
print('Search (indexed) result IDs are:', sorted(res), '\n')

pretty_print_agg(sys.aggregate(res, [('sum', 'Fact1'), ('sum', 'Fact2'), ('count', '')]))

res2 = sys.search(conds, ops, use_index=False)
print('Search (full scan) result IDs are:', sorted(res2), '\n')

sys.delete(11)
sys.delete(16)
print('After deletion index stats are:')
pretty_print_index_stats(sys.debug_index_stats())

res3 = sys.search(conds, ops, use_index=True)
print('After delete search result is:', sorted(res3), '\n')

print('Aggregates on all rows:', sys.aggregate(set(sys.table.rows.keys()), [('count', '')]))

print('\nDone.\n')

