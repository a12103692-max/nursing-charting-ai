"""Conservative numeric correspondence discovery from training pairs only.

No clinical dictionary, evaluation examples, or hand-authored clinical templates.
Literal output text and record boundaries are preserved from the evidence.
"""

from collections import defaultdict
from dataclasses import dataclass
import re


NUMBER = re.compile(r'(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?(?![A-Za-z0-9_])')


def split_numbers(text, normalize=False):
    literals, values, end = [], [], 0
    for match in NUMBER.finditer(text):
        literals.append(text[end:match.start()])
        values.append(match.group())
        end = match.end()
    literals.append(text[end:])
    if normalize:
        literals = [re.sub(r'\s+', '', x).lower() for x in literals]
    return tuple(literals), tuple(values)


@dataclass
class NumericRule:
    context: str
    literals: tuple
    fixed: dict
    outputs: list
    evidence_ids: list
    blocked_by: list

    def generate(self, text, context):
        literals, values = split_numbers(text, normalize=True)
        if context != self.context or literals != self.literals:
            return None
        if any(values[index] != value for index, value in self.fixed.items()):
            return None
        records = []
        for pieces, mapping in self.outputs:
            result = pieces[0]
            for index, (kind, value) in enumerate(mapping):
                result += (values[value] if kind == 'slot' else value) + pieces[index + 1]
            records.append(result)
        return records


def discover(rows):
    groups = defaultdict(list)
    for row in rows:
        literals, values = split_numbers(row['input'], normalize=True)
        if not values:
            continue
        output_literals = tuple(split_numbers(x)[0] for x in row['records'])
        groups[(row['context'], literals, output_literals)].append(row)
    rules = []
    for (context, literals, output_literals), evidence in groups.items():
        inputs = [split_numbers(r['input'])[1] for r in evidence]
        if len(set(inputs)) < 2:
            continue
        columns = list(zip(*inputs))
        varying = {i for i, values in enumerate(columns) if len(set(values)) > 1}
        fixed = {i: values[0] for i, values in enumerate(columns) if i not in varying}
        outputs, used = [], set()
        valid = True
        for record_index, pieces in enumerate(output_literals):
            output_values = [split_numbers(r['records'][record_index])[1] for r in evidence]
            mapping = []
            for column in zip(*output_values):
                if len(set(column)) == 1:
                    mapping.append(('fixed', column[0]))
                    continue
                matches = [i for i in varying if tuple(columns[i]) == tuple(column)]
                if len(matches) != 1:
                    valid = False
                    break
                used.add(matches[0])
                mapping.append(('slot', matches[0]))
            outputs.append((pieces, mapping))
        # Do not guess ambiguous correspondence or silently discard varying facts.
        if not valid or used != varying:
            continue
        rule = NumericRule(context, literals, fixed, outputs, [r['id'] for r in evidence], [])
        # Another observed style/meaning for the same input family is uncertainty,
        # not permission to silently choose a majority answer or normalize Gold.
        for row in rows:
            generated = rule.generate(row['input'], row['context'])
            if generated is not None and generated != row['records']:
                rule.blocked_by.append(row['id'])
        rules.append(rule)
    return rules
