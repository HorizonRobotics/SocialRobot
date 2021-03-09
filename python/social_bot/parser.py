# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
import re

ParseRule = namedtuple('ParseRule',
                       ['pattern', 'rule_name', 'word_classes', 'slot_names'])

ParseResult = namedtuple('ParseResult', ['rule_name', 'slot_values'])


class ReParser(object):
    """A simple NL parse based on regex.

    The parse is based on a set of word classes and rules. It parses a sentence
    to ``ParseResult`` with ``slot_values`` properly filled.

    A word class can be added using ``add_word_class()``. And a parse rule can
    be added using ``add_rule()``. A parse rule is a python regular expression
    with a special syntax to denote slot. A slot can be denoted using the
    following two formats:

    * <WORD_CLASS_NAME>: the value of this slot is from the given word class.
      The name of the slot is also WORD_CLASS_NAME.
    * <WORD_CLASS_NAME:SLOT_NAME>: the value of this slot is from the given word
      class. The name of the slot is SLOT_NAME.

    All the slot names of a single parse rule need to be different.

    Examples:

    .. code-block:: python

        parser = ReParser()
        parser.add_word_class("OBJ", ["apple", "orange", "peach"])
        parser.add_rule("an? <OBJ:OBJ_RIGHT> is on the right of an? <OBJ:OBJ_LEFT>", "OBJ_POS")
        parser.add_rule("an? <OBJ:OBJ_LEFT> is on the left of an? <OBJ:OBJ_RIGHT>", "OBJ_POS")

        result = parser.parse("a peach is on the right of an apple")
        assert result.rule_name == "OBJ_POS")
        assert result.slot_values == {"OBJ_RIGHT": "peach", "OBJ_LEFT": "apple"}

    """

    def __init__(self):
        self._word_classes = {}
        self._rules = []

    def add_rule(self, template, rule_name=''):
        """Add a parse rule.

        Args:
            template (str): template for the sentences to be parsed by this rule.
            rule_name (str): name of the parse rule.
        """
        word_classes = []
        slot_names = []
        new_template = ""
        while True:
            m = re.search(r"<\w+(:\w+)?>", template)
            if not m:
                break
            slot = m.group(0)
            slot = slot[1:-1]
            if ':' in slot:
                word_class, slot_name = slot.split(':')
            else:
                word_class, slot_name = slot, slot
            assert word_class in self._word_classes, "Unknown world class %s" % word_class
            word_classes.append(word_class)
            assert slot_name not in slot_names, "duplicated slot name %s" % slot_name
            slot_names.append(slot_name)
            new_template += template[:m.start()] + ("(?P<%s>(%s))" % (
                slot_name, '|'.join(self._word_classes[word_class])))
            template = template[m.end():]
        pattern = re.compile(new_template)
        self._rules.append(
            ParseRule(pattern, rule_name, word_classes, slot_names))

    def add_word_class(self, word_class_name, words):
        """Add a word class.

        Args:
            word_class_name (str): name of the word class
            words (list[str]): the words/phrases of this word class.
        """
        assert word_class_name not in self._word_classes, (
            "duplicated world class name %s" % word_class_name)
        self._word_classes[word_class_name] = words

    def parse(self, sentence):
        """Parse a sentence.

        All the parse rules will be tried sequentially in the order they were
        added. The first correct parse result will be returned.

        Args:
            sentence (str): sentence to be parsed
        Returns:
            ParseResult: if can be successful parsed by one of the rule.
            - rule_name (str): which parse rule is used to parse this sentence
            - slot_values (dict[str,str]): values of all the slots defined by the
                parse rule.
            None: if cannot be parsed by any rule.
        """
        for rule in self._rules:
            m = rule.pattern.fullmatch(sentence)
            if not m:
                continue
            slot_values = dict((s, m.group(s)) for s in rule.slot_names)
            return ParseResult(rule.rule_name, slot_values)

        return None
