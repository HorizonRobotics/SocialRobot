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

import unittest
from social_bot.parser import ReParser


class ParserTest(unittest.TestCase):
    def test_reparser(self):
        parser = ReParser()
        parser.add_word_class("OBJ", ["apple", "orange", "peach"])
        parser.add_word_class("COLOR", ["red", "orange", "yellow"])
        parser.add_rule("((it is )?an? )?<OBJ>", "OBJ")
        parser.add_rule("(it is )?<COLOR>", "COLOR")
        parser.add_rule("<OBJ> is edible", "EDIBLE")
        parser.add_rule("<OBJ> is <COLOR>", "OBJ_COLOR")
        parser.add_rule(
            "an? <OBJ:OBJ_RIGHT> is on the right of an? <OBJ:OBJ_LEFT>",
            "OBJ_POS")
        parser.add_rule(
            "an? <OBJ:OBJ_LEFT> is on the left of an? <OBJ:OBJ_RIGHT>",
            "OBJ_POS")

        for sentence in ["orange", "an orange", "it is an orange"]:
            result = parser.parse(sentence)
            self.assertIsNotNone(result)
            self.assertEqual(result.rule_name, "OBJ")
            self.assertEqual(result.slot_values, {"OBJ": "orange"})

        for sentence in ["n orange", "it orange"]:
            result = parser.parse(sentence)
            self.assertIsNone(result)

        for sentence in [
                "a peach is on the right of an apple",
                "an apple is on the left of a peach"
        ]:
            result = parser.parse(sentence)
            self.assertIsNotNone(result)
            self.assertEqual(result.rule_name, "OBJ_POS")
            self.assertEqual(result.slot_values, {
                "OBJ_RIGHT": "peach",
                "OBJ_LEFT": "apple"
            })


if __name__ == '__main__':
    unittest.main()
