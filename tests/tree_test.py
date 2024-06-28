import unittest

from src.tree import Tree, generate_random





class TreeTestCase(unittest.TestCase):
    def test_random_tree(self):
        def add(x, y):
            return x + y

        def subtract(x, y):
            return x - y

        func_set = [add, subtract]
        terms = ["3", "2"]
        t = generate_random(func_set,terms, 1, var_count=1)
        print(t)
        t.optimize_values()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
