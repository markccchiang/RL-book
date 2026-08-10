from collections import Counter
import dataclasses
import random
import statistics
import unittest

from rl.chapter1.probability import (Coin, Die, Distribution, OldDie,
                                     expected_value, roll_dice, six_sided)

SEED = 20210101


class TestDistribution(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            Distribution()

    def test_subclass_must_implement_sample(self):
        class Incomplete(Distribution):
            pass

        with self.assertRaises(TypeError):
            Incomplete()

    def test_concrete_subclass_is_usable(self):
        class Always7(Distribution[int]):
            def sample(self) -> int:
                return 7

        self.assertEqual(Always7().sample(), 7)


class TestOldDie(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)
        self.die = OldDie(6)

    def test_sample_within_range(self):
        rolls = [self.die.sample() for _ in range(1000)]
        self.assertTrue(all(1 <= r <= 6 for r in rolls))

    def test_sample_covers_every_face(self):
        rolls = {self.die.sample() for _ in range(1000)}
        self.assertEqual(rolls, {1, 2, 3, 4, 5, 6})

    def test_one_sided_die_is_constant(self):
        self.assertEqual(OldDie(1).sample(), 1)

    def test_not_equal_to_other_types(self):
        self.assertNotEqual(self.die, 6)
        self.assertNotEqual(self.die, "Die(sides=6)")

    def test_differing_sides_are_unequal(self):
        self.assertNotEqual(OldDie(6), OldDie(20))

    def test_two_old_dice_are_equal(self):
        self.assertEqual(OldDie(6), OldDie(6))

    def test_not_equal_to_dataclass_die(self):
        # OldDie and Die are distinct classes that happen to model the
        # same thing, so instances of one are never equal to the other.
        self.assertNotEqual(OldDie(6), Die(6))
        self.assertNotEqual(Die(6), OldDie(6))

    def test_defining_eq_dropped_hashability(self):
        # A class that defines __eq__ without __hash__ is unhashable,
        # so OldDie -- unlike Die -- cannot go in a set or dict.
        with self.assertRaises(TypeError):
            hash(OldDie(6))


class TestDie(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)
        self.die = Die(6)

    def test_sample_within_range(self):
        rolls = [self.die.sample() for _ in range(1000)]
        self.assertTrue(all(1 <= r <= 6 for r in rolls))

    def test_sample_covers_every_face(self):
        rolls = {self.die.sample() for _ in range(1000)}
        self.assertEqual(rolls, {1, 2, 3, 4, 5, 6})

    def test_sample_is_roughly_uniform(self):
        counts = Counter(self.die.sample() for _ in range(60000))
        for face in range(1, 7):
            self.assertLess(abs(counts[face] / 60000 - 1 / 6), 0.01)

    def test_equality_by_value(self):
        self.assertEqual(Die(6), Die(6))
        self.assertNotEqual(Die(6), Die(20))

    def test_hashable(self):
        self.assertEqual(len({Die(6), Die(6), Die(20)}), 2)

    def test_frozen(self):
        # setattr rather than `self.die.sides = 20`: the plain
        # assignment is a static type error ("Property "sides" ... is
        # read-only"), so type checkers and IDEs flag the line even
        # though raising is exactly what the test asserts.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(self.die, "sides", 20)

    def test_replace_returns_a_new_die(self):
        # The supported way to "change" a frozen instance: derive a new
        # one and leave the original alone.
        bigger = dataclasses.replace(self.die, sides=20)
        self.assertEqual(bigger, Die(20))
        self.assertEqual(self.die, Die(6))

    def test_repr(self):
        self.assertEqual(repr(Die(6)), "Die(sides=6)")

    def test_seeding_makes_samples_reproducible(self):
        random.seed(SEED)
        first = [Die(6).sample() for _ in range(20)]
        random.seed(SEED)
        second = [Die(6).sample() for _ in range(20)]
        self.assertEqual(first, second)


class TestCoin(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)
        self.coin = Coin()

    def test_sample_is_heads_or_tails(self):
        flips = {self.coin.sample() for _ in range(1000)}
        self.assertEqual(flips, {"heads", "tails"})

    def test_sample_is_roughly_fair(self):
        flips = [self.coin.sample() for _ in range(60000)]
        heads = flips.count("heads") / len(flips)
        self.assertLess(abs(heads - 0.5), 0.01)

    def test_all_coins_are_equal(self):
        # Coin is a frozen dataclass with no fields, so every instance
        # is equal to (and hashes like) every other.
        self.assertEqual(Coin(), Coin())
        self.assertEqual(len({Coin(), Coin()}), 1)


class TestRollDice(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)

    def test_six_sided_module_global(self):
        self.assertEqual(six_sided.sides, 6)

    def test_sum_within_range(self):
        rolls = [roll_dice() for _ in range(1000)]
        self.assertTrue(all(2 <= r <= 12 for r in rolls))

    def test_covers_every_total(self):
        rolls = {roll_dice() for _ in range(2000)}
        self.assertEqual(rolls, set(range(2, 13)))

    def test_seven_is_the_most_common_total(self):
        counts = Counter(roll_dice() for _ in range(60000))
        self.assertEqual(counts.most_common(1)[0][0], 7)

    def test_mean_is_about_seven(self):
        rolls = [roll_dice() for _ in range(60000)]
        self.assertLess(abs(sum(rolls) / len(rolls) - 7), 0.05)


class TestExpectedValue(unittest.TestCase):
    def setUp(self):
        random.seed(SEED)

    def test_die_expectation(self):
        # The standard error on 60000 rolls is ~0.007, so 0.035 is a
        # 5-sigma bound -- loose enough to survive a change of seed.
        self.assertLess(abs(expected_value(Die(6), 60000) - 3.5), 0.035)

    def test_constant_distribution_expectation(self):
        self.assertEqual(expected_value(Die(1), 100), 1)

    def test_single_sample_is_a_face_of_the_die(self):
        self.assertIn(expected_value(Die(6), 1), range(1, 7))

    def test_rejects_zero_samples(self):
        # statistics.mean raises on an empty sequence.
        with self.assertRaises(statistics.StatisticsError):
            expected_value(Die(6), 0)


if __name__ == '__main__':
    unittest.main()
