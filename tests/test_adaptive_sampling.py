"""Sampler regressions use synthetic image IDs and deterministic model stand-ins."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sampling_marginal_relation_pipeline import LIME_subj_pipeline


class AdaptiveSamplingTests(unittest.TestCase):
    def make_sampler(self, predictor=None, y=1, **options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        images = [f"image{i:02}" for i in range(20)]
        for image in images:
            (root / f"synthetic_{image}.jpg").touch()
        if predictor is None:
            predictor = lambda selected: int("image00" in selected)
        return LIME_subj_pipeline(
            test_data_id="09", img_list=list(reversed(images)), y=y, mi_id="synthetic",
            img_dir=str(root), pred_func=predictor, result_parent_dir=str(root), **options)

    def sample(self, sampler, **options):
        return sampler.predict_on_random_samples_until_convergence(
            n_samples=20, target_positive_proportion=0.5,
            min_sample_size=3, max_sample_size=20, **options)

    def test_image_columns_round_trip_in_sorted_order(self):
        sampler = self.make_sampler()
        sampler.samples = [["image19", "image00"], ["image04"]]
        sampler.sample_pred_results = [1, 0]
        sampler.generate_pred_results_matrix()
        self.assertEqual(np.flatnonzero(sampler.pred_results_x[0]).tolist(), [0, 19])
        self.assertEqual(np.flatnonzero(sampler.pred_results_x[1]).tolist(), [4])
        for image, index in sampler.img_to_indx.items():
            self.assertEqual(sampler.indx_to_img[index], image)

    def test_numpy_zero_label_is_accepted_and_normalized(self):
        sampler = self.make_sampler(y=np.int64(0))
        self.assertIs(type(sampler.y), int)

    def test_pool_semantics_do_not_depend_on_ground_truth(self):
        for label in (0, 1):
            sampler = self.make_sampler(y=label)
            self.sample(sampler)
            self.assertEqual(sampler.positive_img_pool, ["image00"])
            self.assertNotIn("image00", sampler.negative_img_pool)

    def test_negative_quota_first_still_returns_full_batch(self):
        sampler = self.make_sampler(
            predictor=lambda selected: int(len(selected) == 1 and selected[0] == "image00"))
        self.sample(sampler)
        self.assertEqual(len(sampler.samples), 20)
        self.assertEqual(len(sampler.sample_pred_results), 20)
        self.assertFalse(sampler.sampling_summary["class_balance_reached"])

    def test_direct_pool_sampling_preserves_requested_sizes_when_a_pool_is_small(self):
        sampler = self.make_sampler()
        sampler.generate_balanced_random_samples(
            n_samples=5, target_positive_proportion=0.85,
            min_sample_size=20, max_sample_size=20, pred_on_samples=True)
        self.assertEqual([len(set(sample)) for sample in sampler.samples], [20] * 5)
        self.assertEqual(len(sampler.sample_pred_results), 5)

    def test_one_class_fallback_has_full_count_and_explicit_balance_status(self):
        sampler = self.make_sampler(predictor=lambda selected: 0)
        self.sample(sampler)
        self.assertEqual(len(sampler.samples), 20)
        self.assertTrue(all(3 <= len(sample) <= 20 for sample in sampler.samples))
        self.assertTrue(sampler.sampling_summary["one_pool_fallback"])
        self.assertEqual(sampler.sampling_summary["class_counts"], {0: 20, 1: 0})
        self.assertFalse(sampler.sampling_summary["class_balance_reached"])

    def test_seed_reproduces_masks_and_labels(self):
        first = self.make_sampler(random_state=17)
        second = self.make_sampler(random_state=17)
        self.sample(first)
        self.sample(second)
        self.assertEqual([list(s) for s in first.samples], [list(s) for s in second.samples])
        self.assertEqual(first.sample_pred_results, second.sample_pred_results)

    def test_reset_and_append_keep_samples_and_predictions_aligned(self):
        sampler = self.make_sampler(random_state=3)
        self.sample(sampler)
        self.sample(sampler, append_to_original=False)
        self.assertEqual(len(sampler.samples), 20)
        self.sample(sampler)
        self.assertEqual(len(sampler.samples), 40)
        self.assertEqual(len(sampler.sample_pred_results), 40)

    def test_invalid_bounds_and_insufficient_budget_do_not_predict(self):
        calls = []

        def predict(selected):
            calls.append(list(selected))
            return 0

        sampler = self.make_sampler(predictor=predict)
        before = len(calls)
        for options in (
            dict(min_sample_size=0, max_sample_size=20),
            dict(min_sample_size=10, max_sample_size=4),
            dict(min_sample_size=3, max_sample_size=20, max_iter=5),
            dict(min_sample_prop=0.1, max_sample_prop=2.0),
        ):
            with self.assertRaises(ValueError):
                sampler.predict_on_random_samples_until_convergence(
                    n_samples=20, target_positive_proportion=0.5, **options)
        self.assertEqual(len(calls), before)


if __name__ == "__main__":
    unittest.main()
