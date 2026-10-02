"""Actual native CPU processors/autograd; no policy weights, optimizer steps, cloud or CUDA."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Run with the existing pinned native CPU Python environment.")

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from learning.common import ContractError, file_digest
from learning.smolvla import checkpoint_runner as runner

RECIPE = {
    "schema": "physicalai.smolvla-training-recipe/v1",
    "id": "native-temporal-padding-alias-v1",
    "boundary": "post-preprocessor-update-policy",
    "canonical_mask": "action_is_pad",
    "native_mask": "actions_id_pad",
}


class NativePaddingCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            raise AssertionError("Native padding tests must not use a CUDA runtime.")
        torch.set_num_threads(1)
        cls.root = tempfile.TemporaryDirectory(prefix="smol-padding-cpu-")
        cls.path = Path(cls.root.name)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
        tokenizer.save_pretrained(cls.path / "tokenizer")
        cls.config = SmolVLAConfig(
            device="cpu",
            push_to_hub=False,
            use_amp=False,
            vlm_model_name=str(cls.path / "tokenizer"),
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
            n_action_steps=1,
        )
        cls.stats = {
            key: {"mean": torch.zeros(9), "std": torch.ones(9)}
            for key in ("action", "observation.state")
        }

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def processors(self):
        return make_smolvla_pre_post_processors(self.config, dataset_stats=self.stats)

    def batch(self, *, padded=True, size=2):
        mask = torch.zeros(size, 50, dtype=torch.bool)
        if padded:
            mask[:, 25:] = True
        return {
            "observation.state": torch.zeros(size, 1, 9),
            "action": torch.zeros(size, 50, 9),
            "action_is_pad": mask,
            "task": ["place"] * size,
        }

    def loss(self, batch):
        parameter = torch.full((2, 50, 32), 2.0, requires_grad=True)
        fixture = SimpleNamespace(
            config=SimpleNamespace(adapt_to_pi_aloha=False, max_action_dim=32),
            prepare_images=lambda value: (None, None),
            prepare_state=lambda value: None,
            prepare_action=lambda value: None,
            model=SimpleNamespace(forward=lambda *args: parameter.square()),
        )
        value, metrics = SmolVLAPolicy.forward(fixture, batch)
        value.backward()
        return value.detach(), parameter.grad, metrics

    def test_exact_installed_native_hashes_and_recipe_runtime(self):
        value = runner.native_runtime(training_recipe=RECIPE)
        self.assertEqual(value["device"], "cpu")
        self.assertEqual(value["training_recipe"], RECIPE)
        self.assertEqual(
            value["native_source_sha256"]["lerobot/policies/smolvla/modeling_smolvla.py"],
            "3bdbaeecbd0dd3908d08507c13ed3517e63d2a653555322e2428066efb77b5f4",
        )

    def test_preprocessor_drops_premature_alias_but_keeps_canonical_mask(self):
        pre, _ = self.processors()
        raw = self.batch()
        prepared = pre({**raw, "actions_id_pad": raw["action_is_pad"]})
        self.assertNotIn("actions_id_pad", prepared)
        self.assertEqual(prepared["action_is_pad"].dtype, torch.bool)
        self.assertEqual(prepared["action_is_pad"].device, prepared["action"].device)
        self.assertTrue(torch.equal(prepared["action_is_pad"], raw["action_is_pad"]))

    def test_real_preprocessor_to_native_forward_masks_only_padded_steps_and_gradients(self):
        pre, _ = self.processors()
        raw = self.batch()
        prepared = pre(raw)
        original_keys = set(prepared)
        value = runner.training_padding_batch(prepared)
        self.assertEqual(set(prepared), original_keys)
        self.assertIs(value["action_is_pad"], prepared["action_is_pad"])
        self.assertIs(value["actions_id_pad"], prepared["action_is_pad"])
        self.assertIs(value["action"], prepared["action"])
        loss, gradients, metrics = self.loss(value)
        self.assertEqual(loss.item(), 2.0)
        self.assertEqual(torch.count_nonzero(gradients[:, 25:]).item(), 0)
        self.assertTrue(bool((gradients[:, :25] != 0).all()))
        self.assertIn("losses_after_in_ep_bound", metrics)
        self.assertEqual(gradients.shape, (2, 50, 32))

    def test_unpadded_loss_and_gradients_are_bitwise_unchanged(self):
        pre, _ = self.processors()
        prepared = pre(self.batch(padded=False))
        old_loss, old_gradient, _ = self.loss(prepared)
        new_loss, new_gradient, _ = self.loss(runner.training_padding_batch(prepared))
        self.assertTrue(torch.equal(old_loss, new_loss))
        self.assertTrue(torch.equal(old_gradient, new_gradient))

    def test_invalid_canonical_mask_never_reaches_native_update(self):
        for bad in (
            torch.zeros(2, 50),
            torch.zeros(2, 50, dtype=torch.int64),
            torch.zeros(50, dtype=torch.bool),
            torch.zeros(2, 49, dtype=torch.bool),
            torch.zeros(2, 50, 1, dtype=torch.bool),
            [[False] * 50] * 2,
            None,
        ):
            with self.subTest(shape=getattr(bad, "shape", None), dtype=getattr(bad, "dtype", None)):
                batch = self.batch()
                batch["action_is_pad"] = bad
                with self.assertRaises(ContractError):
                    runner.training_padding_batch(batch)

    def test_missing_and_conflicting_alias_fail_without_coercion(self):
        missing = self.batch()
        del missing["action_is_pad"]
        with self.assertRaises(ContractError):
            runner.training_padding_batch(missing)
        for alias in (
            torch.ones(2, 50, dtype=torch.bool),
            torch.zeros(2, 50),
            torch.zeros(2, 49, dtype=torch.bool),
        ):
            with self.subTest(dtype=alias.dtype, shape=alias.shape):
                with self.assertRaises(ContractError):
                    runner.training_padding_batch(
                        {**self.batch(padded=False), "actions_id_pad": alias}
                    )
        original = self.batch()
        output = runner.training_padding_batch(
            {**original, "actions_id_pad": original["action_is_pad"].clone()}
        )
        self.assertIs(output["actions_id_pad"], original["action_is_pad"])

    def test_device_processor_moves_complementary_bool_without_new_helper_casts(self):
        from lerobot.processor.converters import batch_to_transition, transition_to_batch
        from lerobot.processor.device_processor import DeviceProcessorStep

        native = DeviceProcessorStep(device="meta", float_dtype="float16")
        prepared = transition_to_batch(native(batch_to_transition(self.batch())))
        self.assertEqual(prepared["action"].device.type, "meta")
        self.assertEqual(prepared["action"].dtype, torch.float16)
        self.assertEqual(prepared["action_is_pad"].device.type, "meta")
        self.assertEqual(prepared["action_is_pad"].dtype, torch.bool)
        output = runner.training_padding_batch(prepared)
        self.assertIs(output["actions_id_pad"], prepared["action_is_pad"])
        with self.assertRaises(ContractError):
            runner.training_padding_batch(
                {**prepared, "action_is_pad": self.batch()["action_is_pad"]}
            )

    def test_original_action_dtype_is_preserved_not_forced_to_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                batch = self.batch()
                batch["action"] = batch["action"].to(dtype)
                output = runner.training_padding_batch(batch)
                self.assertIs(output["action"], batch["action"])
                self.assertEqual(output["action"].dtype, dtype)

    def test_invalid_action_shape_or_dtype_is_not_silently_adapted(self):
        for action in (
            torch.zeros(2, 49, 9),
            torch.zeros(2, 50, 32),
            torch.zeros(50, 9),
            torch.zeros(0, 50, 9),
            torch.zeros(2, 50, 9, dtype=torch.int64),
            None,
        ):
            with self.subTest(shape=getattr(action, "shape", None)):
                with self.assertRaises(ContractError):
                    runner.training_padding_batch({**self.batch(), "action": action})

    def test_legacy_context_does_not_patch_native_update_or_inference(self):
        from lerobot.scripts import lerobot_train as native

        original_update = native.update_policy
        original_forward = SmolVLAPolicy.forward
        original_prediction = SmolVLAPolicy.predict_action_chunk
        context = {
            "schema": runner.CONTEXT_SCHEMA,
            "training_parameters": {"seed": 42, "resume_mode": "new"},
            "resume_checkpoint_root": None,
            "origin": {"test_only": True},
            "limits": asdict_limits(),
        }

        def native_train():
            self.assertIs(native.update_policy, original_update)

        with patch.object(native, "train", native_train):
            runner.train_native(context, None, lambda: 10)
        self.assertIs(SmolVLAPolicy.forward, original_forward)
        self.assertIs(SmolVLAPolicy.predict_action_chunk, original_prediction)

    def test_scoped_hook_calls_original_update_after_preprocessing_not_before(self):
        from lerobot.scripts import lerobot_train as native

        pre, _ = self.processors()
        prepared = pre(self.batch())
        calls = []

        def original_update(
            train_metrics,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator,
            lr_scheduler=None,
            lock=None,
            rabc_weights_provider=None,
        ):
            calls.append(batch)
            loss, _, _ = self.loss(batch)
            return loss

        context = {
            "schema": "physicalai.smolvla-checkpoint-run/v3",
            "training_recipe": RECIPE,
            "training_parameters": {"seed": 42, "resume_mode": "weights_only"},
            "resume_checkpoint_root": None,
            "origin": {"test_only": True},
            "limits": asdict_limits(),
        }

        def native_train():
            self.assertIsNot(native.update_policy, original_update)
            self.assertEqual(native.update_policy(None, None, prepared, None, 0, None).item(), 2.0)

        with ExitStack() as stack:
            stack.enter_context(patch.object(native, "update_policy", original_update))
            stack.enter_context(patch.object(native, "train", native_train))
            runner.train_native(context, None, lambda: 10)
            self.assertIs(native.update_policy, original_update)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("actions_id_pad", prepared)

    def test_original_native_update_reaches_masked_backward_without_an_optimizer_step(self):
        from lerobot.scripts import lerobot_train as native

        class StopAfterBackward(Exception):
            pass

        parameter = torch.full((2, 50, 32), 2.0, requires_grad=True)
        fixture = SimpleNamespace(
            train=lambda: None,
            config=SimpleNamespace(adapt_to_pi_aloha=False, max_action_dim=32),
            prepare_images=lambda batch: (None, None),
            prepare_state=lambda batch: None,
            prepare_action=lambda batch: None,
            model=SimpleNamespace(forward=lambda *args: parameter.square()),
        )
        fixture.forward = lambda batch: SmolVLAPolicy.forward(fixture, batch)
        observed = []

        def backward(loss):
            observed.append(loss.item())
            loss.backward()
            raise StopAfterBackward

        class NoOptimizer:
            def step(self):
                raise AssertionError("No optimizer step is authorized in this CPU check.")

        accelerator = SimpleNamespace(autocast=nullcontext, backward=backward)
        pre, _ = self.processors()
        original_update = native.update_policy
        context = {
            "schema": "physicalai.smolvla-checkpoint-run/v3",
            "training_recipe": RECIPE,
            "training_parameters": {"seed": 42, "resume_mode": "weights_only"},
            "resume_checkpoint_root": None,
            "origin": {"test_only": True},
            "limits": asdict_limits(),
        }

        def native_train():
            native.update_policy(None, fixture, pre(self.batch()), NoOptimizer(), 1, accelerator)

        with patch.object(native, "train", native_train):
            with self.assertRaises(StopAfterBackward):
                runner.train_native(context, None, lambda: 10)
        self.assertIs(native.update_policy, original_update)
        self.assertEqual(observed, [2.0])
        self.assertEqual(torch.count_nonzero(parameter.grad[:, 25:]).item(), 0)
        self.assertTrue(bool((parameter.grad[:, :25] != 0).all()))

    def test_native_processor_serialization_and_reload_need_no_custom_class(self):
        from lerobot.policies.factory import make_pre_post_processors

        with tempfile.TemporaryDirectory(dir=self.path) as temporary:
            path = Path(temporary)
            pre, post = self.processors()
            pre.save_pretrained(path)
            post.save_pretrained(path)
            before = {file.name: file_digest(file) for file in path.iterdir() if file.is_file()}
            runner.training_padding_batch(pre(self.batch()))
            self.assertEqual(
                before, {file.name: file_digest(file) for file in path.iterdir() if file.is_file()}
            )
            restored_pre, restored_post = make_pre_post_processors(
                self.config,
                pretrained_path=str(path),
                preprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                saved = json.loads((path / name).read_bytes())
                self.assertTrue(all("class" not in step for step in saved["steps"]))
                self.assertNotIn("training_recipe", saved)
            one, two = pre(self.batch()), restored_pre(self.batch())
            self.assertEqual(set(one), set(two))
            self.assertNotIn("actions_id_pad", two)
            for key in ("action", "observation.state", "action_is_pad"):
                self.assertTrue(torch.equal(one[key], two[key]))
            self.assertTrue(torch.equal(post(one["action"]), restored_post(two["action"])))


def asdict_limits():
    from dataclasses import asdict

    from learning.smolvla.checkpoints import DEFAULT_LIMITS

    return asdict(DEFAULT_LIMITS)


if __name__ == "__main__":
    unittest.main()
