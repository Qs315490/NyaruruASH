"""Human-demo control masks must translate into this project's action space.

`data/idm-human.npz` is the only action-labelled data this project has: 24447
frames of recorded play.  Its 17-bit masks are over a *different* button list
than the one this project ships (it includes the map key and four camera keys
that were deliberately excluded, and it lacks this project's cancel/item split),
so the translation is explicit and the approximations are counted rather than
hidden - a snapped label is not the human's actual input.
"""

from __future__ import annotations

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace
from ash.train.demo_mapping import (
    DROPPED_LEGACY,
    LEGACY_CONTROLS,
    LEGACY_TO_BUTTON,
    legacy_mask_to_buttons,
    map_legacy_masks,
)


def _legacy(*names: str) -> int:
    mask = 0
    for name in names:
        mask |= 1 << LEGACY_CONTROLS.index(name)
    return mask


def test_only_the_map_and_camera_keys_are_dropped():
    """The excluded keys are named, so nobody has to guess why a frame is noop."""
    assert set(DROPPED_LEGACY) == {"m", "j", "i", "k", "l"}
    assert LEGACY_TO_BUTTON["f"] == "item"
    assert LEGACY_TO_BUTTON["escape"] == "menu"


def test_legacy_names_translate_to_this_projects_buttons():
    assert legacy_mask_to_buttons(_legacy("z")) == ("jump",)
    assert legacy_mask_to_buttons(_legacy("x")) == ("attack",)
    assert legacy_mask_to_buttons(_legacy("c")) == ("dash",)
    assert legacy_mask_to_buttons(_legacy("v")) == ("special",)
    assert legacy_mask_to_buttons(_legacy("a")) == ("ult",)
    assert legacy_mask_to_buttons(_legacy("f")) == ("item",)
    assert legacy_mask_to_buttons(_legacy("right", "z")) == ("right", "jump")
    # A camera pan is not an action this project can take.
    assert legacy_mask_to_buttons(_legacy("k")) == ()
    assert legacy_mask_to_buttons(0) == ()


def test_mapping_reports_exact_versus_snapped():
    space = ActionSpace.minimal()
    masks = np.array([
        _legacy(),              # idle -> noop, exact
        _legacy("right"),       # in the space, exact
        _legacy("right", "z"),  # in the space, exact
        _legacy("k"),           # dropped-only: noop (approximate by construction)
        _legacy("up", "z"),     # not in the curated space -> snapped
    ])
    index, report = map_legacy_masks(masks, space)
    assert report["frames"] == 5
    # The dropped-only frame becomes noop, which is a real member of the space,
    # so it counts as exact - the loss is reported separately below rather than
    # being folded into the snap count.
    assert report["exact"] == 4
    assert report["snapped"] == 1
    assert index[0] == space.index_of(0)
    assert index[1] == space.index_of(space.mask_at(2))     # ('right',)
    assert report["frames_whose_only_presses_were_dropped_keys"] == 1


def test_every_mapped_index_is_a_real_action():
    space = ActionSpace.minimal()
    masks = np.random.default_rng(0).integers(0, 1 << len(LEGACY_CONTROLS), size=200)
    index, _ = map_legacy_masks(masks, space)
    assert index.min() >= 0 and index.max() < len(space)


def test_real_demo_file_maps_almost_exactly():
    """The shipped 24447-frame recording must not be mostly approximate."""
    path = __import__("pathlib").Path("data/idm-human.npz")
    if not path.exists() and not path.with_suffix("").is_dir():
        pytest.skip("recorded demonstrations are not present")
    from ash.train.demo_mapping import load_demo

    masks = load_demo(path)["control_masks"]
    _, report = map_legacy_masks(masks, ActionSpace.minimal())
    assert report["exact"] > 0.95 * report["frames"], report
    # 471 frames (1.9%) record only a camera pan or the map key.  They become
    # noop, which is a real action, so they are reported rather than hidden -
    # this count read 0 until the check was moved ahead of the exact-match path.
    dropped = report["frames_whose_only_presses_were_dropped_keys"]
    assert 0 < dropped < 0.05 * report["frames"], report


def test_replay_is_bounded_and_varies_per_round():
    """A round's replay must be bounded, valid, and different each round.

    Loading all 24434 transitions every round would make a round cost minutes;
    replaying the *same* slice every round would overfit to that slice.  The
    round index is the seed precisely so successive rounds see new parts.
    """
    from ash.train.demo_mapping import demo_replay_trajectories

    path = __import__("pathlib").Path("data/idm-human.npz")
    if not path.exists() and not path.with_suffix("").is_dir():
        pytest.skip("recorded demonstrations are not present")
    space = ActionSpace.minimal()
    a = demo_replay_trajectories(path, space, steps=2000, seed=0)
    b = demo_replay_trajectories(path, space, steps=2000, seed=7)
    assert a and b
    total = sum(len(t["act"]) for t in a)
    assert 0 < total <= 2000 * 1.2, total
    # One action per transition, never one per frame.
    for traj in a:
        assert len(traj["act"]) == len(traj["obs"]) - 1
    assert any(not np.array_equal(x["obs"], y["obs"]) for x, y in zip(a, b)), \
        "the replay must move on between rounds"
    assert demo_replay_trajectories(path, space, steps=0, seed=0) == []


class _CountingIdm(torch.nn.Module):
    """Counts the transitions it is actually asked to fit on.

    Asserting on a reported count is not enough: the first version of the replay
    plumbing reported `replay_transitions: 3978` while the training set never
    contained them, because the edit that was supposed to add them silently
    matched nothing.  The only assertion that catches that is one that watches
    the data the model is fitted on.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        from ash.models.idm import IdmConfig

        self.config = IdmConfig(image_size=16, embed_dim=16, num_actions=num_actions)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.rows = 0

    def forward(self, frame_a, frame_b):
        n = int(frame_a.shape[0])
        self.rows += n
        # Must depend on a parameter, or the loss has no grad_fn and _fit_idm's
        # backward pass fails.
        scale = frame_a.mean(dim=(1, 2, 3)).unsqueeze(1) * 0.0
        return self.bias.expand(n, self.config.num_actions) + scale


def test_replay_reaches_the_idm_training_set(tmp_path):
    """The demonstrations must be fitted on, not merely counted."""
    from ash.loop.bootstrap import Bootstrapper, BootstrapConfig
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    space = ActionSpace.minimal()
    rng = np.random.default_rng(0)
    agent = {"obs": rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8),
             "act": np.zeros(7, dtype=np.int64)}
    replay = [{"obs": rng.integers(0, 255, (400, 16, 16, 3), dtype=np.uint8),
               "act": np.zeros(399, dtype=np.int64)}]
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))

    def loader(ids=None):
        return iter(())

    rows = {}
    for label, extra in (("plain", {}), ("replay", {"replay_trajectories": replay})):
        idm = _CountingIdm(len(space))
        b = Bootstrapper(BootstrapConfig(image_size=16, w_s=4, w_l=2, idm_epochs=1,
                                         policy_epochs=1, batch_size=8, device="cpu"))
        b.run(policy, idm, _K_stub(), [agent], loader, tmp_path,
              retrieved_ids=[], random_trajectories=[], **extra)
        rows[label] = idm.rows
    assert rows["plain"] > 0
    assert rows["replay"] > rows["plain"] + 300, (
        "the demonstrations are reported but never fitted on: %s" % rows
    )


def test_bootstrap_counts_the_replayed_transitions(tmp_path):
    """The replay must actually reach the IDM's training set, and be reported."""
    from ash.loop.bootstrap import Bootstrapper, BootstrapConfig
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig
    from ash.models.idm import IdmConfig, IdmModel

    space = ActionSpace.minimal()
    b = Bootstrapper(BootstrapConfig(image_size=16, w_s=4, w_l=2, idm_epochs=1,
                                     policy_epochs=1, batch_size=4, device="cpu"))
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))
    rng = np.random.default_rng(0)
    agent = {"obs": rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8),
             "act": np.zeros(7, dtype=np.int64)}
    replay = [{"obs": rng.integers(0, 255, (40, 16, 16, 3), dtype=np.uint8),
               "act": np.zeros(39, dtype=np.int64)}]

    def loader(ids=None):
        return iter(())

    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    report = b.run(policy, idm, _K_stub(), [agent], loader, tmp_path,
                   retrieved_ids=[], random_trajectories=[],
                   replay_trajectories=replay)
    assert report["idm"]["agent_transitions"] == 7
    assert report["idm"]["replay_transitions"] == 39, (
        "the demonstrations are not in the IDM's training set"
    )


def test_bootstrap_without_replay_reports_zero(tmp_path):
    from ash.loop.bootstrap import Bootstrapper, BootstrapConfig
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig
    from ash.models.idm import IdmConfig, IdmModel

    space = ActionSpace.minimal()
    b = Bootstrapper(BootstrapConfig(image_size=16, w_s=4, w_l=2, idm_epochs=1,
                                     policy_epochs=1, batch_size=4, device="cpu"))
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))
    rng = np.random.default_rng(0)
    agent = {"obs": rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8),
             "act": np.zeros(7, dtype=np.int64)}

    def loader(ids=None):
        return iter(())

    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    report = b.run(policy, idm, _K_stub(), [agent], loader, tmp_path,
                   retrieved_ids=[], random_trajectories=[])
    assert report["idm"]["replay_transitions"] == 0


class _K_stub:
    class _Emb:
        def embed(self, frames, batch_size=64):
            return np.zeros((len(frames), 4), dtype=np.float32)

    embedder = _Emb()

    def observe(self, embedding, seen):
        return False, -1, False

    def classify_sequence(self, embeddings):
        return np.zeros(len(embeddings), dtype=bool)

    def fit(self, embeddings, trajectory_ids):
        return {"clusters_total": 0, "clusters_kept": 0, "noise_rate": 1.0}


def test_idm_trains_on_mixed_resolutions():
    """The agent captures at 256x256; the demonstrations are stored at 128x128.

    `update_idm_from_trajectories` used to concatenate the raw frames and resize
    afterwards, so mixing the two sources raised
    "all the input array dimensions ... must match exactly" and killed the round.
    """
    from ash.loop.bootstrap import BootstrapConfig, Bootstrapper
    from ash.models.idm import IdmConfig, IdmModel

    space = ActionSpace.minimal()
    b = Bootstrapper(BootstrapConfig(image_size=16, idm_epochs=1, batch_size=4,
                                     device="cpu"))
    rng = np.random.default_rng(0)
    live = {"obs": rng.integers(0, 255, (6, 64, 64, 3), dtype=np.uint8),
            "act": np.zeros(5, dtype=np.int64)}
    demo = {"obs": rng.integers(0, 255, (6, 16, 16, 3), dtype=np.uint8),
            "act": np.zeros(5, dtype=np.int64)}
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    report = b.update_idm_from_trajectories(idm, [live, demo])
    assert report.get("idm_val") is not None, report


def test_corpus_videos_are_streamed_not_materialized(tmp_path):
    """The corpus loop must not hold every video at once.

    Adding progress logging to that loop, `videos = list(corpus_loader(...))`
    materialized all four retrieved videos at their capture resolution (2.4 GB
    each).  The kernel ran the machine out of memory and the round stalled in
    swap with 0 GB available.  The observable difference is *when* the loader is
    consumed relative to the first video being built: lazily, exactly one video
    of the policy pass has been pulled by then; eagerly, all of them have.
    """
    from ash.loop.bootstrap import BootstrapConfig, Bootstrapper
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig
    from ash.models.idm import IdmConfig, IdmModel

    space = ActionSpace.minimal()
    rng = np.random.default_rng(0)
    n_videos = 3
    module = __import__("ash.loop.bootstrap", fromlist=["x"])
    frames = {i: rng.integers(0, 255, (12, 16, 16, 3), dtype=np.uint8)
              for i in range(n_videos)}
    loads = {"n": 0}
    first_build_loads: list[int] = []

    def loader(ids=None):
        for i in range(n_videos):
            loads["n"] += 1
            yield "video%d" % i, frames[i]

    b = Bootstrapper(BootstrapConfig(image_size=16, w_s=4, w_l=2, idm_epochs=1,
                                     policy_epochs=1, batch_size=4, device="cpu"))
    logger = __import__("logging").getLogger("ash.loop.bootstrap")
    logger.disabled = True
    original = module.Bootstrapper.build_policy_dataset

    def spy(self, obs, idm, kdm, embeddings=None):
        if not first_build_loads:
            first_build_loads.append(loads["n"])
        return original(self, obs, idm, kdm, embeddings)

    module.Bootstrapper.build_policy_dataset = spy
    try:
        policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                           num_actions=len(space)))
        agent = {"obs": frames[0][:6], "act": np.zeros(5, dtype=np.int64)}
        idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
        b.run(policy, idm, _K_stub(), [agent], loader, tmp_path,
              retrieved_ids=["video%d" % i for i in range(n_videos)],
              random_trajectories=[])
    finally:
        module.Bootstrapper.build_policy_dataset = original
        logger.disabled = False

    assert first_build_loads, "no policy dataset was built"
    # K's pass over D^R is a full pass of its own, so n_videos loads have already
    # happened; a streaming policy pass adds exactly one more.
    assert first_build_loads[0] <= n_videos + 1, (
        "the corpus loader was materialized before the first video was built "
        "(%d loads for %d videos)" % (first_build_loads[0], n_videos)
    )
