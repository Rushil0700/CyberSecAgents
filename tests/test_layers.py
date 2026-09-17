"""Tests for the six-layer precondition model.

Why these matter: if a layer precondition is wrong, red still learns *something* and the
curve still looks plausible -- it is just a curve for an easier game than the one we
claim in the report. Like the topology tests, this is a silent failure mode, so every
structural property of section 3.1 is asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from marlsoc.env import layers as L
from marlsoc.env.layers import Layer, LayerStatus


class TestSpecTable:
    """Properties PROJECT.md section 3.1 claims about the layer stack itself."""

    def test_all_six_layers_are_defined(self) -> None:
        assert set(L.LAYERS) == set(Layer)
        assert len(L.LAYERS) == 6

    def test_each_layer_is_a_different_class_of_control(self) -> None:
        """Section 3.1's actual thesis. Two layers sharing a control class would be the
        "just add more zones" design the spec explicitly rejects -- a longer corridor
        rather than a richer decision problem."""
        controls = [spec.control for spec in L.LAYERS.values()]
        assert len(set(controls)) == 6, controls

    def test_each_layer_is_beaten_by_a_different_action(self) -> None:
        # The 1:1 action-to-layer mapping in section 4.1, and the thing Slide 3 claims.
        actions = [spec.beaten_by for spec in L.LAYERS.values()]
        assert len(set(actions)) == 6, actions

    def test_the_reward_ladder_is_monotone_in_depth(self) -> None:
        """Section 5.4's argument only works if deeper is worth more.

        A non-monotone ladder would create a local optimum: red could earn more by
        loitering on a shallow layer than by advancing, which is precisely the
        sparse-reward failure the ladder exists to prevent.
        """
        rewards = [L.LAYERS[ly].breach_reward for ly in sorted(Layer)]
        assert rewards == sorted(rewards)
        assert rewards == [10, 20, 30, 40, 50, 60]

    def test_the_win_pays_more_than_any_single_layer(self) -> None:
        assert L.ALTER_CREDENTIALS_REWARD > max(s.breach_reward for s in L.LAYERS.values())
        assert L.total_ladder() == 310.0

    def test_action_to_layer_mapping_round_trips(self) -> None:
        for layer, spec in L.LAYERS.items():
            assert L.layer_for_action(spec.beaten_by) is layer
        assert L.layer_for_action("wait") is None
        assert L.layer_for_action("alter_credentials") is None  # the win, not a layer

    def test_the_privilege_layer_is_the_loudest(self) -> None:
        # Section 3.1 calls escalate_privilege "noisy, high detection risk"; slow_scan
        # exists precisely to be quiet. If these were equal, stealth would not be a
        # decision and SARSA's cliff-walking advantage (section 7.2) would vanish.
        noises = {ly: L.LAYERS[ly].noise for ly in Layer}
        assert noises[Layer.PRIVILEGE] == max(noises.values())
        assert noises[Layer.PERIMETER] == min(noises.values())


class TestPartialOrder:
    """The ordering red must respect -- and the one place it gets a choice."""

    def test_only_the_perimeter_is_attemptable_at_reset(self) -> None:
        assert LayerStatus.full().attemptable() == frozenset({Layer.PERIMETER})

    def test_the_chain_advances_one_layer_at_a_time(self) -> None:
        status = LayerStatus.full()
        for expected in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH,
                         Layer.SEGMENTATION):
            assert status.attemptable() == frozenset({expected})
            status = status.breach(expected)

    def test_layers_five_and_six_are_independent_once_the_pivot_is_held(self) -> None:
        """The design decision in the module docstring, asserted.

        If this ever collapses to a single attemptable layer, the environment has become
        a corridor and "red learned an ordering" stops being a finding.
        """
        status = LayerStatus.full()
        for ly in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH, Layer.SEGMENTATION):
            status = status.breach(ly)
        assert status.attemptable() == frozenset({Layer.PRIVILEGE, Layer.APPROVAL})

        # ...and either order reaches a winnable position.
        assert status.breach(Layer.PRIVILEGE).breach(Layer.APPROVAL).can_win()
        assert status.breach(Layer.APPROVAL).breach(Layer.PRIVILEGE).can_win()

    def test_skipping_a_layer_is_refused_loudly(self) -> None:
        # A silent no-op here would mean the layer model enforces nothing and the only
        # symptom would be red looking mysteriously good.
        with pytest.raises(ValueError):
            LayerStatus.full().breach(Layer.SEGMENTATION)

    def test_a_layer_cannot_be_breached_twice(self) -> None:
        # Otherwise red farms the +10 for slow_scan forever and never advances -- the
        # reward-shaping trap in section 15.
        once = LayerStatus.full().breach(Layer.PERIMETER)
        assert not once.can_attempt(Layer.PERIMETER)
        with pytest.raises(ValueError):
            once.breach(Layer.PERIMETER)

    def test_status_is_immutable(self) -> None:
        # Episodes must be exactly reproducible from a seed (CLAUDE.md 3.7); a shared
        # mutable flag set is how one episode's progress leaks into the next.
        before = LayerStatus.full()
        after = before.breach(Layer.PERIMETER)
        assert before.breached == frozenset()
        assert after.breached == frozenset({Layer.PERIMETER})


class TestCurriculumStages:
    """Section 7.4: layers switch on progressively, and off is not the same as broken."""

    def test_stage_one_activates_only_the_first_two_layers(self) -> None:
        stage = LayerStatus.for_stage(2)
        assert stage.active == frozenset({Layer.PERIMETER, Layer.DMZ_BOUNDARY})
        assert not stage.is_active(Layer.AUTH)

    def test_an_inactive_layer_is_satisfied_but_not_breached(self) -> None:
        """The distinction the second design note is about.

        Satisfied gates the *precondition*; breached gates the *reward*. Conflating them
        would pay stage 1 the full +210 ladder for four layers it never fought, and the
        progressive signal curriculum learning depends on would be noise.
        """
        stage = LayerStatus.for_stage(2)
        assert stage.is_satisfied(Layer.AUTH)
        assert not stage.is_breached(Layer.AUTH)
        assert not stage.can_attempt(Layer.AUTH)

    def test_a_shallow_stage_is_winnable_without_the_deep_layers(self) -> None:
        # Stage 1 must be completable or red never gets the positive signal that the
        # whole curriculum exists to provide (section 15, "red never wins").
        stage = LayerStatus.for_stage(2)
        stage = stage.breach(Layer.PERIMETER).breach(Layer.DMZ_BOUNDARY)
        assert stage.can_win()

    def test_the_full_game_is_not_winnable_until_both_deep_layers_fall(self) -> None:
        status = LayerStatus.full()
        for ly in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH, Layer.SEGMENTATION):
            status = status.breach(ly)
        assert not status.can_win()
        assert not status.breach(Layer.PRIVILEGE).can_win()   # L6 still standing
        assert not status.breach(Layer.APPROVAL).can_win()    # L5 still standing

    def test_every_stage_activates_a_prefix_of_the_layers(self) -> None:
        """No stage may activate a layer whose prerequisite is inactive -- that would be
        an unreachable layer and the stage could never promote (section 15)."""
        for deepest in range(1, 7):
            stage = LayerStatus.for_stage(deepest)
            for layer in stage.active:
                for req in L.LAYERS[layer].requires:
                    assert stage.is_active(req), f"stage {deepest}: {layer} needs {req}"


class TestDepthMetric:
    """Section 9's "layers breached per episode" -- the money graph."""

    def test_depth_counts_breaches_rather_than_taking_the_maximum(self) -> None:
        """Because the order is partial, max() would over-report.

        Breaching L6 before L5 is legal; calling that "depth 6" would claim red got
        further than it did and would make the headline graph a lie.
        """
        status = LayerStatus.full()
        for ly in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH, Layer.SEGMENTATION):
            status = status.breach(ly)
        status = status.breach(Layer.APPROVAL)   # L6 before L5
        assert status.depth == 5
        assert max(status.breached) == Layer.APPROVAL  # what we deliberately do not use

    def test_depth_is_zero_at_reset_and_six_when_fully_breached(self) -> None:
        assert LayerStatus.full().depth == 0
        status = LayerStatus.full()
        for ly in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH,
                   Layer.SEGMENTATION, Layer.PRIVILEGE, Layer.APPROVAL):
            status = status.breach(ly)
        assert status.depth == 6
