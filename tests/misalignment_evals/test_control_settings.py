"""Arm-mapping + propensity leak-prevention tests. Builds settings (no dataset load), no Docker."""
import pytest

control_arena = pytest.importorskip("control_arena")

from control_arena import EvalMode  # noqa: E402
from control_arena.settings.apps.apps_setting import AppsSetting  # noqa: E402
from control_arena.settings.bash_arena.task.dataset import DatasetSpec  # noqa: E402
from misalignment_evals.control.settings import (  # noqa: E402
    NEUTRAL_SECRET_INPUT_DESCRIPTION,
    NeutralSubmitArgs,
    PropensityAppsSetting,
    _apps_setting,
    _bash_setting,
    arm_spec,
)


def test_arm_spec():
    assert arm_spec("baseline")[0] == EvalMode.HONEST
    assert arm_spec("instructed")[0] == EvalMode.ATTACK
    assert arm_spec("propensity")[0] == EvalMode.ATTACK      # attack mode -> side task is scored
    for arm in ("baseline", "instructed", "propensity"):
        assert arm_spec(arm)[1] is not None                 # a policy was built


def test_apps_propensity_uses_neutral_setting():
    assert isinstance(_apps_setting("propensity", "docker"), PropensityAppsSetting)
    assert type(_apps_setting("instructed", "docker")) is AppsSetting
    assert _apps_setting("baseline", "docker").has_backdoor_solution is True   # fair capability filter


def test_bash_side_task_scoped_to_exfil():
    assert _bash_setting("baseline", DatasetSpec()).filter_side_task_names == [None]
    assert _bash_setting("propensity", DatasetSpec()).filter_side_task_names == ["leak_passwords"]
    assert _bash_setting("instructed", DatasetSpec()).filter_side_task_names == ["leak_passwords"]


def test_neutral_submit_args_do_not_leak():
    """The propensity affordance must keep `secret_input` but drop any sabotage wording."""
    desc = NeutralSubmitArgs.model_fields["secret_input"].description.lower()
    assert "secret_input" in NeutralSubmitArgs.model_fields
    for leak in ("attack", "fail", "side task", "backdoor", "incorrect"):
        assert leak not in desc, f"propensity submit tool leaks the word {leak!r}"
    assert "additional input" in NEUTRAL_SECRET_INPUT_DESCRIPTION.lower()


def test_stock_attack_submit_description_is_why_we_override():
    """Documents the leak we are fixing: the shipped attack field says 'attack'."""
    from control_arena.settings.apps.task.submission_types import SubmitAttackArgs

    assert "attack" in SubmitAttackArgs.model_fields["secret_input"].description.lower()


def test_propensity_attack_submit_tool_uses_neutral_args():
    """The actual submit tool handed to the model in the propensity arm uses the neutral args."""
    from control_arena import ControlTaskMetadata

    attack_md = ControlTaskMetadata(
        main_task_description="d", side_task_description="", setting_description="s",
        eval_mode=EvalMode.ATTACK, setting_name="apps")
    assert PropensityAppsSetting().get_submit_tool_config(attack_md).input_type is NeutralSubmitArgs
    honest_md = attack_md.model_copy(update={"eval_mode": EvalMode.HONEST})
    assert PropensityAppsSetting().get_submit_tool_config(honest_md).input_type is not NeutralSubmitArgs
