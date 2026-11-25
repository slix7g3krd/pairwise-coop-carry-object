import pytest
import gameplay.coop_carry_object as mod

CASES = [
    ("single_constraint_solve", (1,)),
    ("grab_serial", (1,)),
    ("release_on_disconnect", (1,)),
    ("tilt_by_mass", (1,)),
    ("contact_stuck", (1,)),
    ("rollback_freeze_input", (1,)),
    ("grab", (1,)),
    ("carry", (1,)),
    ("release", (1,)),
    ("sync", (1,)),
    ("validate_carry", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
