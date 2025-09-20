from typing import List

MOVE_GROUP_ARM: str = "cr5_group"

def joint_names() -> List[str]:
    return [f"joint{i}" for i in range(1,7)]


def base_link_name() -> str:
    return "dummy_link"


def end_effector_name() -> str:
    return "Link6"