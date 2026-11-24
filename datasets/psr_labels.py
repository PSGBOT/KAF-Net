PSR_FUNC_CAT = [
    "other",  # 0
    "handle",  # 1
    "housing",  # 2
    "support",  # 3
    "frame",  # 4
    "button",  # 5
    "wheel",  # 6
    "display",  # 7
    "cover",  # 8
    "plug",  # 9
    "port",  # 10
    "door",  # 11
    "container",  # 12
]
PSR_FUNC_CAT_IDX = {v: i for i, v in enumerate(PSR_FUNC_CAT)}

PSR_KR_CAT = [
    "unknown",
    "fixed",
    "revolute-free",
    "revolute-controlled",
    "revolute-static",
    "prismatic-free",
    "prismatic-controlled",
    "prismatic-static",
    "spherical-free",
    "spherical-controlled",
    "spherical-static",
    "supported",
    "flexible",
    "unrelated",
]

PSR_KR_CAT_IDX = {v: i for i, v in enumerate(PSR_KR_CAT)}

PSR_MEAN = [0.40789654, 0.44719302, 0.47026115]
PSR_STD = [0.28863828, 0.27408164, 0.27809835]
PSR_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
PSR_EIGEN_VECTORS = [
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938],
]


def relation_index(joint, categories=PSR_KR_CAT_IDX):
    joint_type = joint.get("joint_type", "unknown")
    control = joint.get("controllable")
    label = joint_type
    if joint_type not in {"fixed", "unrelated", "supported", "flexible"} and control:
        label = f"{joint_type}-{control}"
    return categories.get(label, 0)
