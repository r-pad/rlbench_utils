from rlbench.tasks import (
    InsertOntoSquarePeg,
    InsertUsbInComputer,
    PhoneOnBase,
    PickAndLift,
    PickUpCup,
    PlaceHangerOnRack,
    PushButton,
    PutKnifeOnChoppingBoard,
    PutMoneyInSafe,
    PutToiletRollOnStand,
    ReachTarget,
    SlideBlockToTarget,
    SolvePuzzle,
    StackWine,
    TakeMoneyOutSafe,
    TakeUmbrellaOutOfUmbrellaStand,
)

########################################################################################
# common
########################################################################################

GRIPPER_OBJ_NAMES = [
    "Panda_leftfinger_visual",
    "Panda_rightfinger_visual",
    "Panda_gripper_visual",
    "Panda_link7_visual",  # This is the collar, which breaks symmetry.
]
GRIPPER_POSE_NAME = "gripper"

########################################################################################
# insert_onto_square_peg
########################################################################################


INSERT_ONTO_SQUARE_PEG = {
    "task_class": InsertOntoSquarePeg,
    "phase_order": ["grasp", "place"],
    "phase": {
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": ["square_ring"],
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": "square_ring",
        },
        "place": {
            "action_obj_names": ["square_ring"],
            "anchor_obj_names": ["square_base", "pillar0", "pillar1", "pillar2"],
            "action_pose_name": "square_ring",
            "anchor_pose_name": "square_base",
        },
    },
}


########################################################################################
# pick_and_lift
########################################################################################

BLOCK_OBJ_NAMES = ["pick_and_lift_target"]
LIFT_GOAL_OBJ_NAMES = ["success_visual"]
BLOCK_POSE_NAME = "pick_and_lift_target"
LIFT_GOAL_POSE_NAME = "success_visual"

PICK_AND_LIFT = {
    "task_class": PickAndLift,
    "phase_order": ["pregrasp", "grasp", "lift", "final"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": BLOCK_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": BLOCK_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": BLOCK_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": BLOCK_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": BLOCK_OBJ_NAMES,
            "anchor_obj_names": BLOCK_OBJ_NAMES,
            "action_pose_name": BLOCK_POSE_NAME,
            "anchor_pose_name": BLOCK_POSE_NAME,
            "gripper_open": False,
        },
        "final": {
            "action_obj_names": BLOCK_OBJ_NAMES,
            "anchor_obj_names": LIFT_GOAL_OBJ_NAMES,
            "action_pose_name": BLOCK_POSE_NAME,
            "anchor_pose_name": LIFT_GOAL_POSE_NAME,
            "gripper_open": False,
        },
    },
    # First index = start ix, second index = length.
    # "custom_lowdim": {
    #     BLOCK_POSE_NAME: (0, 3),
    #     LIFT_GOAL_POSE_NAME: (3, 3),
    # },
}

########################################################################################
# pick_up_cup
########################################################################################

CUP_OBJ_NAMES = ["cup1_visual"]
CUP_POSE_NAME = "cup1_visual"

PICK_UP_CUP = {
    "task_class": PickUpCup,
    "phase_order": ["pregrasp", "grasp", "lift"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": CUP_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": CUP_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": CUP_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": CUP_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": CUP_OBJ_NAMES,
            "anchor_obj_names": CUP_OBJ_NAMES,
            "action_pose_name": CUP_POSE_NAME,
            "anchor_pose_name": CUP_POSE_NAME,
            "gripper_open": False,
        },
    },
}

########################################################################################
# phone_on_base
########################################################################################

PHONE_OBJ_NAMES = ["phone_visual"]
PHONE_CASE_OBJ_NAMES = ["phone_case_visual"]
PHONE_POSE_NAME = "phone_visual"
PHONE_CASE_POSE_NAME = "phone_case_visual"

PHONE_ON_BASE = {
    "task_class": PhoneOnBase,
    "phase_order": ["pregrasp", "grasp", "lift", "place"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": PHONE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": PHONE_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": PHONE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": PHONE_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": PHONE_OBJ_NAMES,
            "anchor_obj_names": PHONE_OBJ_NAMES,
            "action_pose_name": PHONE_POSE_NAME,
            "anchor_pose_name": PHONE_POSE_NAME,
            "gripper_open": False,
        },
        "place": {
            "action_obj_names": PHONE_OBJ_NAMES,
            "anchor_obj_names": PHONE_CASE_OBJ_NAMES,
            "action_pose_name": PHONE_POSE_NAME,
            "anchor_pose_name": PHONE_CASE_POSE_NAME,
            "gripper_open": True,
        },
    },
}

########################################################################################
# place_hanger_on_rack
########################################################################################

PLACE_HANGER_ON_RACK = {
    "task_class": PlaceHangerOnRack,
    "phase_order": ["grasp", "place"],
    "phase": {
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": ["clothes_hanger_visual"],
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": "clothes_hanger_visual",
        },
        "place": {
            "action_obj_names": ["clothes_hanger_visual"],
            "anchor_obj_names": [
                "clothes_rack",
                "clothes_rack_sub2",
                "clothes_rack_sub3",
            ],
            "action_pose_name": "clothes_hanger_visual",
            "anchor_pose_name": "clothes_rack",
        },
    },
}

########################################################################################
# push_button
########################################################################################

BUTTON_OBJ_NAMES = [
    "push_button_target",
    # "target_button_topPlate",  # For now, ignore this.
]
BUTTON_POSE_NAME = "push_button_target"

PUSH_BUTTON = {
    "task_class": PushButton,
    "phase_order": ["prepush", "postpush"],
    "phase": {
        "prepush": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": BUTTON_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": BUTTON_POSE_NAME,
            "gripper_open": False,
        },
        "postpush": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": BUTTON_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": BUTTON_POSE_NAME,
            "gripper_open": False,
        },
    },
}
########################################################################################
# put_knife_on_chopping_board
########################################################################################

KNIFE_OBJ_NAMES = ["knife_visual"]
CHOPPING_BOARD_OBJ_NAMES = ["chopping_board_visual"]
KNIFE_POSE_NAME = "knife_visual"
CHOPPING_BOARD_POSE_NAME = "chopping_board_visual"

PUT_KNIFE_ON_CHOPPING_BOARD = {
    "task_class": PutKnifeOnChoppingBoard,
    "phase_order": ["pregrasp", "grasp", "lift", "place"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": KNIFE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": KNIFE_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": KNIFE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": KNIFE_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": KNIFE_OBJ_NAMES,
            "anchor_obj_names": KNIFE_OBJ_NAMES,
            "action_pose_name": KNIFE_POSE_NAME,
            "anchor_pose_name": KNIFE_POSE_NAME,
            "gripper_open": False,
        },
        "place": {
            "action_obj_names": KNIFE_OBJ_NAMES,
            "anchor_obj_names": CHOPPING_BOARD_OBJ_NAMES,
            "action_pose_name": KNIFE_POSE_NAME,
            "anchor_pose_name": CHOPPING_BOARD_POSE_NAME,
            "gripper_open": True,
        },
    },
}

########################################################################################
# put_money_in_safe
########################################################################################

DOLLAR_STACK_OBJ_NAMES = ["dollar_stack"]
SAFE_OBJ_NAMES = ["safe_body", "safe_door"]
DOLLAR_STACK_POSE_NAME = "dollar_front_visual"
SAFE_POSE_NAME = "safe_body"

PUT_MONEY_IN_SAFE = {
    "task_class": PutMoneyInSafe,
    "phase_order": ["pregrasp", "grasp", "lift", "preplace", "place"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": DOLLAR_STACK_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": DOLLAR_STACK_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "anchor_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "action_pose_name": DOLLAR_STACK_POSE_NAME,
            "anchor_pose_name": DOLLAR_STACK_POSE_NAME,
            "gripper_open": False,
        },
        "preplace": {
            "action_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "anchor_obj_names": SAFE_OBJ_NAMES,
            "action_pose_name": DOLLAR_STACK_POSE_NAME,
            "anchor_pose_name": SAFE_POSE_NAME,
            "gripper_open": False,
        },
        "place": {
            "action_obj_names": DOLLAR_STACK_OBJ_NAMES,
            "anchor_obj_names": SAFE_OBJ_NAMES,
            "action_pose_name": DOLLAR_STACK_POSE_NAME,
            "anchor_pose_name": SAFE_POSE_NAME,
            "gripper_open": True,
        },
    },
}

########################################################################################
# put_toilet_roll_on_stand
########################################################################################

PUT_TOILET_ROLL_ON_STAND = {
    "task_class": PutToiletRollOnStand,
    "phase_order": ["grasp", "place"],
    "phase": {
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": ["toilet_roll_visual"],
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": "toilet_roll_visual",
        },
        "place": {
            "action_obj_names": ["toilet_roll_visual"],
            "anchor_obj_names": ["holder_visual", "stand_base"],
            "action_pose_name": "toilet_roll_visual",
            "anchor_pose_name": "holder_visual",
        },
    },
}

########################################################################################
# reach_target
########################################################################################

TARGET_OBJ_NAMES = ["target"]
TARGET_POSE_NAME = "target"

REACH_TARGET = {
    "task_class": ReachTarget,
    "phase_order": ["reach"],
    "phase": {
        "reach": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": TARGET_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": TARGET_POSE_NAME,
            "gripper_open": True,
        },
    },
    # "custom_lowdim": {
    #     TARGET_POSE_NAME: (0, 3),
    # },
}

########################################################################################
# slide_block_to_target
########################################################################################

SLIDE_BLOCK_OBJ_NAMES = ["block"]
TARGET_OBJ_NAMES = ["target"]
SLIDE_BLOCK_POSE_NAME = "block"
TARGET_POSE_NAME = "target"

SLIDE_BLOCK_TO_TARGET = {
    "task_class": SlideBlockToTarget,
    "phase_order": ["preslide", "postslide"],
    "phase": {
        "preslide": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": SLIDE_BLOCK_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": SLIDE_BLOCK_POSE_NAME,
            "gripper_open": False,
        },
        "postslide": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": TARGET_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": TARGET_POSE_NAME,
            "gripper_open": False,
        },
    },
}

########################################################################################
# solve_puzzle
########################################################################################

SOLVE_PUZZLE = {
    "task_class": SolvePuzzle,
    "phase_order": ["grasp", "place"],
    "phase": {
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": ["solve_puzzle_piece_visual2"],
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": "solve_puzzle_piece_visual2",
        },
        "place": {
            "action_obj_names": ["solve_puzzle_piece_visual2"],
            "anchor_obj_names": ["solve_puzzle_piece1"]
            + [f"solve_puzzle_piece{i}" for i in range(3, 25)],
            "action_pose_name": "solve_puzzle_piece_visual2",
            "anchor_pose_name": "solve_puzzle_piece1",
        },
    },
}

########################################################################################
# stack_wine
########################################################################################

WINE_OBJ_NAMES = ["wine_bottle_visual"]
RACK_OBJ_NAMES = ["rack_bottom_visual", "rack_top_visual"]
WINE_POSE_NAME = "wine_bottle"
RACK_POSE_NAME = "rack_top_visual"

STACK_WINE = {
    "task_class": StackWine,
    "phase_order": ["pregrasp", "grasp", "lift", "preplace", "place"],
    "phase": {
        # Pre-grasp position relative to the bottle.
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": WINE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": WINE_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": WINE_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": WINE_POSE_NAME,
            "gripper_open": False,
        },
        # Lift relative to itself.
        "lift": {
            "action_obj_names": WINE_OBJ_NAMES,
            "anchor_obj_names": WINE_OBJ_NAMES,
            "action_pose_name": WINE_POSE_NAME,
            "anchor_pose_name": WINE_POSE_NAME,
            "gripper_open": False,
        },
        # Pre-place position relative to the rack.
        "preplace": {
            "action_obj_names": WINE_OBJ_NAMES,
            "anchor_obj_names": RACK_OBJ_NAMES,
            "action_pose_name": WINE_POSE_NAME,
            "anchor_pose_name": RACK_POSE_NAME,
            "gripper_open": False,
        },
        "place": {
            "action_obj_names": WINE_OBJ_NAMES,
            "anchor_obj_names": RACK_OBJ_NAMES,
            "action_pose_name": WINE_POSE_NAME,
            "anchor_pose_name": RACK_POSE_NAME,
            "gripper_open": True,
        },
    },
}

########################################################################################
# take_money_out_of_safe
########################################################################################

SAFE_OBJ_NAMES = ["safe_body", "safe_door"]
MONEY_OBJ_NAMES = ["dollar_stack0", "dollar_back_visual0"]
SAFE_POSE_NAME = "safe_body"
# MONEY_POSE_NAME = "dollar_stack0"
MONEY_POSE_NAME = "dollar_front_visual0"


TAKE_MONEY_OUT_SAFE = {
    "task_class": TakeMoneyOutSafe,
    "phase_order": ["pregrasp", "grasp", "lift", "place"],
    "phase": {
        "pregrasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": MONEY_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": MONEY_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": MONEY_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": MONEY_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": MONEY_OBJ_NAMES,
            "anchor_obj_names": MONEY_OBJ_NAMES,
            "action_pose_name": MONEY_POSE_NAME,
            "anchor_pose_name": MONEY_POSE_NAME,
            "gripper_open": False,
        },
        "place": {
            "action_obj_names": MONEY_OBJ_NAMES,
            "anchor_obj_names": SAFE_OBJ_NAMES,
            "action_pose_name": MONEY_POSE_NAME,
            "anchor_pose_name": SAFE_POSE_NAME,
            "gripper_open": True,
        },
    },
}

########################################################################################
# take_umbrella_out_of_umbrella_stand
########################################################################################


UMBRELLA_OBJ_NAMES = ["umbrella_visual"]
UMBRELLA_STAND_OBJ_NAMES = ["stand_visual"]
UMBRELLA_POSE_NAME = "umbrella_visual"
UMBRELLA_STAND_POSE_NAME = "stand_visual"

TAKE_UMBRELLA_OUT_OF_UMBRELLA_STAND = {
    "task_class": TakeUmbrellaOutOfUmbrellaStand,
    "phase_order": ["pregasp", "grasp", "lift"],
    "phase": {
        "pregasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": UMBRELLA_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": UMBRELLA_POSE_NAME,
            "gripper_open": True,
        },
        "grasp": {
            "action_obj_names": GRIPPER_OBJ_NAMES,
            "anchor_obj_names": UMBRELLA_OBJ_NAMES,
            "action_pose_name": GRIPPER_POSE_NAME,
            "anchor_pose_name": UMBRELLA_POSE_NAME,
            "gripper_open": False,
        },
        "lift": {
            "action_obj_names": UMBRELLA_OBJ_NAMES,
            "anchor_obj_names": UMBRELLA_OBJ_NAMES,
            "action_pose_name": UMBRELLA_POSE_NAME,
            "anchor_pose_name": UMBRELLA_POSE_NAME,
            "gripper_open": False,
        },
    },
}


RLBENCH_10_TASKS = [
    "pick_and_lift",
    "pick_up_cup",
    "put_knife_on_chopping_board",
    "put_money_in_safe",
    "push_button",
    "reach_target",
    "slide_block_to_target",
    "stack_wine",
    "take_money_out_safe",
    "take_umbrella_out_of_umbrella_stand",
]


TASK_DICT = {
    "insert_onto_square_peg": INSERT_ONTO_SQUARE_PEG,
    # THIS ONE SEEMS TO BE BROKEN
    "insert_usb_in_computer": {
        "task_class": InsertUsbInComputer,
        "phase": {
            "grasp": {
                "action_obj_names": GRIPPER_OBJ_NAMES,
                "anchor_obj_names": ["usb", "usb_visual0", "usb_visual1", "tip"],
            },
            "place": {
                "action_obj_names": ["usb", "usb_visual0", "usb_visual1", "tip"],
                "anchor_obj_names": ["computer", "computer_visual"],
            },
        },
    },
    "pick_and_lift": PICK_AND_LIFT,
    "pick_up_cup": PICK_UP_CUP,
    "put_knife_on_chopping_board": PUT_KNIFE_ON_CHOPPING_BOARD,
    "put_money_in_safe": PUT_MONEY_IN_SAFE,
    "push_button": PUSH_BUTTON,
    "reach_target": REACH_TARGET,
    "slide_block_to_target": SLIDE_BLOCK_TO_TARGET,
    "stack_wine": STACK_WINE,
    "take_money_out_safe": TAKE_MONEY_OUT_SAFE,
    "take_umbrella_out_of_umbrella_stand": TAKE_UMBRELLA_OUT_OF_UMBRELLA_STAND,
    "phone_on_base": PHONE_ON_BASE,
    "put_toilet_roll_on_stand": PUT_TOILET_ROLL_ON_STAND,
    "place_hanger_on_rack": PLACE_HANGER_ON_RACK,
    "solve_puzzle": SOLVE_PUZZLE,
}
