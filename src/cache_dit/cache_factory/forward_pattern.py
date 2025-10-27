from enum import Enum


class ForwardPattern(Enum):
    def __init__(
        self,
        Return_H_First,
        Return_H_Only,
        Forward_H_only,
        In,
        Out,
        Supported,
    ):
        self.Return_H_First = Return_H_First
        self.Return_H_Only = Return_H_Only
        self.Forward_H_only = Forward_H_only
        self.In = In
        self.Out = Out
        self.Supported = Supported

    Pattern_0 = (
        True,  # Return_H_First
        False,  # Return_H_Only
        False,  # Forward_H_only
        ("hidden_states", "encoder_hidden_states"),  # In
        ("hidden_states", "encoder_hidden_states"),  # Out
        True,  # Supported
    )

    Pattern_1 = (
        False,  # Return_H_First
        False,  # Return_H_Only
        False,  # Forward_H_only
        ("hidden_states", "encoder_hidden_states"),  # In
        ("encoder_hidden_states", "hidden_states"),  # Out
        True,  # Supported
    )

    Pattern_2 = (
        False,  # Return_H_First
        True,  # Return_H_Only
        False,  # Forward_H_only
        ("hidden_states", "encoder_hidden_states"),  # In
        ("hidden_states",),  # Out
        True,  # Supported
    )

    Pattern_3 = (
        False,  # Return_H_First
        True,  # Return_H_Only
        True,  # Forward_H_only
        ("hidden_states",),  # In
        ("hidden_states",),  # Out
        True,  # Supported
    )

    Pattern_4 = (
        True,  # Return_H_First
        False,  # Return_H_Only
        True,  # Forward_H_only
        ("hidden_states",),  # In
        ("hidden_states", "encoder_hidden_states"),  # Out
        True,  # Supported
    )

    Pattern_5 = (
        False,  # Return_H_First
        False,  # Return_H_Only
        True,  # Forward_H_only
        ("hidden_states",),  # In
        ("encoder_hidden_states", "hidden_states"),  # Out
        True,  # Supported
    )

    @staticmethod
    def supported_patterns():
        return [
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_2,
            ForwardPattern.Pattern_3,
            ForwardPattern.Pattern_4,
            ForwardPattern.Pattern_5,
        ]
