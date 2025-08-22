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
        True,
        False,
        False,
        ("hidden_states", "encoder_hidden_states"),
        ("hidden_states", "encoder_hidden_states"),
        True,
    )

    Pattern_1 = (
        False,
        False,
        False,
        ("hidden_states", "encoder_hidden_states"),
        ("encoder_hidden_states", "hidden_states"),
        True,
    )

    Pattern_2 = (
        False,
        True,
        False,
        ("hidden_states", "encoder_hidden_states"),
        ("hidden_states",),
        True,
    )

    Pattern_3 = (
        False,
        True,
        False,
        ("hidden_states",),
        ("hidden_states",),
        False,
    )

    @staticmethod
    def supported_patterns():
        return [
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_2,
        ]
