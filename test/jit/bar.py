import torch
# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script  # noqa: B903
class FooSameName(object):
    def __init__(self, y):
        self.y = y
