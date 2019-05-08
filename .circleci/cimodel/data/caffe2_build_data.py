#!/usr/bin/env python3

from cimodel.lib.conf_tree import ConfigNode, X
from cimodel.lib.conf_tree import Ver
import cimodel.data.dimensions as dimensions


CONFIG_TREE_DATA = [
    (Ver("ubuntu", "14.04"), [
        (Ver("gcc", "4.8"), [X("py2")]),
        (Ver("gcc", "4.9"), [X("py2")]),
    ]),
    (Ver("ubuntu", "16.04"), [
        (Ver("cuda", "8.0"), [X("py2")]),
        (Ver("cuda", "9.0"), [
            # TODO make explicit that this is a "secret TensorRT build"
            #  (see https://github.com/pytorch/pytorch/pull/17323#discussion_r259446749)
            X("py2"),
            X("cmake"),
        ]),
        (Ver("cuda", "9.1"), [X("py2")]),
        (Ver("mkl"), [X("py2")]),
        (Ver("gcc", "5"), [X("onnx_py2")]),
        (Ver("clang", "3.8"), [X("py2")]),
        (Ver("clang", "3.9"), [X("py2")]),
        (Ver("clang", "7"), [X("py2")]),
        (Ver("android"), [X("py2")]),
    ]),
    (Ver("centos", "7"), [
        (Ver("cuda", "9.0"), [X("py2")]),
    ]),
    (Ver("macos", "10.13"), [
        # TODO ios and system aren't related. system qualifies where the python comes
        #  from (use the system python instead of homebrew or anaconda)
        (Ver("ios"), [X("py2")]),
        (Ver("system"), [X("py2")]),
    ]),
]


class TreeConfigNode(ConfigNode):
    def __init__(self, parent, node_name, subtree):
        super(TreeConfigNode, self).__init__(parent, self.modify_label(node_name))
        self.subtree = subtree
        self.init2(node_name)

    # noinspection PyMethodMayBeStatic
    def modify_label(self, label):
        return str(label)

    def init2(self, node_name):
        pass

    def get_children(self):
        return [self.child_constructor()(self, k, v) for (k, v) in self.subtree]

    def is_build_only(self):
        return str(self.find_prop("compiler_version")) in [
            "gcc4.9",
            "clang3.8",
            "clang3.9",
            "clang7",
            "android",
        ] or self.find_prop("distro_version").name == "macos"


class TopLevelNode(TreeConfigNode):
    def __init__(self, node_name, subtree):
        super(TopLevelNode, self).__init__(None, node_name, subtree)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return DistroConfigNode


class DistroConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["distro_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return CompilerConfigNode


class CompilerConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["compiler_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return LanguageConfigNode


class LanguageConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["language_version"] = node_name
        self.props["build_only"] = self.is_build_only()

    def get_children(self):

        children = []
        for phase in dimensions.PHASES:
            if phase == "build" or not self.props["build_only"]:
                children.append(PhaseConfigNode(self, phase, []))

        return children


class PhaseConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["phase_name"] = node_name
