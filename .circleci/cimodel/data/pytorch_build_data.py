#!/usr/bin/env python3

from cimodel.lib.conf_tree import ConfigNode, X


CONFIG_TREE_DATA = [
    ("trusty", [
        (None, [
            X("2.7.9"),
            X("2.7"),
            X("3.5"),
            X("nightly"),
        ]),
        ("gcc", [
            ("4.8", [X("3.6")]),
            ("5.4", [("3.6", [X(False), X(True)])]),
            ("7", [X("3.6")]),
        ]),
    ]),
    ("xenial", [
        ("clang", [
            ("5", [X("3.6")]),
        ]),
        ("cuda", [
            ("8", [X("3.6")]),
            ("9", [
                # Note there are magic strings here
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L21
                # and
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L143
                # and
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L153
                # (from https://github.com/pytorch/pytorch/pull/17323#discussion_r259453144)
                X("2.7"),
                X("3.6"),
            ]),
            ("9.2", [X("3.6")]),
            ("10", [X("3.6")]),
        ]),
        ("android", [
            ("r19c", [X("3.6")]),
        ]),
    ]),
]


def get_major_pyver(dotted_version):
    parts = dotted_version.split(".")
    return "py" + parts[0]


class TreeConfigNode(ConfigNode):
    def __init__(self, parent, node_name, subtree):
        super(TreeConfigNode, self).__init__(parent, self.modify_label(node_name))
        self.subtree = subtree
        self.init2(node_name)

    def modify_label(self, label):
        return label

    def init2(self, node_name):
        pass

    def get_children(self):
        return [self.child_constructor()(self, k, v) for (k, v) in self.subtree]


class TopLevelNode(TreeConfigNode):
    def __init__(self, node_name, subtree):
        super(TopLevelNode, self).__init__(None, node_name, subtree)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return DistroConfigNode


class DistroConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["distro_name"] = node_name

    def child_constructor(self):
        distro = self.find_prop("distro_name")

        next_nodes = {
            "trusty": TrustyCompilerConfigNode,
            "xenial": XenialCompilerConfigNode,
        }
        return next_nodes[distro]


class TrustyCompilerConfigNode(TreeConfigNode):

    def modify_label(self, label):
        return label or "<unspecified>"

    def init2(self, node_name):
        self.props["compiler_name"] = node_name

    def child_constructor(self):
        return TrustyCompilerVersionConfigNode if self.props["compiler_name"] else PyVerConfigNode


class TrustyCompilerVersionConfigNode(TreeConfigNode):

    def init2(self, node_name):
        self.props["compiler_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return PyVerConfigNode


class PyVerConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["pyver"] = node_name
        self.props["abbreviated_pyver"] = get_major_pyver(node_name)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return XlaConfigNode


class XlaConfigNode(TreeConfigNode):
    def modify_label(self, label):
        return "XLA=" + str(label)

    def init2(self, node_name):
        self.props["is_xla"] = node_name


class XenialCompilerConfigNode(TreeConfigNode):

    def init2(self, node_name):
        self.props["compiler_name"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return XenialCompilerVersionConfigNode


class XenialCompilerVersionConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["compiler_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return PyVerConfigNode
