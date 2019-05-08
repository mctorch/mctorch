#!/usr/bin/env python3


from collections import OrderedDict


LIST_MARKER = "- "
INDENTATION_WIDTH = 2


def is_dict(data):
    return type(data) is dict or type(data) is OrderedDict


def is_collection(data):
    return is_dict(data) or type(data) is list


# TODO can eventually drop this custom sorting
def sortkey(x):
    k = x[0]
    return (
        k == "<<",
        k != "environment",
        k,
    )


def render(fh, data, depth, is_list_member=False):
    """
    PyYaml does not allow precise control over the quoting
    behavior, especially for merge references.
    Therefore, we use this custom YAML renderer.
    """

    indentation = " " * INDENTATION_WIDTH * depth

    if is_dict(data):

        tuples = list(data.items())
        if type(data) is not OrderedDict:
            tuples.sort(key=sortkey)

        for i, (k, v) in enumerate(tuples):

            # If this dict is itself a list member, the first key gets prefixed with a list marker
            list_marker_prefix = LIST_MARKER if is_list_member and not i else ""

            trailing_whitespace = "\n" if is_collection(v) else " "
            fh.write(indentation + list_marker_prefix + k + ":" + trailing_whitespace)

            render(fh, v, depth + 1 + int(is_list_member))

        # TODO Could eventually drop this cosmetic convention
        if depth == 2:
            fh.write("\n")

    elif type(data) is list:
        for v in data:
            render(fh, v, depth, True)

    else:
        list_member_prefix = indentation + LIST_MARKER if is_list_member else ""
        fh.write(list_member_prefix + str(data) + "\n")
