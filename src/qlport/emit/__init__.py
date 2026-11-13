"""Back ends: IR -> query text, or Bottom(gap) when the target cannot express a node.

A back end must never approximate silently: an ε-sound (non-exact) emission is allowed
only when the caller opts in, and is reported with its Gap.
"""
