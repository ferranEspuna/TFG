from persim.persistent_entropy import persistent_entropy


def get_all_summaries():
    return [lambda diagram: persistent_entropy(diagram[0])[0], lambda diagram: persistent_entropy(diagram[1])[0]]
