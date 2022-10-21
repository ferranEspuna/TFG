from persim import persistent_entropy


def get_all_summaries():
    return [lambda diag: persistent_entropy(diag[0])[0], lambda diag: persistent_entropy(diag[1])[0]]
