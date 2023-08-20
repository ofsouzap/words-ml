import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from saved_models import load_matrix, load_word_indexes, ModelFilepaths


def main():

    data = load_matrix(ModelFilepaths.PCA_WORD_REL_POS_UNSIGNED)
    word_indexes = load_word_indexes(ModelFilepaths.WORD_REL_POS_UNSIGNED_WORD_INDEXES)

    fig, ax = plt.subplots(1,1)

    ax.scatter(data[:,0], data[:,1])

    for i in islice(word_indexes.iterate_from(), 1000):
        word = word_indexes.get_from(i)
        ax.annotate(word, (data[i,0],data[i,1]))

    plt.show()


if __name__ == "__main__":
    main()
