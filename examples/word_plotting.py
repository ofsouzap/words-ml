import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from os.path import join as joinpath
from saved_models import load_matrix, load_word_indexes, ModelFilenames


def main():

    data = load_matrix(ModelFilenames.PCA_WORD_REL_POS_UNSIGNED_FILENAME)
    word_indexes = load_word_indexes(ModelFilenames.WORD_REL_POS_UNSIGNED_WORD_INDEXES_FILENAME)

    fig, ax = plt.subplots(1,1)

    ax.scatter(data[:,0], data[:,1])

    for i in islice(word_indexes.iterate_from(), 100):
        word = word_indexes.get_from(i)
        ax.annotate(word, (data[i,0],data[i,1]))

    plt.show()


if __name__ == "__main__":
    main()
