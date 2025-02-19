import itertools
import numpy as np
import re
import random

def famous_substrings_of_length_3(filename, top_n):
    """
    Returns the topn substrings of length 3 in number of occurences in a file containing english wording
    """
    substring_counts = {}

    with open(filename, 'r') as file:
        content = file.read()

    for i in range(len(content) - 2):
        substring = content[i:i+3]
        if substring in substring_counts:
            substring_counts[substring] += 1
        else:
            substring_counts[substring] = 1

    substrings_sort = sorted(substring_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return list(set([re.sub(r'[^a-zA-Z0-9 .]', " ", x[0]) for x in substrings_sort]))

def find_min_nb_partitions(s, codebook):
    """ 
    Returns a list of indices representing a full partition of the string s into elements of the codebook
    Greedy but is optimal for substrings of length 2 and 3
    """
    resul = []
    i = 0
    while i < len(s):
        if len(s) - i > 1:
            if i + 2 < len(s) and f"{s[i]}{s[i+1]}{s[i+2]}" in codebook:
                resul.append(codebook.index(f"{s[i]}{s[i+1]}{s[i+2]}"))
                i += 3
            else:
                resul.append(codebook.index(f"{s[i]}{s[i+1]}"))
                i += 2
        else:
            resul.append(codebook.index(f"{s[i]} "))
            i += 2
    return resul

def transmitter(text):
    """
    2/3 Letter by 2/3 Letter transmitter.
    n = 64^2.601606 // 2 => worst case length of input is 500000 (40*n/2)
    len(codewords) = 64^2.601606
    k = ln(2n)/ln(2) = 15.60936
    Average msg energy = 21000
    sqrt(3letter_energy) = 37
    P(errror on 2/3 letters) < ???
    P(error on msg) < 0.05
    len(X) = n(40 + nb_letter - 40 % nb_letter)/nb_letter ; For 2.601606 : depends on input (but lower than 500000 and bigger than 350000)
    """
    nb_codewords = 64**2.601606
    nb_letter = 2
    n = 64**nb_letter
    k = 15.609636
    A = 3.187
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
    combinations_len_2 = list(map("".join, itertools.product(characters, repeat=nb_letter)))
    input_filename = 'words.txt'
    combinations_recuring_3 = famous_substrings_of_length_3(input_filename, round(nb_codewords - len(combinations_len_2)))
    combinations_len_2_3 = combinations_recuring_3 + combinations_len_2
    combinations_recuring_3_set = set(combinations_recuring_3)
    combinations_len_3 = [x for x in map("".join, itertools.product(characters, repeat=nb_letter+1)) if x not in combinations_recuring_3_set][:round(nb_codewords - len(combinations_len_2_3))]
    codebook = combinations_len_3 + combinations_len_2_3
    codebook.sort()

    indices = find_min_nb_partitions(text, codebook)

    energy_per_letters = 25*A*k

    X = np.array([])

    for index in indices:
        if index >= len(codebook)//2:
            index -= len(codebook)//2
            value = -np.sqrt(energy_per_letters)
        else:
            value = np.sqrt(energy_per_letters)

        c_i = np.zeros(len(codebook)//2)
        np.put(c_i, index, value)
        X = np.concatenate((X, c_i))

    return np.array(X)

if __name__ == '__main__':
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
    text = "".join(random.sample(characters, 40))
    text = "1930s when Mick Leahy and a party of men travelled from what later "[:40]
    print(text)
    X = transmitter(text)
    print("Length of X", len(X))
    print("Energy of X", np.sum(np.square(np.abs(X))))
    np.savetxt("input.txt", X)