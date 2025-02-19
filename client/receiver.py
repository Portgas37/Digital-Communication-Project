import itertools
import numpy as np
from scipy.stats import normaltest
import re

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

def receiver(Y):
    nb_letter = 2
    n = 64**nb_letter
    nb_letter = 2
    nb_codewords = 64**2.601606
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
    combinations_len_2 = list(map("".join, itertools.product(characters, repeat=nb_letter)))
    input_filename = 'words.txt'
    combinations_recuring_3 = famous_substrings_of_length_3(input_filename, round(nb_codewords - len(combinations_len_2)))
    combinations_len_2_3 = combinations_recuring_3 + combinations_len_2
    combinations_recuring_3_set = set(combinations_recuring_3)
    combinations_len_3 = [x for x in map("".join, itertools.product(characters, repeat=nb_letter+1)) if x not in combinations_recuring_3_set][:round(nb_codewords - len(combinations_len_2_3))]
    codebook = combinations_len_3 + combinations_len_2_3
    codebook.sort()

    received_split = np.reshape(Y, (-1, len(codebook)//2))
    # ----------------- Search for the correct half ------------------------
    first_half_shap = 0
    second_half_shap = 0
    for i in received_split[:len(received_split)//2]:
        if normaltest(i,)[1] >= 0.05:
            first_half_shap += 1
        else:
            first_half_shap -= 1
    for i in received_split[len(received_split)//2:]:
        if normaltest(i,)[1] >= 0.05:
            second_half_shap += 1
        else:
            second_half_shap -= 1

    if first_half_shap < second_half_shap:
        received_split = received_split[:len(received_split)//2]
    else:
        received_split = received_split[len(received_split)//2:]

    # ------------------- Decoding process -----------------
    resul = []

    # This list contains indices of c_i's that have the 2 or 3 highest value close to the tolerance 
    # if the corresponding strings are of different length and therefore they are prone to errors. 
    # We will then try and find the correct ones by knowing our string final can only have length 40
    # or 41 and a ' ' as last character.
    possible_corrections = [] 
    for ind, i in enumerate(received_split):
        maximum_index = np.argmax(np.abs(i))
        if i[maximum_index] < 0:
            maximum_index += len(codebook)//2

        s = np.sort(np.abs(i))[-2:]
        arg = np.argsort(np.abs(i))[-2:]
        tol = 5
        if s[1] - s[0] <= tol:
            index1 = arg[0]
            if i[index1] < 0:
                index1 += len(codebook)//2 
            if len(codebook[maximum_index]) > len(codebook[index1]):
                possible_corrections.append((ind, maximum_index, index1, s[1] - s[0]))
            elif len(codebook[maximum_index]) < len(codebook[index1]):
                possible_corrections.append((ind, index1, maximum_index, -s[1] + s[0]))
            else:
                s = np.sort(np.abs(i))[-3:]
                arg = np.argsort(np.abs(i))[-3:]
                if s[2] - s[0] <= tol:
                    index1 = arg[0]
                    if i[index1] < 0:
                        index1 += len(codebook)//2 
                    if len(codebook[maximum_index]) > len(codebook[index1]):
                        possible_corrections.append((ind, maximum_index, index1, s[2] - s[0])) 
                    elif len(codebook[maximum_index]) < len(codebook[index1]):
                        possible_corrections.append((ind, index1, maximum_index, -s[2] + s[0]))
                    else:
                        resul.append(codebook[maximum_index])
                else:
                    resul.append(codebook[maximum_index])
        else:
            resul.append(codebook[maximum_index])

    if possible_corrections:
        nb_found_char = sum([len(x) for x in resul])
        lower_bound = 2*len(possible_corrections)
        upper_bound = 3*len(possible_corrections)
        if nb_found_char + lower_bound == 41 and (resul[-1][-1] == " " or len(resul) - 1 in [x[0] for x in possible_corrections]):
            for index, long_index, short_index, _ in possible_corrections:
                resul.insert(index, codebook[short_index])
        elif nb_found_char + upper_bound == 40:
            for index, long_index, short_index, _ in possible_corrections:
                resul.insert(index, codebook[long_index])
        else: 
            nb_found_3 = len([x for x in resul if len(x) == 3])
            nb_found_2 = len([x for x in resul if len(x) == 2])
            nb_3 = (20 - len(received_split)) * 2 
            nb_2 = len(received_split) - nb_3
            empiric_combination = [3] * (nb_3 - nb_found_3) + [2] * (nb_2 - nb_found_2)

            index_comb = 0
            for index, long_index, short_index, _ in possible_corrections:
                if empiric_combination[index_comb] == 3:
                    resul.insert(index, codebook[long_index])
                else:
                    resul.insert(index, codebook[short_index])
                index_comb += 1

    return "".join(resul)[:40]

if __name__ == '__main__':
    Y = np.loadtxt("output.txt")
    print(receiver(Y))
