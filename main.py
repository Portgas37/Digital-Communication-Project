# ############################################################################
# client.py for PDC 2024
# =========
# Original Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# Current version: Adway Girish
# ############################################################################

"""
Black-box channel simulator. (client)

Instructions
------------
python3 client.py --input_file=[FILENAME] --output_file=[FILENAME] --srv_hostname=[HOSTNAME] --srv_port=[PORT]
python client.py --input_file input.txt --output_file output.txt --srv_hostname=iscsrv72.epfl.ch --srv_port=80
"""

import argparse
import pathlib
import socket

import numpy as np

import channel_helper as ch
import re
import random
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="COM-302 black-box channel simulator. (client)",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="To enforce efficient communication schemes, transmissions are limited to 5e5 samples.")

    parser.add_argument('--input_file', type=str, required=True,
                        help='.txt file containing (N_sample,) rows of float samples.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='.txt file to which channel output is saved.')
    parser.add_argument('--srv_hostname', type=str, required=True,
                        help='Server IP address.')
    parser.add_argument('--srv_port', type=int, required=True,
                        help='Server port.')

    args = parser.parse_args()

    args.input_file = pathlib.Path(args.input_file).resolve(strict=True)
    if not (args.input_file.is_file() and
            (args.input_file.suffix == '.txt')):
        raise ValueError('Parameter[input_file] is not a .txt file.')

    args.output_file = pathlib.Path(args.output_file).resolve(strict=False)
    if not (args.output_file.suffix == '.txt'):
        raise ValueError('Parameter[output_file] is not a .txt file.')

    return args


def channel(x):
    energy = np.sum(np.square(np.abs(x)))
    print("Energy of input", energy)
    n = x.size
    B = random.choice([0, 1])
    sigma_sq = 25
    Z = np.random.normal(0, np.sqrt(sigma_sq), (2*n))
    X = np.zeros(2*n)
    if B == 1:
        X[0:n] = x
    else:
        X[n:2*n] = x
    Y = X + Z
    Y = np.reshape(Y, (-1))
    return Y


def transmitter0(text):
    """
    Bit by Bit transmitter => impossible to distinguish from noise
    Average energy = 120
    """
    n = 64
    k = 6
    codebook = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', ' ']
    bits = [f"{x[0]}{x[1]}{x[2]}{x[3]}{x[4]}{x[5]}" for x in itertools.product([0, 1], repeat=k)]
    letter2bit = dict(map(lambda i,j : (i,j) , codebook, bits))
    X = map(list, map(letter2bit.get, text))
    X = np.array(list(map(int, itertools.chain(*X))))
    return X

def transmitter1(text):
    """
    Letter by Letter transmitter => No error
    Average energy = 40960
    """
    n = 64
    k = 6
    codebook = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    bits = [f"{x[0]}{x[1]}{x[2]}{x[3]}{x[4]}{x[5]}" for x in itertools.product([0, 1], repeat=k)]
    letter2bit = dict(map(lambda i,j : (i,j) , codebook, bits))
    X = list(map(lambda x: int(x, 2), map(letter2bit.get, text)))
    return np.array(X)

def transmitter2(text):
    """
    2 Letter by 2 Letter transmitter.
    n = 64^2 // 2
    len(codewords) = 64^2
    Pr(error_msg) < 10^-1
    """
    nb_letter = 2
    n = 64**nb_letter
    k = 12
    A = 3.056
    text_chunk = re.findall('..', text)
    if len(text) % nb_letter != 0:
        text_chunk.append(text[-(len(text) % nb_letter)].ljust(nb_letter))

    codebook = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    combs = list(map("".join, itertools.product(codebook, repeat=nb_letter)))

    energy_per_letters = 25*A*k
    X = np.zeros(len(combs)//2)
    index = combs.index(text_chunk[0])

    if index >= len(combs)//2:
        index -= len(combs)//2
        value = -np.sqrt(energy_per_letters)
    else:
        value = np.sqrt(energy_per_letters)
    np.put(X, index, value)

    for i in text_chunk[1:]:

        index = combs.index(i)
        if index >= len(combs)//2:
            index -= len(combs)//2
            value = -np.sqrt(energy_per_letters)
        else:
            value = np.sqrt(energy_per_letters)

        zero = np.zeros(len(combs)//2)
        np.put(zero, index, value)
        X = np.concatenate((X, zero))

    return np.array(X)

def receiver2(Y):
    nb_letter = 2
    n = 64**nb_letter
    codebook = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    combs = list(map("".join, itertools.product(codebook, repeat=nb_letter)))
    received_split = np.reshape(Y, (-1, n//2))

    resul = []
    for i in received_split:
        maximum_index = np.argmax(np.abs(i))
        if i[maximum_index] < 0:
            maximum_index += len(combs)//2
        if maximum_index >= len(combs):
            print("Not in this half")
        else:
            resul.append(combs[maximum_index])

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
        return "".join(resul[:len(resul)//2])[:40]
    return "".join(resul[len(resul)//2:])[:40]

def transmitter3(text):
    """
    3 Letter by 3 Letter transmitter.
    n = 64^3 // 2
    len(codewords) = 64^3
    k = 19
    Average msg energy = 33700
    sqrt(3letter_energy) = 50
    P(errror on 3 letters) < 10^-3
    len(X) = n(40 + nb_letter - 40 % nb_letter)/nb_letter ; For 3 : 1835008
    """
    nb_letter = 3
    n = 64**nb_letter
    k = 19
    A = 5.31987
    text_chunk = re.findall('...', text)
    if len(text) % nb_letter != 0:
        text_chunk.append(text[-(len(text) % nb_letter)].ljust(nb_letter))

    codebook = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    combs = list(map("".join, itertools.product(codebook, repeat=nb_letter)))

    energy_per_letters = 25*A*k
    X = np.zeros(len(combs)//2)
    index = combs.index(text_chunk[0])

    if index >= len(combs)//2:
        index -= len(combs)//2
        value = -np.sqrt(energy_per_letters)
    else:
        value = np.sqrt(energy_per_letters)
    np.put(X, index, value)

    for i in text_chunk[1:]:

        index = combs.index(i)
        if index >= len(combs)//2:
            index -= len(combs)//2
            value = -np.sqrt(energy_per_letters)
        else:
            value = np.sqrt(energy_per_letters)

        zero = np.zeros(len(combs)//2)
        np.put(zero, index, value)
        X = np.concatenate((X, zero))

    return np.array(X)

def receiver3(Y):
    nb_letter = 3
    n = 64**nb_letter
    codebook = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    combs = list(map("".join, itertools.product(codebook, repeat=nb_letter)))
    received_split = np.reshape(Y, (-1, n//2))

    resul = []
    for i in received_split:
        maximum_index = np.argmax(np.abs(i))
        if i[maximum_index] < 0:
            maximum_index += len(combs)//2
        if maximum_index >= len(combs):
            print("Not in this half")
        else:
            resul.append(combs[maximum_index])

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
        return "".join(resul[:len(resul)//2])[:40]
    return "".join(resul[len(resul)//2:])[:40]

def find_min_nb_partitions(s, combs):
    """ Greedy but is optimal for substrings of length 2 and 3"""
    resul = []
    i = 0
    while i < len(s):
        if len(s) - i > 1:
            if i + 2 < len(s) and f"{s[i]}{s[i+1]}{s[i+2]}" in combs:
                resul.append(combs.index(f"{s[i]}{s[i+1]}{s[i+2]}"))
                i += 3
            else:
                resul.append(combs.index(f"{s[i]}{s[i+1]}"))
                i += 2
        else:
            resul.append(combs.index(f"{s[i]} "))
            i += 2
    return resul

def famous_substrings_of_length_3(filename, top_n):
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
   
def transmitter4(text, combs):
    """
    2/3 Letter by 2/3 Letter transmitter.
    n = 64^2.601606 // 2 => worst case length of input is 500000 (40*n/2)
    len(codewords) = 64^2.601606
    k = ln(2n)/ln(2) = 15.60936
    Average msg energy = 29836
    sqrt(3letter_energy) = 42.94
    P(errror on 2/3 letters) < ???
    P(error on msg) < 10^-2
    len(X) = n(40 + nb_letter - 40 % nb_letter)/nb_letter ; For 2.601606 : depends on input (but lower than 500000 and bigger than 350000)
    """
    nb_codewords = 64**2.601606
    nb_letter = 2
    n = 64**nb_letter
    k = 15.609636
    A = 3.1

    index = find_min_nb_partitions(text, combs)

    energy_per_letters = 25*A*k

    X = np.zeros(len(combs)//2)
    ind = index[0]
    if ind >= len(combs)//2:
        ind -= len(combs)//2
        value = -np.sqrt(energy_per_letters)
    else:
        value = np.sqrt(energy_per_letters)
    np.put(X, ind, value)

    for ind in index[1:]:
        if ind >= len(combs)//2:
            ind -= len(combs)//2
            value = -np.sqrt(energy_per_letters)
        else:
            value = np.sqrt(energy_per_letters)

        zero = np.zeros(len(combs)//2)
        np.put(zero, ind, value)
        X = np.concatenate((X, zero))

    return np.array(X)

from scipy.stats import normaltest

def receiver4(Y, combs):
    nb_letter = 2
    n = 64**nb_letter

    received_split = np.reshape(Y, (-1, len(combs)//2))

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

    resul = []
    possible_correction = []
    for ind, i in enumerate(received_split):
        maximum_index = np.argmax(np.abs(i))
        if i[maximum_index] < 0:
            maximum_index += len(combs)//2

        s = np.sort(np.abs(i))[-2:]
        arg = np.argsort(np.abs(i))[-2:]
        tol = 5
        print(s)
        if s[1] - s[0] <= tol:
            index1 = arg[0]
            if i[index1] < 0:
                index1 += len(combs)//2 
            if len(combs[maximum_index]) > len(combs[index1]):
                possible_correction.append((ind, maximum_index, index1, s[1] - s[0]))
            elif len(combs[maximum_index]) < len(combs[index1]):
                possible_correction.append((ind, index1, maximum_index, -s[1] + s[0]))
            else:
                s = np.sort(np.abs(i))[-3:]
                arg = np.argsort(np.abs(i))[-3:]
                print(s)
                if s[2] - s[0] <= tol:
                    index1 = arg[0]
                    if i[index1] < 0:
                        index1 += len(combs)//2 
                    if len(combs[maximum_index]) > len(combs[index1]):
                        possible_correction.append((ind, maximum_index, index1, s[2] - s[0])) 
                    elif len(combs[maximum_index]) < len(combs[index1]):
                        possible_correction.append((ind, index1, maximum_index, -s[2] + s[0]))
                    else:
                        resul.append(combs[maximum_index])
                else:
                    resul.append(combs[maximum_index])
        else:
            resul.append(combs[maximum_index])

    if possible_correction:
        print(possible_correction)
        nb_found_char = sum([len(x) for x in resul])
        print(nb_found_char)
        lower_bound = 2*len(possible_correction)
        upper_bound = 3*len(possible_correction)
        if nb_found_char + lower_bound == 41 and (resul[-1][-1] == " " or len(resul) - 1 in [x[0] for x in possible_correction]):
            print(41)
            for index, long_index, short_index, _ in possible_correction:
                resul.insert(index, combs[short_index])
        elif nb_found_char + upper_bound == 40:
            print(40)
            for index, long_index, short_index, _ in possible_correction:
                resul.insert(index, combs[long_index])
        else: 
            nb_found_3 = len([x for x in resul if len(x) == 3])
            nb_found_2 = len([x for x in resul if len(x) == 2])
            nb_3 = (20 - len(received_split)) * 2 
            nb_2 = len(received_split) - nb_3
            empiric_comb = [3] * (nb_3 - nb_found_3) + [2] * (nb_2 - nb_found_2)
            # empiric_comb_2 = [3] * (nb_3 + 1 - nb_found_3) + [2] * (nb_2 - 1 - nb_found_2)

            # print(nb_3, nb_2, nb_found_3, nb_found_2)
            # if len(empiric_comb) != len(empiric_comb_2) or len(empiric_comb) != len(possible_correction):
            #     for index, long_index, short_index, _ in possible_correction:
            #         resul.insert(index, combs[long_index])
            # else:
            #     print(empiric_comb, empiric_comb_2, len(received_split))
            #     losses = []
            #     if resul[-1][-1] == " " or len(resul) - 1 in [x[0] for x in possible_correction]: 
            #         permutations = set(list(itertools.permutations(empiric_comb)) + list(itertools.permutations(empiric_comb_2)))
            #     else:
            #         permutations = set(list(itertools.permutations(empiric_comb)))
            #     for comb in permutations:
            #         index_comb = 0
            #         loss = 0
            #         for index, long_index, short_index, diff in possible_correction:
            #             if comb[index_comb] == 3:
            #                 loss -= diff
            #             else:
            #                 loss += diff
            #             index_comb += 1
            #         losses.append((loss, comb))
            #     best_comb = min(losses, key = lambda x : x[0])[1]
            #     print(best_comb, losses)
            #     index_comb = 0
            #     for index, long_index, short_index, _ in possible_correction:
            #         if best_comb[index_comb] == 3:
            #             print(combs[long_index])
            #             resul.insert(index, combs[long_index])
            #         else:
            #             print(combs[short_index])
            #             resul.insert(index, combs[short_index])
            #         index_comb += 1

            index_comb = 0
            for index, long_index, short_index, _ in possible_correction:
                if empiric_comb[index_comb] == 3:
                    resul.insert(index, combs[long_index])
                else:
                    resul.insert(index, combs[short_index])
                index_comb += 1

    return "".join(resul)[:40]

nb_letter = 2
nb_codewords = 64**2.601606
codebook = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
combs = list(map("".join, itertools.product(codebook, repeat=nb_letter)))
input_filename = 'C:\\Users\\luka_\\OneDrive\\Bureau\\epfl\BA6\\Principles of Digital Communications\\PDC-project\\words.txt'
combs2 = [x for x in famous_substrings_of_length_3(input_filename, round(nb_codewords - len(combs)))]
combs = combs2 + combs
combs2_set = set(combs2)
combs3 = [x for x in map("".join, itertools.product(codebook, repeat=nb_letter+1)) if x not in combs2_set][:round(nb_codewords - len(combs))]
combs = combs3 + combs
combs.sort()
codebook = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '.', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

# text = "1930s when Mick Leahy and a party of men travelled from what later "[:40]
# text = "".join(random.sample(codebook, 40))
# print(text)
# X = transmitter4(text, combs)
# print("Length of input", len(X))

# noisy_input = channel(X)
# print(receiver4(noisy_input, combs), receiver4(noisy_input, combs) == text, text)

failure = 0
n = 10000
energy = 0
for i in range(n):
    text = "1930s when Mick Leahy and a party of men travelled from what later "[:40]
    text = "".join(random.sample(codebook, 40))
    X = transmitter4(text, combs)
    energy += np.sum(np.square(np.abs(X)))
    noisy_input = channel(X)
    if receiver4(noisy_input, combs) != text:
        failure += 1
print(failure*100/n) # 6.29 (human text at A = 3.1) 4.7 (random at A = 3.1)
print(energy/n) # 16936 (human text at A = 3.1) 22693 (random at A = 3.1)

# if __name__ == '__main__':
#     version_cl = b'dUS'  # Always length-3 alphanumeric
#     text = "IDHSlKbjfk3zxZBPw9ANUv1MOLCqa5QGroTu08cR"
#     X = transmitter4(text)
#     np.savetxt("input.txt", X)
#     args = parse_args()
#     tx_p_signal = np.loadtxt(args.input_file)

#     N_sample = tx_p_signal.size
#     if not ((tx_p_signal.shape == (N_sample,)) and
#             np.issubdtype(tx_p_signal.dtype, np.floating)):
#         raise ValueError('Parameter[input_file] must contain a real-valued sequence.')

#     if N_sample > 500000:
#         raise ValueError(('Parameter[input_file] contains more than 500,000 samples. '
#                           'Design a more efficient communication system.'))

#     energy = np.sum(np.square(np.abs(tx_p_signal)))
#     if energy > 40960:
#         raise ValueError(('Energy of the signal exceeds the limit 40960. '
#                           'Design a more efficient communication system.'))

#     with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as sock_cl:
#         sock_cl.connect((args.srv_hostname, args.srv_port))

#         tx_header = b'0' + version_cl
#         ch.send_msg(sock_cl, tx_header, tx_p_signal)

#         rx_header, rx_data = ch.recv_msg(sock_cl)
#         if rx_header[:1] == b'0':  # Data
#             np.savetxt(args.output_file, rx_data)
#         elif rx_header[:1] == b'1':  # Rate limit
#             raise Exception(rx_data.tobytes())
#         elif rx_header[:1] == b'2':  # Outdated version of client.py
#             raise Exception(rx_data.tobytes())
#         else:  # Unknown header
#             err_msg = f'Unknown header: {rx_header}'
#             raise Exception(err_msg)

#         Y = np.loadtxt("output.txt")
#         print(receiver4(Y))
