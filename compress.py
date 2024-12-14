"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq = {}

    # for all bytes in the text, if the int is already in the dictionary,
    # increment its value by one, if it is not in the dictionary,
    # create a new key-value pair.
    for i in text:
        if i in freq:
            freq[i] = freq[i] + 1
        else:
            freq[i] = 1
    return freq


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    >>> freq = build_frequency_dict(bytes([65, 66, 67, 68, 68, 69, 69, 69,\
     70, 70, 70, 70, 70, 70]))
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None,\
    HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> t == result
    True
    """

    keys = list(freq_dict.keys())
    values = list(freq_dict.values())
    # Corner case for if the frequency dictionary has only one key-value pair
    if len(keys) == 1:
        dummy = (keys[0] + 1) % 256
        return HuffmanTree(None, HuffmanTree(keys[0]), HuffmanTree(dummy))
    sym = []
    freq = []
    for i in range(len(keys)):
        sym.append(keys[i])
        freq.append(values[i])
    # Use the huffman algorithm provided on the assignment page. Get the two
    # lowest frequencies and make them into a huffman tree. Then add the new
    # huffman tree into the sym list and the combined frequency of the two
    # "symbol/huffman tree" into the freq list, both in the last index position.
    while len(sym) > 1:
        lowest = freq.index(min(freq))
        sym1, freq1 = sym.pop(lowest), freq.pop(lowest)
        lowest = freq.index(min(freq))
        sym2, freq2 = sym.pop(lowest), freq.pop(lowest)
        if not isinstance(sym1, HuffmanTree):
            sym1 = HuffmanTree(sym1)
        if not isinstance(sym2, HuffmanTree):
            sym2 = HuffmanTree(sym2)
        new_tree = HuffmanTree(None, sym1, sym2)
        sym.append(new_tree)
        freq.append(freq1 + freq2)
    return sym[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> result = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None,\
    HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> d = get_codes(result)
    >>> d == {65: "1100", 66: "1101", 67: "100", 68: "101", 69: "111", 70: "0"}
    True
    >>> d = get_codes(HuffmanTree(None, HuffmanTree(1), HuffmanTree(3)))
    >>> d == {1: "0", 3: "1"}
    True
    >>> get_codes(HuffmanTree(None)) == {}
    True
    """
    # If the tree is a leaf and the symbol is empty, then it is an empty tree,
    # and so return an empty dictionary. If it is not empty, then return a
    # dictionary with the key as the symbol and and empty string for the value.
    if tree.is_leaf():
        if tree.symbol is not None:
            return {tree.symbol: ""}
        return {}
    else:
        # Run a recursion statement, so get the codes for the left tree and the
        # right tree, and get the dictionary. For every key-value pair in the
        # left tree, add a 0 to the front and 1 for every in the right tree
        # and return a combined dictionary.
        left = get_codes(tree.left)
        right = get_codes(tree.right)
        for key in left.keys():
            left[key] = "0" + left.get(key)
        for key in right.keys():
            right[key] = "1" + right.get(key)
        left.update(right)
        return left


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> tree = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None,\
    HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> number_nodes(tree)
    >>> tree.right.right.number == 2
    True
    """
    # Take every tree from the list, since its postorder and name it in order
    # using a for loop.
    lst = post_order(tree)
    for i in range(len(lst)):
        lst[i].number = i


def post_order(tree: HuffmanTree) -> list[HuffmanTree]:
    """ Return all the trees within the HuffmanTree is postorder as an array

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = post_order(tree)
    >>> d == [HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))]
    True
    >>> tree = HuffmanTree(None, HuffmanTree(2), None)
    >>> d = post_order(tree)
    >>> d == [HuffmanTree(None, HuffmanTree(2), None)]
    True
    """
    # Helper function for number_nodes()
    # Base case, if tree is a leaf
    if tree.is_leaf():
        return []
    # If the left or right branch of tree is empty, then return the recursive
    # call of the other tree and add the current tree.
    elif not tree.left:
        lst = post_order(tree.right)
    elif not tree.right:
        lst = post_order(tree.left)
    # If both sides of the tree are not empty, then return both trees, in the
    # order, left tree then right tree
    else:
        lst = post_order(tree.left)
        lst.extend(post_order(tree.right))
    lst.append(tree)
    return lst


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    >>> result = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
        HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None, \
        HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> freq_dic = {65: 1, 66: 1, 67: 1, 68: 2, 69: 3, 70: 6}
    >>> avg_length(result, freq_dic)
    2.2857142857142856
    """
    # Get the total frequency and the total bits * frequency and divide
    codes = get_codes(tree)
    total = 0
    freq = 0
    for key in codes.keys():
        total += len(codes.get(key)) * freq_dict.get(key)
        freq += freq_dict.get(key)
    return total / freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> text = bytes([1, 0, 2, 1, 1, 0, 2, 1, 2, 1])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10011101', '00111011', '10000000']
    >>> text = b'\x3f'
    >>> freq = build_frequency_dict(text)
    >>> tree = build_huffman_tree(freq)
    >>> codes = get_codes(tree)
    >>> result = compress_bytes(text, codes)
    >>> result == bytes([0])
    True
    """

    # Get all the bytes in text by their respective codes, and combine into
    # one string. If the length will not suffice mod 8 = 0, then add 0's till it
    # will be enough.
    s = ""
    for i in text:
        s += str(codes.get(i))

    while len(s) % 8 != 0:
        s += "0"

    lst = []

    # https://www.techiedelight.com/convert-binary-string-to-integer-python/
    # Using the link above, convert every 8bit of binary string into an integer
    # and put it into a list, then convert the integer list into a bytes object.
    for i in range(0, len(s), 8):
        lst.append(int(s[i: i + 8], 2))
    return bytes(lst)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> tree = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None,\
    HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 67, 0, 68, 0, 65, 0, 66, 1, 1, 0, 69, 1, 0, 1, 2, 0, 70, 1, 3]
    """
    byte = []
    post = post_order(tree)
    # We want to return the nodes in postorder, therefore we shall get every
    # tree in the list from postorder() and then convert it into a 4 byte number
    # if the left is a leaf, then we add 0 and the left.symbol, if not then we
    # add 1 and the left.number and we do the same for the right tree.
    for subtree in post:
        if subtree.left.is_leaf():
            byte.append(0)
            byte.append(subtree.left.symbol)
        else:
            byte.append(1)
            byte.append(subtree.left.number)
        if subtree.right.is_leaf():
            byte.append(0)
            byte.append(subtree.right.symbol)
        else:
            byte.append(1)
            byte.append(subtree.right.number)
    return bytes(byte)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(1, 2, 1, 0), \
    ReadNode(0, 100, 0, 2)]
    >>> generate_tree_general(lst, 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(100, None, None), \
HuffmanTree(2, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(0, 67, 0, 68), ReadNode(0, 65, 0, 66),\
     ReadNode(1, 1, 0, 69), ReadNode(1, 0, 1, 2), ReadNode(0, 70, 1, 3)]
    >>> generate_tree_general(lst, 4)
    HuffmanTree(None, HuffmanTree(70, None, None), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(67, None, None), HuffmanTree(68, None, None)), \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(65, None, None), \
HuffmanTree(66, None, None)), HuffmanTree(69, None, None))))
    """
    # Base case for recursive call
    if not node_lst:
        return HuffmanTree(None)

    # If node.l_type at position root_index is 0, then make a tree with symbol
    # node.l_data, if it is 1, then use recursion to make a tree at index
    # node.l_data, same goes for the right tree. Then combine these two trees
    # and return the tree.
    node = node_lst[root_index]
    if node.l_type == 0:
        left_tree = HuffmanTree(node.l_data)
    else:
        left_tree = generate_tree_general(node_lst, node.l_data)
    if node.r_type == 0:
        right_tree = HuffmanTree(node.r_data)
    else:
        right_tree = generate_tree_general(node_lst, node.r_data)

    return HuffmanTree(None, left_tree, right_tree)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst = [ReadNode(0, 67, 0, 68), ReadNode(0, 65, 0, 66),\
     ReadNode(1, 1, 0, 69), ReadNode(1, 0, 1, 2), ReadNode(0, 70, 1, 3)]
    >>> generate_tree_postorder(lst, 4)
    HuffmanTree(None, HuffmanTree(70, None, None), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(67, None, None), HuffmanTree(68, None, None)), \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(65, None, None), \
HuffmanTree(66, None, None)), HuffmanTree(69, None, None))))
    """
    # Base case for recursive call
    if not node_lst:
        return HuffmanTree(None)
    node = node_lst[root_index]

    # Check right then left, since the index for the left tree will be at the
    # position root_index - (total nodes in right tree + 1). The algorithm is
    # very similar to generate_tree_general, however when we are generating the
    # right tree, we use root_index - 1 and for the left, we use the statement
    # above, using the count_nodes function.
    if node.r_type == 0:
        right_tree = HuffmanTree(node.r_data)
    else:
        right_tree = generate_tree_postorder(node_lst, root_index - 1)
    if node.l_type == 0:
        left_tree = HuffmanTree(node.l_data)
    else:
        left_tree_index = root_index - (count_nodes(right_tree) + 1)
        left_tree = generate_tree_postorder(node_lst, left_tree_index)

    return HuffmanTree(None, left_tree, right_tree)


def count_nodes(tree: HuffmanTree) -> int:
    """ Return the number of internal nodes of a tree.
    >>> tree = HuffmanTree(None, HuffmanTree(70), HuffmanTree(None, \
    HuffmanTree(None, HuffmanTree(67), HuffmanTree(68)), HuffmanTree(None,\
    HuffmanTree(None, HuffmanTree(65), HuffmanTree(66)), HuffmanTree(69))))
    >>> count_nodes(tree)
    5
    >>> count_nodes(tree.left)
    0
    >>> count_nodes(tree.right)
    4
    """
    # Helper function for generate_tree_postorder()
    # if tree is a leaf then return 0, since there are no nodes, if not then
    # use recursion to find the number of nodes in the left and right trees and
    # add 1 to the total.
    if tree.is_leaf():
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    >>> b = bytes([110, 95, 226, 85])
    >>> tree = build_huffman_tree(build_frequency_dict(b))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b, get_codes(tree)), len(b))
    b'n_\xe2U'
    """

    # https://mkyong.com/python/python-how-to-convert-int-to-a-binary-string/
    # Convert all the bytes in text to 8bit binary strings and combine them into
    # one string.
    b = ""
    for byte in text:
        b += "{0:b}".format(byte).zfill(8)

    # Switch the key and value pairs to value and key pairs for easy accessing
    codes = get_codes(tree)
    code_symbol = {}
    for key in codes:
        code_symbol[codes[key]] = key

    # Check every index of the string b, and add it to the code string. If the
    # code string is present in the code_symbol dictionary as a key, then add
    # the value that is corresponding to the code in the dictionary to the list
    # and reset code to "". Then return the byte format of result up to the
    # length that is equal to size.
    code = ""
    result = []
    for byte in b:
        code += byte
        if code in code_symbol:
            result.append(code_symbol[code])
            code = ""
    return bytes(result[:size])


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))

# ====================
# Other functions


def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # Find the biggest frequency in the dictionary and pop it from both the keys
    # and values list. Then if the number at position i in the pre_order list
    # is in the list keys, then switch the two symbols. This is to prevent
    # switching a symbol that has already been put into place. This algorithm is
    # similar to selection sort.
    keys = list(freq_dict.keys())
    values = list(freq_dict.values())
    pre_order = leaf_pre_order(tree)
    for i in range(len(freq_dict) - 1):
        index_symbol = values.index(max(values))
        values.pop(index_symbol)
        symbol = keys.pop(index_symbol)
        if pre_order[i] in keys:
            swap_symbols(tree, symbol, pre_order[i])


def leaf_pre_order(tree: HuffmanTree) -> list:
    """ Returns all the leaf nodes in the tree in preorder in list.
    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> leaf_pre_order(tree)
    [99, 100, 101, 97, 98]
    """
    # Helper function for improve_tree()
    # If the tree is a leaf, then return the tree's symbol as a list. If not
    # then use recursion to get left and right tree's preorder of symbols and
    # add the two lists together.
    if tree.is_leaf():
        return [tree.symbol]
    lst = leaf_pre_order(tree.left)
    lst += leaf_pre_order(tree.right)
    return lst


def swap_symbols(tree: HuffmanTree, symbol1: int, symbol2: int) \
        -> None:
    """ Swaps the tree2 and tree3 in tree and returns nothing.

    Precondition: tree2 and tree3 are in tree.
    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> swap_symbols(tree, 99, 97)
    >>> tree.left.left.symbol
    97
    >>> tree.right.right.left.symbol
    99
    """
    # Helper function for improve_tree()
    # If the tree is a leaf and is either one of the two symbols provided, then
    # switch the symbol to the other symbol. If the tree is not a leaf, then use
    # recursive calls on both the left and right trees to swap symbols.
    if tree.is_leaf():
        if tree.symbol == symbol1:
            tree.symbol = symbol2
        elif tree.symbol == symbol2:
            tree.symbol = symbol1
    else:
        swap_symbols(tree.left, symbol1, symbol2)
        swap_symbols(tree.right, symbol1, symbol2)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
