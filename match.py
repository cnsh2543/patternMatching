## gen msg, parse tree are modified, raw form check

import re
import os
import json
import sys
import uuid 
from zss import simple_distance, Node
import sympy as sp
from sympy import sympify, Function, Symbol, Number, Integer, Rational, Float, pi, E
from numpy import spacing
import numpy as np
from sympy.parsing.sympy_parser import T as parser_transformations
import Levenshtein
import itertools
from fractions import Fraction
from itertools import permutations
from collections import Counter
from evaluationFunction.expression_utilities import (
    substitute_input_symbols,
    parse_expression,
    create_sympy_parsing_params,
    convert_absolute_notation
)




def lambda_handler(event, context):
    '''Provide an event that contains the following keys:
      - message: contains the text 
    '''
    try:
        inputText = event['commonMistakes']
        processedText = run_all(inputText)

        return {
            'statusCode': 200,
            'body': json.dumps(processedText)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": repr(e)})
        } 
    

## modified from Karl's evaluation function
def check_equality(response, answer, params, eval_response) -> dict:

    if not isinstance(answer, str):
        raise Exception("No answer was given.")
    if not isinstance(response, str):
        return

    answer = answer.strip()
    response = response.strip()
    if len(answer) == 0:
        raise Exception("No answer was given.")
    if len(response) == 0:
        return

    answer, response = substitute_input_symbols([answer, response], params)
    parsing_params = create_sympy_parsing_params(params)
    parsing_params.update({"rationalise": True, "simplify": True})
    parsing_params["extra_transformations"] = parser_transformations[9]  # Add conversion of equal signs

    # Converting absolute value notation to a form that SymPy accepts
    response, response_feedback = convert_absolute_notation(response, "response")

    answer, answer_feedback = convert_absolute_notation(answer, "answer")


    # Safely try to parse answer and response into symbolic expressions
    try:
        res = parse_expression(response, parsing_params)
    except Exception:
        return (sympify('a'), sympify('b'))

    try:
        ans = parse_expression(answer, parsing_params)
    except Exception as e:
        return (sympify('a'), sympify('b'))
    return (res, ans)

## creating node

labelMap = {
    "<class 'sympy.core.mul.Mul'>": '*', 
    "<class 'sympy.core.add.Add'>": "+",
    "<class 'sympy.core.power.Pow'>": "**",
    "<class 'sympy.core.relational.Equality'>": "=",
    "<class 'sympy.core.function.AppliedUndef'>": {
        "NEG": "NEG",
        "DIV": "DIV"
    }
}

class CustomNode:

    counter = 0

    def __init__(self, type, value, isLeaf, parent=None):
        self.type = type
        self.value = value
        self.isLeaf = isLeaf
        self.parent = parent
        self.children = []
        self.name = value + str(CustomNode.counter)
        CustomNode.counter += 1

    def add_child(self, child):
        self.children.append(child)
    
    def set_children(self, children):
        self.children= children

    def delete_child(self, child):
        self.children.remove(child)

    def set_parent(self, parent):
        self.parent = parent

    def set_value(self, value):
        self.value = value
    
    def set_type(self, type):
        self.type = type
    
    def set_isLeaf(self, isLeaf):
        self.isLeaf = isLeaf

    def __repr__(self):
        return (f"Type: {self.type}, value: {self.value}, Is Leaf: {self.isLeaf}, Parent: {self.parent.value if self.parent else None}, Children: {self.children}")

    
    def print_prop(self):
        print(self.type, self.value, self.isLeaf, self.parent.value if self.parent else '', [node.value for node in self.children])


## go through the srepr by sympy and create our own nodes
def recursive_extract(nodes, sym_arg, labelCounterTracker, parent=None):
    numeric_types = (Number, Integer, Rational, Float)
    if isinstance(sym_arg, numeric_types) or sym_arg in [pi,E]:
        if str(sym_arg) not in labelCounterTracker:
            current_node = CustomNode("numeric",str(sym_arg),True, parent)
            labelCounterTracker[str(sym_arg)] = 1
        else:
            current_node = CustomNode("numeric",str(sym_arg) + '|' + str(labelCounterTracker[str(sym_arg)]),True, parent)
            labelCounterTracker[str(sym_arg)] += 1
    elif isinstance(sym_arg, Symbol):
        if str(sym_arg) not in labelCounterTracker:
            if str(sym_arg) in ("pi","epsilon"):
                current_node = CustomNode("numeric",str(sym_arg),True, parent)
            else:
                current_node = CustomNode("variable",str(sym_arg),True, parent)
            labelCounterTracker[str(sym_arg)] = 1
        else:
            current_node = CustomNode("variable",str(sym_arg) + '|' + str(labelCounterTracker[str(sym_arg)]),True, parent)
            labelCounterTracker[str(sym_arg)] += 1
        
    elif isinstance(sym_arg, Function):
        if str(sym_arg) not in labelCounterTracker:
            current_node = CustomNode("function",str(sym_arg),False, parent)
            labelCounterTracker[str(sym_arg)] = 1
        else:
            current_node = CustomNode("function",str(sym_arg) + '|' + str(labelCounterTracker[str(sym_arg)]),False, parent)
            labelCounterTracker[str(sym_arg)] += 1

    elif isinstance(sym_arg, sp.Basic):
        if labelMap.get(str(sym_arg.func),str(sym_arg.func)) not in labelCounterTracker:
            current_node = CustomNode("operator",labelMap.get(str(sym_arg.func),str(sym_arg.func)),False, parent)
            labelCounterTracker[labelMap.get(str(sym_arg.func),str(sym_arg.func))] = 1
        else:
            current_node = CustomNode("operator",labelMap.get(str(sym_arg.func),str(sym_arg.func)),False, parent)
            current_node = CustomNode("operator",labelMap.get(str(sym_arg.func),str(sym_arg.func)) + '|' + str(labelCounterTracker[labelMap.get(str(sym_arg.func),str(sym_arg.func))]),False, parent)
            labelCounterTracker[labelMap.get(str(sym_arg.func),str(sym_arg.func))] += 1

    if parent:
        parent.add_child(current_node)
    nodes.append(current_node)
    for arg in sym_arg.args:
        recursive_extract(nodes, arg, labelCounterTracker, current_node)
        
## re-arrange the nodes associated with -ve to be consistent
def transform_tree(nodes):
    
    nodes_to_remove = []
    
    for node in nodes:
        # combine -ve w number if only 2 child
        if node.value[0].count('*') == 1 and len(node.children) == 2 and re.sub(r'\|(\d)+','',node.children[0].value) == '-1' and node.children[1].type in ('numeric','variable'):
            nodes_to_remove.append(node.children[0])
            node.delete_child(node.children[0])
            node.set_value('-'+node.children[0].value)
            node.set_type(node.children[0].type)
            node.set_isLeaf(True)
            nodes_to_remove.append(node.children[0])
            node.delete_child(node.children[0])
        # combine -ve w number if > 2 child
        elif node.value[0].count('*') == 1 and len(node.children) > 2 and re.sub(r'\|(\d)+','',node.children[0].value) == '-1' and node.children[1].type in ('numeric','variable'):
            nodes_to_remove.append(node.children[0])
            node.delete_child(node.children[0])
            node.children[0].set_value('-'+node.children[0].value)

    for i in nodes_to_remove:
        nodes.remove(i)

## sort the nodes to get a normalized version
def sort_func(node):
    hashMap = {
        "numeric" : 1,
        "variable" : 2,
        "operator" : 3,
        "function" : 4
    }
    type_sort = hashMap.get(node.type)

    children_sorted = sorted(node.children, key=lambda x: (hashMap.get(x.type, 0), x.value))
    
    # Extract sorted types and values of children
    children_type_sort = [hashMap.get(child.type, 0) for child in children_sorted]
    children_value_sort = [child.value for child in children_sorted]

    return (type_sort,re.sub(r'\-','',re.sub(r'\|(\d)+','',node.value)), children_type_sort, children_value_sort) # 


def add_child(node, root):
    for child in root.children:
        child_node = Node(child.value)
        node.addkid(child_node)
        if not child.isLeaf:
            add_child(child_node, child)


# Print trees
def print_tree(node, level=0):
    print(' ' * level + str(node.label))
    for child in node.children:
        print_tree(child, level + 2)


## apply the functions above to create the normalized tree
def parse_equations(sympy_expr_a):  
    
    nodesA = []
    labelCounterTracker= dict()
    
    numeric_types = (Number, Integer, Rational, Float)
    if isinstance(sympy_expr_a, numeric_types) or isinstance(sympy_expr_a, Symbol) or sympy_expr_a in [pi,E]:
        recursive_extract(nodesA, sympy_expr_a, labelCounterTracker, None)
    else:
        if isinstance(sympy_expr_a, Function):
            current_nodeA = CustomNode("function",str(sympy_expr_a.func),False,0)
        else:
            current_nodeA = CustomNode("operator",labelMap.get(str(sympy_expr_a.func),str(sympy_expr_a.func)),False,0)
        labelCounterTracker[labelMap.get(str(sympy_expr_a.func),str(sympy_expr_a.func))] = 1
        nodesA.append(current_nodeA)
        for arg in sympy_expr_a.args:
            recursive_extract(nodesA, arg, labelCounterTracker, current_nodeA)
        for node in nodesA:
            if re.sub(r'\|(\d)+','',node.value) in ['+','*']:
                node.children = sorted(node.children, key=sort_func)

        transform_tree(nodesA)
            
        for node in nodesA:
            if re.sub(r'\|(\d)+','',node.value) in ['+','*']:
                node.children = sorted(node.children, key=sort_func)

    rootA = nodesA[0]
    while rootA.parent:
        rootA = rootA.parent
    A = Node(rootA.value)
    add_child(A, rootA)

    return A, nodesA


# To recursively consolidate edits of children    
def remove_children(node, to_mod):
    to_remove = []

    # recurse if not leaf. else append node to be removed.
    for x in node.children:
        if not x.isLeaf:
            remove_children(x, to_mod)
        else:
            to_remove.extend([y for y in to_mod if (y[1] == x and y[0] in ('R')) or (y[2] == x and y[0] == 'I')])
    
    # remove the function node itself if it's parents exits in to_mod
    if node.parent and node.parent in [z[1] if z[0] == 'R' else z[2] for z in to_mod if z[0] in ('R', 'I')]:
        to_remove.extend([g for g in to_mod if (g[1] == node or g[2] == node) and g[0] in ('R','I')])
    
    # remove the function node itself if it's parents exits in to_mod
    for element in to_remove:
        if element in to_mod:
            to_mod.remove(element)


# helper function to extract all the children into a flatten list
def extract_recursive(node, children_list):
    for i in node.children:
        children_list.append(i)
        if not i.isLeaf:
            extract_recursive(i, children_list)




## this is to compare the raw string, using string edit distance. 
## can help to capture some initial mistakes
def raw_form_check(str1, str2):
    raw_str1 = re.sub(r'\s+','',str(str1))
    raw_str2 = re.sub(r'\s+','',str(str2))
    str1 = re.sub(r'[\(\) ]+','',str(str1))
    str2 = re.sub(r'[\(\) ]+','',str(str2))
    sorted_str1 = ''.join(sorted(str(str1)))
    sorted_str2 = ''.join(sorted(str(str2)))

    raw_counter_str1 = Counter(raw_str1)
    raw_counter_str2 = Counter(raw_str2)

    counter_str1 = Counter(sorted_str1)
    counter_str2 = Counter(sorted_str2)

    # Characters in str1 but not in str2
    in_1_not_2 = sorted(list((counter_str1 - counter_str2).elements()))
    # Characters in str2 but not in str1
    in_2_not_1 = sorted(list((counter_str2 - counter_str1).elements()))

    # Characters in str1 but not in str2
    in_1_not_2_raw = sorted(list((raw_counter_str1 - raw_counter_str2).elements()))
    # Characters in str2 but not in str1
    in_2_not_1_raw = sorted(list((raw_counter_str2 - raw_counter_str1).elements()))

    diff_char = []
    uniq_char = set(counter_str1.keys()).union(set(counter_str2.keys()))

    diff_char_raw = []
    uniq_char_raw = set(raw_counter_str1.keys()).union(set(raw_counter_str2.keys()))

    for char in uniq_char:
        if counter_str1[char] != counter_str2[char]:
            diff_count = abs(counter_str1[char] - counter_str2[char])
            diff_char.extend([char] * diff_count)
    
    for char in uniq_char_raw:
        if raw_counter_str1[char] != raw_counter_str2[char]:
            diff_count = abs(raw_counter_str1[char] - raw_counter_str2[char])
            diff_char_raw.extend([char] * diff_count)
    
    if diff_char_raw and set(diff_char_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
        if in_1_not_2_raw and in_2_not_1_raw and set(in_1_not_2_raw).issubset(set(['(', ')', '_','*','/','-','+'])) and set(in_2_not_1_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
            return True, f"The student's response has excess term {', '.join(list(set(in_1_not_2_raw)))} and is missing term {', '.join(list(set(in_2_not_1_raw)))}"
        if in_1_not_2_raw and set(in_1_not_2_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
            return True, f"The student's response has excess term {', '.join(list(set(in_1_not_2_raw)))}"
        if in_2_not_1_raw and set(in_2_not_1_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
            return True, f"The student's response has missing term {', '.join(list(set(in_2_not_1_raw)))}"
    
    elif len(diff_char) == 1:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has excess term {', '.join(in_1_not_2)}"
        else:
            return True, f"The student's response has missing term {', '.join(in_2_not_1)}"
    elif len(set(diff_char)) == 2:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has term {in_1_not_2[0]} instead of term {in_2_not_1[0]}"    
   
    return False, 'NA'

## printing utilities

def print_results(row, message, to_mod, treeA, treeB):
    print(f"Row {row}")
    print_tree(treeA)
    print("------------")
    print_tree(treeB)
    print(message)
    for i in to_mod:
        print(i)

def store_results(storage, message, to_mod, treeA, treeB, expr_a, expr_b, raw_A, raw_B,  row):
    storage.append({"message": message, 
                "row" : row,
                'raw_A': raw_A,
                'raw_B': raw_B,
                'expr_A' : expr_a,
                'expr_B' : expr_b,
                "to_mod": to_mod,
                "treeA": treeA,
                "treeB": treeB})
    
def recursive_extract_node(node, string):
    node.set_value(re.sub(r'\|.*','',node.value))
    if node.isLeaf:
        string += "(" + node.value + ")"
        return string
    elif node.type in ('operator'):
        range_len = len(node.children)
        for i in range(range_len):
            if i == 0:
                string += '('
            string = recursive_extract_node(node.children[i], string)
            if i != range_len - 1: 
                string += node.value
            elif i == range_len - 1:
                string += ')'
    elif node.type in ('function'):
        string += node.value
        if '(' not in string:
            string = recursive_extract_node(node.children[0], string)
    return string

def get_sibling(node):
    if node.parent is not None and node.parent!=0:
        siblings = [ i for i in node.parent.children if i != node]
        ## get idx of sibling to be used for append
        if len(siblings) >= 1:
            idx = 0
            for i in range(len(siblings)):
                if siblings[i].type in ["numeric","variable"]:
                    ## because 1/2 is noramlly not represented as 1/2 but e.g pi/2 if there's another var
                    idx = i
                    break
                ## pattern matching to find the position in the string to append the new item
            # if siblings[idx].type in ["function","operator"]:
            #     sibling = re.sub(r"\|\d+", "", recursive_extract_node(siblings[idx],''))
            # else:
            sibling = re.sub(r"\|\d+", "", recursive_extract_node(siblings[idx],''))
        else:
            sibling = ''
    else:
        sibling = ''
    return sibling
        
def escape_selected_characters(text, chars_to_escape):
    # Escape each character in chars_to_escape by replacing it with its escaped version
    for char in chars_to_escape:
        text = text.replace(char, "\\" + char)
    return text

def generate_message(ops):
    if ops[0] == 'I':
        sibling = get_sibling(ops[2])
        hasParent = False
        isNotBase = True
        if ops[2].parent is not None and ops[2].parent!=0:
            hasParent = True
            if re.sub(r'\|.*','',ops[2].parent.value) == "**":
                if ops[2].parent.children[0] == ops[2]:
                    isNotBase = False
        if ops[2].type in ['numeric','variable']:
            temp = f"The student's response is missing term { re.sub(r'\|.*','',ops[2].parent.value) if hasParent and ops[2].parent.type != 'function' and isNotBase else ''}({re.sub(r'\|.*','',ops[2].value)}){" applied to the term " + sibling if sibling !='' and hasParent and isNotBase else ''}."
            return escape_selected_characters(temp,"*+.")   
        elif ops[2].type == 'function':
            temp = f"The student's response is missing term {recursive_extract_node(ops[2],'')}{" applied to the term " + sibling if sibling !='' else ''}."
            return escape_selected_characters(temp,"*+.")    
        else:
            temp = f"The student's response is missing term { re.sub(r'\|.*','',ops[2].parent.value) if hasParent and ops[2].parent.type != 'function' and isNotBase else ''}{recursive_extract_node(ops[2],'')}{" applied to the term " + sibling if sibling !='' and hasParent and isNotBase else ''}."
            return escape_selected_characters(temp,"*+.")      
    elif ops[0] == 'R':
        hasParent = False
        isNotBase = True
        if ops[1].parent is not None and ops[1].parent!=0:
            hasParent = True
            if re.sub(r'\|.*','',ops[1].parent.value) == "**":
                if ops[1].parent.children[0] == ops[1]:
                    isNotBase = False
        sibling = get_sibling(ops[1])
        if ops[1].type in ['numeric','variable']:
            temp = f"The student's response has excess term { re.sub(r'\|.*','',ops[1].parent.value) if hasParent and ops[1].parent.type != 'function' and isNotBase else ''}({re.sub(r'\|.*','',ops[1].value)}){" applied to the term " + sibling if sibling !='' and hasParent and isNotBase else ''}."
            return escape_selected_characters(temp,"*+.")         
        elif ops[1].type == 'function':
            temp = f"The student's response is missing term {recursive_extract_node(ops[1],'')}{" applied to the term " + sibling if sibling !='' else ''}."
            return escape_selected_characters(temp,"*+.")   
        else:
            temp = f"The student's response has excess term {re.sub(r'\|.*','',ops[1].parent.value) if hasParent and ops[1].parent.type != 'function' and isNotBase else ''}{recursive_extract_node(ops[1],'')}{" applied to the term " + sibling if sibling !='' and hasParent and isNotBase else ''}."
            return escape_selected_characters(temp,"*+.")   
    else:
        hasParent = False
        isNotBase = True
        if ops[2].parent is not None and ops[2].parent!=0:
            hasParent = True
            if re.sub(r'\|.*','',ops[2].parent.value) == "**":
                if ops[2].parent.children[0] == ops[2]:
                    isNotBase = False
        hasParent_1 = False
        isNotBase_1 = True
        if ops[1].parent is not None and ops[1].parent!=0:
            hasParent_1 = True
            if re.sub(r'\|.*','',ops[1].parent.value) == "**":
                if ops[1].parent.children[0] == ops[1]:
                    isNotBase_1 = False
        sibling = get_sibling(ops[2])
        if ops[2].type in ['numeric','variable']:
            ins_term_str =  f'{re.sub(r'\|.*','',ops[2].parent.value) if hasParent and ops[2].parent.type != 'function' and isNotBase else ''}({re.sub(r'\|.*','',ops[2].value)})'
        elif ops[2].type == 'function':
            ins_term_str = f'{recursive_extract_node(ops[2],'')}'
        else:
            ins_term_str = f'{re.sub(r'\|.*','',ops[2].parent.value) if hasParent and ops[2].parent.type != 'function' and isNotBase else ''}{recursive_extract_node(ops[2],'')}'
        if ops[1].type in ['numeric','variable']:
            rem_term_str = f'{re.sub(r'\|.*','',ops[1].parent.value) if hasParent_1 and ops[1].parent.type != 'function' and isNotBase_1 else ''}({re.sub(r'\|.*','',ops[1].value)})'
        elif ops[1].type == 'function':
            rem_term_str = f'{recursive_extract_node(ops[1],'')}'
        else:
            rem_term_str = f'{re.sub(r'\|.*','',ops[1].parent.value) if hasParent_1 and ops[1].parent.type != 'function' and isNotBase_1 else ''}{recursive_extract_node(ops[1],'')}'
        temp = f"The student's response has the term {rem_term_str} instead of the term {ins_term_str}{" applied to the term " + sibling if sibling !='' and hasParent and isNotBase else ''}."
        return escape_selected_characters(temp,"*+.")   

def generate_category(ops):
    if ops[0] == 'I':
        if ops[2].type in ['numeric']:
            return f"The student's response is missing a single numeric term."
        elif ops[2].type in ['variable']:
            return f"The student's response is missing a single variable term."
        elif ops[2].type == 'function':
            return f"The student's response is missing a single function term."
        else:
            return f"The student's response is missing terms."
    elif ops[0] == 'R':
        if ops[1].type in ['numeric']:
            return f"The student's response has one excess numeric term."
        elif ops[1].type in ['variable']:
            return f"The student's response has one excess variable term."
        elif ops[1].type == 'function':
            return f"The student's response is missing a single function term." 
        else:
            return f"The student's response has excess terms."
    else:
        if set(re.sub(r'\|(\d)+','',ops[1].value)) ^ set(re.sub(r'\|(\d)+','',ops[2].value)) == set('-'):
            if '-' in re.sub(r'\|(\d)+','',ops[2].value):
                return f"The student's response is missing the term -."
            else:
                return f"The student's response has excess term -."
        elif ops[2].type == ops[1].type:
            return f"The student's response has one wrong {ops[1].type} term."
        else:
            return f"The student's response has one {ops[2].type} term instead of one {ops[1].type} term."

def generate_mult_msg(to_mod):
    if len(set([generate_message(to_mod[i]) for i in range(len(to_mod))])) < len([generate_message(to_mod[i]) for i in range(len(to_mod))]):
        uniq_msg = list(set([generate_message(to_mod[i]) for i in range(len(to_mod))]))
    else:
        uniq_msg = [generate_message(to_mod[i]) for i in range(len(to_mod))]
    msg = ''
       
    for i in range(len(uniq_msg)):
        msg += f'({i+1}) {uniq_msg[i]}'

    
    return msg

def generate_mult_category(to_mod):
    if len(set([generate_category(to_mod[i]) for i in range(len(to_mod))])) < len([generate_category(to_mod[i]) for i in range(len(to_mod))]):
        uniq_msg = list(set([generate_category(to_mod[i]) for i in range(len(to_mod))]))
    else:
        uniq_msg = [generate_category(to_mod[i]) for i in range(len(to_mod))]
    msg = ''
       
    for i in range(len(uniq_msg)):
        msg += f'({i+1}) {uniq_msg[i]}'

    
    return msg

## to clean up the raw edit dist operations provided by zss
def parse_tree(expr_a, expr_b):

    A, nodesA = parse_equations(expr_a)
    B, nodesB = parse_equations(expr_b)

    # [node.print_prop() for node in nodes]

    ## get the edit distance from the parsed trees
    dist, edist = simple_distance(A,B, return_operations=True)

    # edist_ops = ['<Operation Remove: x>', '<Operation Remove: *|1>', '<Operation Update: x to 5>']
    edist_ops = [str(i) for i in edist]

    # to_mod stores the # edits required to convert tree A to tree B
    # This is the format of a node:
    # (type of element e.g operator, value, function), (value of element), (whether element is leaf node), (parent of element), (list of children node of element))
    # for clarity this is the format of to_mod:
    # [(#operation Insert/Update/Remove), (node removed) , (node added), (size of element), (size of entire answer) ]

    to_mod = []
    for i in edist_ops[:]:
        idx_start = i.find(':') + 2
        idx_end = i.find('>') 
        if "Update" in i or "Match" in i:
            cleaned_i = re.sub(r'\|(\d)+','',i)
            idx_start = cleaned_i.find(':') + 2
            idx_end = cleaned_i.find('>') 
            idx_mid=cleaned_i.find(' to') 
            val_a = cleaned_i[idx_start: idx_mid]
            val_b = cleaned_i[idx_mid + 4: idx_end]
            if val_a == val_b or "Match" in i:
                idx_start = i.find(':') + 2
                idx_end = i.find('>') 
                idx_mid=i.find(' to') 
                val_a = i[idx_start: idx_mid]
                val_b = i[idx_mid + 4: idx_end]
                a = next((i for i in nodesA if val_a == i.value), None)
                b = next((i for i in nodesB if val_b == i.value), None)
                if a is not None and b is not None:
                    to_mod.append(['M',a,b,1, len(nodesB)])
            else:
                idx_start = i.find(':') + 2
                idx_end = i.find('>') 
                idx_mid=i.find(' to') 
                val_a = i[idx_start: idx_mid]
                val_b = i[idx_mid + 4: idx_end]
                # print(val_a)
                a = next((i for i in nodesA if val_a == i.value), None)
                b = next((i for i in nodesB if val_b == i.value), None)
                if a is not None and b is not None:
                    to_mod.append(['U',a,b,1,len(nodesB)])
        elif "Remove" in i:
            val_a = i[idx_start:idx_end]
            a = next((i for i in nodesA if val_a == i.value), None)
            if a is not None:
                to_mod.append(['R',a,None,1,len(nodesB)])
        elif "Insert" in i:
            val_b = i[idx_start:idx_end]
            b = next((i for i in nodesB if val_b == i.value), None)
            if b is not None:
                to_mod.append(['I',None,b,1,len(nodesB)])


    ## to compress the edit distance. e.g if entire term 2 * x is missing, edist is default 3 but we compress to edist 1 with * as the operator missing
    ## but if it is 2 * x and only  x is removed, raw edist is 2 as (x, *) are removed but in such cases, we remove * so that edist is still 1 for removing x
    ## because irl we don't really consider operators a term in such scenario
    for i in to_mod[:]:
        if i[0] == 'R' and i[1] and i[1].type == 'operator':
            for x in i[1].children:
                if x not in [j[1] for j in to_mod if j[0] in ('R', 'I')]:
                    to_mod.remove(i)
                    break
                    
        elif i[0] == 'I' and i[2] and i[2].type == 'operator':
            for x in i[2].children:
                if x not in [j[2] for j in to_mod[:] if j[0] in ('R','I')]:
                    to_mod.remove(i)
                    break



    # to remove all the children of operators from to_mod to prevent double count of edist
    for i in to_mod[:]:
        if i[0] == 'R' and i[1] and i[1].type in ['operator','function']:
            remove_children(i[1], to_mod)
        elif i[0] == 'I' and i[2] and i[2].type in ['operator','function']:
            remove_children(i[2], to_mod)

    # M is matched.
    matched_removed = [i[1] for i in to_mod if i[0] =='M']
    matched_added = [i[2] for i in to_mod if i[0] =='M']

    to_delete = []
    for i in to_mod:
        if i[0] == 'R' and i[1].type == 'operator':
            children_list = []
            extract_recursive(i[1], children_list)
            i[3] = len([i for i in children_list if i not in matched_removed])
        elif i[0] == 'I' and i[2].type == 'operator':
            children_list = []
            extract_recursive(i[2], children_list)
            i[3] = len([i for i in children_list if i not in matched_added])
        # to remove the issue of function same, but arg different, yet considered update to function.
        elif i[0] == 'U' and i[1].type == 'function' and i[2].type == 'function':
            if re.sub(r"\(.*\)","",i[1].value) == re.sub(r"\(.*\)","",i[2].value):
                to_delete.append(i)

    ## remaining operations after removing matched ones and deleted ones (e.g duplicate, compression of edist)
    to_mod = [i for i in to_mod if i[0] != 'M' and i not in to_delete]

    return to_mod, A , B

##identify differences in char
def get_diff(a, b):
    str1 = a
    str2 = b
    sorted_str1 = ''.join(sorted(str(str1)))
    sorted_str2 = ''.join(sorted(str(str2)))
    counter_str1 = Counter(sorted_str1)
    counter_str2 = Counter(sorted_str2)


    # Characters in str1 but not in str2
    in_1_not_2 = sorted(list((counter_str1 - counter_str2).elements()))

    # Characters in str2 but not in str1
    in_2_not_1 = sorted(list((counter_str2 - counter_str1).elements()))

    diff_char = []
    uniq_char = set(counter_str1.keys()).union(set(counter_str2.keys()))

    for char in uniq_char:
        if counter_str1[char] != counter_str2[char]:
            diff_count = abs(counter_str1[char] - counter_str2[char])
            diff_char.extend([char] * diff_count)

    return diff_char


def replace_sqrt(exp):
    res = ""
    i = 0
    while i < len(exp):
        if exp[i:i+5] == "sqrt(":
            res += "("  
            i += 5  
            bracket_cnt = 1
            start_idx = i
            while i < len(exp) and bracket_cnt > 0:
                if exp[i] == '(':
                    bracket_cnt += 1
                elif exp[i] == ')':
                    bracket_cnt -= 1
                i += 1
    
            res += exp[start_idx:i-1] + ")**(1/2)"
        else:
            res += exp[i]
            i += 1
    
    return res

## this is for validate insert operations suggested, by applying them to the submission string

# for insert operators, you would typically have another operand e.g trying to insert *2x to y in (2+y). So we need to find the pos of y so that we can append
def find_pos(string, substr):
    idx = string.find(substr)
    positions = []
    while idx != -1:
        positions.append(idx)
        idx = string.find(substr, idx + 1)
    return positions

# for insert operators, you would typically have another operand e.g trying to insert *2x to y in (2+y). So we need to find the pos of y so that we can append
def find_pos_regex(string, pattern):
    print("string", string)
    print("pattern", pattern)
    matches = re.finditer(pattern, string)
    positions = []
    for match in matches:
        start_idx = match.start()
        end_idx = match.end()
        len_match = end_idx - start_idx
        positions.append((start_idx,len_match))
    print(positions)
    return positions
    
## this is for validate insert operations suggested, by applying them to the submission string
def test_insert(submission, answer, params, expr_a, correction):

    ## short-circuit, when replacing the entire submission i.e replacing the entire (3+y) with 5
    if correction.parent is None or correction.parent == 0:
        submission = ''
        to_add =recursive_extract_node(correction,'')
        a, b = check_equality(to_add,answer,params,{})
        if a == b:
            return True
    elif correction.parent.parent is None or correction.parent.parent == 0 or correction.parent.parent.value == '=':
        to_add = re.sub(r"\|\d+", "",correction.parent.value) + (recursive_extract_node(correction,''))
        new_str = "(" + submission + ")" + to_add
        a, b = check_equality(new_str,answer,params,{})
        if a == b:
            return True
    else:
        # operator + operand
        to_add = re.sub(r"\|\d+", "",correction.parent.value) + (recursive_extract_node(correction,''))


    ## for cases where submission string is e.g 0.7 but sympy is 7/10. Can't match if we match str, so evaluate to match
    try:
        submission_value = float(sp.sympify(submission).evalf())
        expr_a_value = float(expr_a)
        if submission_value == expr_a_value:
            new_str = "(" + submission + ")" + "(" + to_add + ")"
            a, b = check_equality(new_str,answer,params,{})
            if a == b:
                return True
    except (TypeError, ValueError, sp.SympifyError) as e:
        pass

    # parse ^ in submission to ** to standardize w sympy
    submission = re.sub(r"\^","**",submission)
    expr_a = str(expr_a)
    submission = re.sub(r"\s|","",submission)
    ## replace sqrt str with **(1/2)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",expr_a)
    
    ## there could be multiple sibling operands. e.g if i'm adding *2z to x*y, the sibling could be x or y
    if correction.parent:
        siblings = [ i for i in correction.parent.children if i != correction]
    
    ## get idx of sibling to be used for append
    idx = 0
    for i in range(len(siblings)):
        if siblings[i].type in ["numeric","variable"]:
            ## because 1/2 is noramlly not represented as 1/2 but e.g pi/2 if there's another var
            if siblings[i].value == '1/2':
                idx = i
                continue
            ## prioritize non 1/2 sibling
            if siblings[i].value in submission:
                idx = i
                break

    ## pattern matching to find the position in the string to append the new item
    if siblings[idx].type in ["function","operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(siblings[0],'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        ## to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
            
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)
        
        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    else:
        clean_sibling_element = re.sub(r"\|\d+", "", siblings[idx].value)
        possible_pos = find_pos(submission, clean_sibling_element)
        len_siblings = [len(clean_sibling_element) for i in possible_pos]
    
    
    ## try appending the new element and check if the new submission matches the answer
    for idx in range(len(possible_pos)):
        
        new_str = submission[:possible_pos[idx]] + "((" + submission[possible_pos[idx]:]
        new_str = new_str[:possible_pos[idx]+len_siblings[idx]+2] +')' + to_add + ')' + new_str[possible_pos[idx]+len_siblings[idx]+2:]
        a, b = check_equality(new_str,answer,params,{})
        if a == b:
            return True
    
    return False

## this is for validate remove operations suggested, by applying them to the submission string

def test_removal(submission, answer, params, expr_a, correction):

    if correction.parent is not None and correction.parent != 0:
        to_remove = re.sub(r"\|\d+", "",correction.parent.value) + (recursive_extract_node(correction,''))
    else:
        to_remove = recursive_extract_node(correction,'')
    expr_a = str(expr_a)
    submission = re.sub(r"\^","**",submission)
    
    submission = re.sub(r"\s","",submission)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",str(expr_a))
        
    
    if correction.type in ["function","operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(correction,'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        # to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:   
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)

        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    ## this is for numeric, variables
    else:
        clean_element = re.sub(r"\|\d+", "", correction.value)
        ## because 1/2 typically expressed as .../(2..)
        clean_match = re.search(r'1/(\d+)',clean_element)
        if clean_match:
            clean_element = clean_match.group(1)
        ## -1 in AST is just - in string
        elif clean_element == '-1' and correction.parent.value == '*':
            clean_element = '-'
            
        possible_pos = find_pos(submission, clean_element)
        len_siblings = [len(clean_element) for i in possible_pos]
    
    ## replace multiples with * 1 and additions with + 0.
    for idx in range(len(possible_pos)):
        if re.sub(r"\|\d+", "", correction.parent.value) in ['+']:
            new_str = submission[:possible_pos[idx]] + '(0)' + submission[possible_pos[idx]+len_siblings[idx]:]
            a, b = check_equality(new_str,answer,params,{})
            if a == b:
                return True
        elif re.sub(r"\|\d+", "", correction.parent.value)  in ['*','**']:
            ## tried subbing in 10 instead but seems like parser doesn't parse the log properly
            if to_remove == "*(log(E)**(-1))" and possible_pos:
                return True
            else:
                new_str = submission[:possible_pos[idx]] + '(1)' + submission[possible_pos[idx]+len_siblings[idx]:]
                a, b = check_equality(new_str,answer,params,{})
                if a == b:
                    return True
    
    return False




## this is for validate update operations suggested, by applying them to the submission string

def test_update(submission, answer, params, expr_a, remove, update):
    
    if get_diff(remove.value, update.value) == ['-']:
        if remove.value[0] == '-':
            possible_pos = find_pos(submission, '-')
            len_siblings = [1 for i in possible_pos]
            for idx in range(len(possible_pos)):
                new_str = submission[:possible_pos[idx]] + submission[possible_pos[idx]+len_siblings[idx]:]
                a, b = check_equality(new_str,answer,params,{})
                if a == b:
                    return True  
    
    if update.parent:
        to_add_w_signs = re.sub(r"\|\d+", "",update.parent.value) + (recursive_extract_node(update,''))
    


    to_remove = recursive_extract_node(remove,'')
    to_add = recursive_extract_node(update,'')


    if remove.type == 'numeric':
        try:
            numerator_match = re.search(r'(\d+)/?(\d+)?', remove.value)
            numerator = numerator_match.group(1)
            possible_pos = find_pos(submission, numerator)
            len_siblings = [len(numerator) for i in possible_pos]
            for idx in range(len(possible_pos)):
                if re.sub(r"\|\d+", "",remove.parent.value) == '+':
                    new_str = submission[:possible_pos[idx] + len_siblings[idx]] + f'+({to_add})-({to_remove})' + submission[possible_pos[idx]+len_siblings[idx]:]
                    a, b = check_equality(new_str,answer,params,{})
                    if a == b:
                        return True
                else:
                    new_str = submission[:possible_pos[idx] + len_siblings[idx]] + f'*({to_add})/({to_remove})' + submission[possible_pos[idx]+len_siblings[idx]:]
                    a, b = check_equality(new_str,answer,params,{})
                    if a == b:
                        return True
                
            
        
        except:
            pass

    ## when replcing the entire char with function. row 617
    if update.parent is None or update.parent == 0:
        submission = ''
        to_add =recursive_extract_node(update,'')
        a, b = check_equality(to_add,answer,params,{})
        if a == b:
            return True
    

    expr_a = str(expr_a)
    submission = re.sub(r"\s","",submission)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",str(expr_a))
        

    if remove.type in ["operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(remove,'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        # to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            print('in match 1')
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)

        ## to match 1/b
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)

        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    ## this is for numeric, variables
    else:
        pattern_to_match = re.sub(r"\|\d+", "", remove.value)
        ## because 1/2 typically expressed as .../(2..)
        clean_match = re.search(r'1/(\d+)',pattern_to_match)
        if clean_match:

            pattern_to_match = clean_match.group(1)
            if to_add_w_signs:
                if re.search(r'1/(\d+)',to_add_w_signs):  
                    to_add_w_signs = re.sub(r'1/(\d+)',r'\1',to_add_w_signs)
                elif re.search(r'(\d+)/(\d+)', to_add_w_signs):
                    to_add_w_signs = re.sub(r'(\d+)/(\d+)', r'\2/\1', to_add_w_signs)  
                else:
                    to_add_w_signs = re.sub(r'(\d+)', r'1/\1', to_add_w_signs) 

            if re.search(r'1/(\d+)',to_add):
                to_add =re.sub(r'1/(\d+)',r'\1',to_add)
            elif re.search(r'(\d+)/(\d+)', to_add):
                to_add = re.sub(r'(\d+)/(\d+)', r'\2/\1', to_add)  
            else:
                to_add = re.sub(r'(\d+)', r'1/\1', to_add) 
            
        ## -1 in AST is just - in string
        elif pattern_to_match == '-1' and remove.parent.value == '*':
            pattern_to_match = '-'
        
        possible_pos = find_pos(submission, pattern_to_match)
        if possible_pos == []:
            try:
                fraction = Fraction(pattern_to_match)
                pattern_to_match = str(float(fraction))
                possible_pos = find_pos(submission, pattern_to_match)
            except:
                pass

        len_siblings = [len(pattern_to_match) for i in possible_pos]
    
   
    for idx in range(len(possible_pos)):
        new_str = submission[:possible_pos[idx]] + f'{to_add}' + submission[possible_pos[idx]+len_siblings[idx]:]
        a, b = check_equality(new_str,answer,params,{})
        if a == b:
            return True
        new_str = submission[:possible_pos[idx]] + f'{to_add[1:-1]}' + submission[possible_pos[idx]+len_siblings[idx]:]
        a, b = check_equality(new_str,answer,params,{})
        if a == b:
            return True
        if update.parent:
            new_str = submission[:possible_pos[idx]] + f'{to_add_w_signs}' + submission[possible_pos[idx]+len_siblings[idx]:]

            a, b = check_equality(new_str,answer,params,{})
            if a == b:
                return True
    new_str = submission.replace(pattern_to_match, to_add)
    a, b = check_equality(new_str,answer,params,{})
    if a == b:
        return True
    
    return False

## this is for 2 steps operations, starting with update. Mainly same as normal update operations, just that it triggers the 2nd step when done instead of checking if output matches answer

def comb_test_update(submission, answer, params, expr_a, remove, update, next_step):
    
    if get_diff(remove.value, update.value) == ['-']:
        if remove.value[0] == '-':
            possible_pos = find_pos(submission, '-')
            len_siblings = [1 for i in possible_pos]
            for idx in range(len(possible_pos)):
                new_str = submission[:possible_pos[idx]] + submission[possible_pos[idx]+len_siblings[idx]:]
                if trigger_check(new_str, answer, params, expr_a, next_step):
                    return True
            

     ## when replcing the entire char with function. row 617
    if remove.parent:
        to_remove_w_signs = re.sub(r"\|\d+", "",remove.parent.value) + (recursive_extract_node(remove,''))

    if update.parent:
        to_add_w_signs = re.sub(r"\|\d+", "",update.parent.value) + (recursive_extract_node(update,''))
    


    to_remove = recursive_extract_node(remove,'')
    to_add = recursive_extract_node(update,'')


    if remove.type == 'numeric':
        try:
            numerator_match = re.search(r'(\d+)/?(\d+)?', remove.value)
            numerator = numerator_match.group(1)
            possible_pos = find_pos(submission, numerator)
            len_siblings = [len(numerator) for i in possible_pos]
            for idx in range(len(possible_pos)):
                if re.sub(r"\|\d+", "",remove.parent.value) == '+':
                    new_str = submission[:possible_pos[idx] + len_siblings[idx]] + f'+({to_add})-({to_remove})' + submission[possible_pos[idx]+len_siblings[idx]:]
                    if trigger_check(new_str, answer, params, expr_a, next_step):
                        return True
                else:
                    new_str = submission[:possible_pos[idx] + len_siblings[idx]] + f'*({to_add})/({to_remove})' + submission[possible_pos[idx]+len_siblings[idx]:]
                    if trigger_check(new_str, answer, params, expr_a, next_step):
                        return True
                
        
        
        except:
            pass

    expr_a = str(expr_a)
    submission = re.sub(r"\s","",submission)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",str(expr_a))
            

    
    if remove.type in ["operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(remove,'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        # to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)

        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    ## this is for numeric, variables
    else:
        pattern_to_match = re.sub(r"\|\d+", "", remove.value)
        ## because 1/2 typically expressed as .../(2..)
        clean_match = re.search(r'1/(\d+)',pattern_to_match)
        if clean_match:
            pattern_to_match = clean_match.group(1)
            if to_add_w_signs:
                if re.search(r'1/(\d+)',to_add_w_signs):  
                    to_add_w_signs = re.sub(r'1/(\d+)',r'\1',to_add_w_signs)
                elif re.search(r'(\d+)/(\d+)', to_add_w_signs):
                    to_add_w_signs = re.sub(r'(\d+)/(\d+)', r'\2/\1', to_add_w_signs)  
                else:
                    to_add_w_signs = re.sub(r'(\d+)', r'1/\1', to_add_w_signs) 

            if re.search(r'1/(\d+)',to_add):
                to_add =re.sub(r'1/(\d+)',r'\1',to_add)
            elif re.search(r'(\d+)/(\d+)', to_add):
                to_add = re.sub(r'(\d+)/(\d+)', r'\2/\1', to_add)  
            else:
                to_add = re.sub(r'(\d+)', r'1/\1', to_add) 
            
        ## -1 in AST is just - in string
        elif pattern_to_match == '-1' and remove.parent.value == '*':
            pattern_to_match = '-'
        
        possible_pos = find_pos(submission, pattern_to_match)
        if possible_pos == []:
            try:
                fraction = Fraction(pattern_to_match)
                pattern_to_match = str(float(fraction))
                possible_pos = find_pos(submission, pattern_to_match)
            except:
                pass
        len_siblings = [len(pattern_to_match) for i in possible_pos]
    
   
    for idx in range(len(possible_pos)):
        new_str = submission[:possible_pos[idx]] + f'{to_add}' + submission[possible_pos[idx]+len_siblings[idx]:]
        if trigger_check(new_str, answer, params, expr_a, next_step):
            return True
        new_str = submission[:possible_pos[idx]] + f'{to_add[1:-1]}' + submission[possible_pos[idx]+len_siblings[idx]:]
        if trigger_check(new_str, answer, params, expr_a, next_step):
            return True
        if update.parent and remove.parent:
            new_str = submission[:possible_pos[idx]] + f'{to_add_w_signs}' + submission[possible_pos[idx]+len_siblings[idx]:]

            if trigger_check(new_str, answer, params, expr_a, next_step):
                return True
    new_str = submission.replace(pattern_to_match, to_add)
    if trigger_check(new_str, answer, params, expr_a, next_step):
        return True
    

    return False



## this is for 2 steps operations, starting with removal. Mainly same as normal removal operations, just that it triggers the 2nd step when done instead of checking if output matches answer


def comb_test_removal(submission, answer, params, expr_a, correction, next_step):

    if correction.parent is not None and correction.parent != 0:
        to_remove = re.sub(r"\|\d+", "",correction.parent.value) + (recursive_extract_node(correction,''))
    else:
        to_remove = recursive_extract_node(correction,'')

    submission = re.sub(r"\^","**",submission)
    expr_a = str(expr_a)
    submission = re.sub(r"\s","",submission)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",str(expr_a))
            
    
    if correction.type in ["function","operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(correction,'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        # to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)

        ## to match 1/b
                
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)

        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    else:
        clean_element = re.sub(r"\|\d+", "", correction.value)
        ## because 1/2 typically expressed as .../(2..)
        clean_match = re.search(r'1/(\d+)',clean_element)
        if clean_match:
            clean_element = clean_match.group(1)
        ## -1 in AST is just - in string
        elif clean_element == '-1' and correction.parent.value == '*':
            clean_element = '-'
        possible_pos = find_pos(submission, clean_element)
        len_siblings = [len(clean_element) for i in possible_pos]
    
    ## replace multiples with * 1 and additions with + 0.
    for idx in range(len(possible_pos)):
        if re.sub(r"\|\d+", "", correction.parent.value) in ['+']:
            new_str = submission[:possible_pos[idx]] + '(0)' + submission[possible_pos[idx]+len_siblings[idx]:]
            if trigger_check(new_str, answer, params,expr_a, next_step):
                return True
        elif re.sub(r"\|\d+", "", correction.parent.value)  in ['*','**']:
            ## tried subbing in 10 instead but seems like parser doesn't parse the log properly
            new_str = submission[:possible_pos[idx]] + '(1)' + submission[possible_pos[idx]+len_siblings[idx]:]
            if trigger_check(new_str, answer, params, expr_a, next_step):
                return True
    
    return False


## this is for 2 steps operations, starting with insert. Mainly same as normal insert operations, just that it triggers the 2nd step when done instead of checking if output matches answer

def comb_test_insert(submission, answer, params, expr_a, correction, next_step):

    to_add = re.sub(r"\|\d+", "",correction.parent.value) + (recursive_extract_node(correction,''))

    try:
        submission_value = float(sp.sympify(submission).evalf())
        expr_a_value = float(expr_a)
        if submission_value == expr_a_value:
            new_str = "(" + submission + ")" + "(" + to_add + ")"
            if trigger_check(new_str, answer, params, expr_a, next_step):
                return True
    except (TypeError, ValueError, sp.SympifyError) as e:
        pass

    submission = re.sub(r"\^","**",submission)
    expr_a = str(expr_a)


    if correction.parent.parent is None or correction.parent.parent == 0 or correction.parent.parent.value == '=':
        new_str = "(" + submission + ")" + to_add
        if trigger_check(new_str, answer, params, expr_a, next_step):
            return True
    
    submission = re.sub(r"\s|","",submission)
    submission = replace_sqrt(submission)
    expr_a = re.sub(r"|\s","",str(expr_a))
            
    if correction.parent:
        siblings = [ i for i in correction.parent.children if i != correction]
    
    print("siblings", siblings)
    idx = 0
    for i in range(len(siblings)):
        if siblings[i].type in ["numeric","variable"]:
            ## because 1/2 is noramlly not represented as 1/2 but e.g pi/2 if there's another var
            if siblings[i].value == '1/2':
                idx = i
                continue
            if siblings[i].value in submission:
                idx = i
                break
    ## get sibling elements return the sibling operand. clean_sibling removes the |2 etc.
    if siblings[idx].type in ["function","operator"]:
        str_to_find = re.sub(r"\|\d+", "", recursive_extract_node(siblings[0],'')[1:-1])

        ## to match a/b
        match = re.search(r"(.+)\*\((.+)\*\*\(\-1\)\)",  str_to_find)
        ## to match 1/b
        match2 = re.search(r"(\(.+\))\*\*\(\-1\)",  str_to_find)
        if match:
            pattern_to_match = r"(" + match.group(1) + r")/(" + match.group(2) + r")"
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)            
            pattern_to_match_escaped = re.escape(pattern_to_match)
            # submission = re.sub(r'\(|\)', '', submission)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                # submission = re.sub(r'\(|\)', '', submission)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        elif match2:
            pattern_to_match = ''.join(sorted(match2.group(1)))
            pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', '',pattern_to_match)
            pattern_to_match_escaped = re.escape(pattern_to_match)
            results = find_pos_regex(submission, pattern_to_match_escaped)
            if results == []:
                pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
        else:
            if re.search(r'(\+|(?<!\*)\*(?!\*))', str_to_find):
                items = re.split(r'(\+|(?<!\*)\*(?!\*))', str_to_find ) 
                if len(items) >= 5:
                    str_to_find_permutation = [str_to_find]
                else:
                    str_to_find_permutation = [''.join(itertools.chain.from_iterable(p)) for p in itertools.permutations(items)]
                patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|(?<=\*\*\()\-', r'',i)) for i in str_to_find_permutation]
                patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                results = []
                counter = 0
                for pattern in patterns_to_match_escaped:
                    result = find_pos_regex(submission, pattern)
                    results.extend(result)
                    counter+= 1
                    if counter > 20:
                        break
                if results == []:
                    patterns_to_match = [re.sub(r'\+\-','-',re.sub(r'\(|\)|\*', r'',i)) for i in str_to_find_permutation]
                    patterns_to_match_escaped = [re.escape(i) for i in patterns_to_match]
                    counter = 0
                    for pattern in patterns_to_match_escaped:
                        result = find_pos_regex(submission, pattern)
                        results.extend(result) 
                        counter+= 1
                        if counter > 20:
                            break
            else:
                pattern_to_match = re.sub(r'\(|\)|(?<=\*\*\()\-', r'',str_to_find)
                pattern_to_match_escaped = re.escape(pattern_to_match)
                results = find_pos_regex(submission, pattern_to_match_escaped)
                if results == []:
                    pattern_to_match = re.sub(r'\*', '',pattern_to_match)
                    pattern_to_match_escaped = re.escape(pattern_to_match)
                    results = find_pos_regex(submission, pattern_to_match_escaped)
        
        possible_pos = [i[0] for i in results]
        len_siblings = [i[1] for i in results]
    else:
        clean_sibling_element = re.sub(r"\|\d+", "", siblings[idx].value)
        possible_pos = find_pos(submission, clean_sibling_element)
        len_siblings = [len(clean_sibling_element) for i in possible_pos]
    
    
    for idx in range(len(possible_pos)):
        
        new_str = submission[:possible_pos[idx]] + "((" + submission[possible_pos[idx]:]
        new_str = new_str[:possible_pos[idx]+len_siblings[idx]+2] +')' + to_add + ')' + new_str[possible_pos[idx]+len_siblings[idx]+2:]
        if trigger_check(new_str, answer, params, expr_a, next_step):
            return True
    
    return False



## tool to trigger the various function

def trigger_check(submission, answer, params, expr_a, next_step):
    if next_step[0] == 'U':
        return test_update(submission, answer, params, expr_a, next_step[1], next_step[2])
    elif next_step[0] == 'R':
        return test_removal(submission, answer, params, expr_a, next_step[1])
    elif next_step[0] == 'I':
        return test_insert(submission, answer, params, expr_a, next_step[2])

def comb_trigger_check(submission, answer, params, expr_a, first_step, next_step):
    if first_step[0] == 'U':
        return comb_test_update(submission, answer, params, expr_a, first_step[1], first_step[2], next_step)
    elif first_step[0] == 'R':
        return comb_test_removal(submission, answer, params, expr_a, first_step[1], next_step)
    elif first_step[0] == 'I':
        return comb_test_insert(submission, answer, params, expr_a, first_step[2], next_step)


## run 

def run_all(commonMistakes):
    for i in commonMistakes:

        ## load params provided
        expr_a, expr_b  = check_equality(i["submission"].replace('"',''),i["answer"].replace('"',''),i["params"],{})
        params = i["params"]
        raw_A = i["submission"].replace('"','')
        raw_B = i["answer"].replace('"','')
        # [node.print_prop() for node in nodes]
        to_mod, A, B = parse_tree(expr_a, expr_b)
        # Compare the raw string to catch some scenarios
        form_check_bool_raw, form_check_msg_raw = raw_form_check(str(raw_A), str(raw_B))
        
        # # catch the brackets first.
        if form_check_msg_raw in ["The student's response has excess term ), (", "The student's response has excess term (, )"]:
            i["recommendedFeedback"] = "(1) The student's response has excess parenthesis."
        elif form_check_msg_raw in ["The student's response has missing term ), (", "The student's response has missing term (, )"]:
            i["recommendedFeedback"] = "(1) The student's response is missing parenthesis."
        elif len(to_mod) == 1:
            if to_mod[0][0] == 'U' and to_mod[0][1].value == 'a' and to_mod[0][2].value == 'b':
                i["recommendedFeedback"] =  "(1) Unable to be parsed!" 
            elif trigger_check(raw_A, raw_B, params, expr_a, to_mod[0]):
                if to_mod[0][0] == 'U' and to_mod[0][4] == 1 and to_mod[0][1].type == to_mod[0][2].type and to_mod[0][1].type == 'numeric':
                    atol = params.get('atol',0)*2
                    rtol = max(params.get("rtol", 0.05)*2,0.1)
                    real_diff = None
                    response = float(sp.sympify(to_mod[0][1].value).evalf())
                    answer = float(sp.sympify(to_mod[0][2].value).evalf())
                    real_diff = abs(response - answer)
                    allowed_diff = atol + rtol * abs(answer)
                    allowed_diff += spacing(answer)
                    is_close = bool(real_diff <= allowed_diff)
                    is_factor = False
                    if response != 0 and answer != 0:
                        ratio = response / answer
                        log_ratio = np.log10(abs(ratio))
                        is_factor = log_ratio.is_integer()
                    if is_close:
                        i["recommendedFeedback"] =  "(1) The student's reponse is close, within twice the allowed tolerance range."
                    elif set(re.sub(r'\|(\d)+','',to_mod[0][1].value)) ^ set(re.sub(r'\|(\d)+','',to_mod[0][2].value)) == set('-'):
                        i["recommendedFeedback"] =  "(1) The student's response differs by the term -."  
                    elif is_factor:
                        i["recommendedFeedback"] = f"(1) The student's response is a factor of 10**{-log_ratio} away from the answer."

                else:
                    if to_mod[0][4] > 1:
                        i["recommendedFeedback"] =  generate_mult_msg(to_mod)
            # if message_store != []:
            #     store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
        # elif form_check_bool_raw:
        #     i["recommendedFeedback"] =  "(1) " + form_check_msg_raw
             # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
        elif len(to_mod) == 2 and comb_trigger_check(raw_A, raw_B, params, expr_a, to_mod[0], to_mod[1]):
            i["recommendedFeedback"] =  generate_mult_msg(to_mod)
            # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
    
    return commonMistakes
