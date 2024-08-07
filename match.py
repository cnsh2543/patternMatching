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
            return True, f"The student's response has excess terms {', '.join(list(set(in_1_not_2_raw)))} and is missing terms {', '.join(list(set(in_2_not_1_raw)))}"
        if in_1_not_2_raw and set(in_1_not_2_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
            return True, f"The student's response has excess terms {', '.join(list(set(in_1_not_2_raw)))}"
        if in_2_not_1_raw and set(in_2_not_1_raw).issubset(set(['(', ')', '_','*','/','-','+'])):
            return True, f"The student's response has missing terms {', '.join(list(set(in_2_not_1_raw)))}"
    
    elif len(diff_char) == 1:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has excess term {', '.join(in_1_not_2)}"
        else:
            return True, f"The student's response has missing term {', '.join(in_2_not_1)}"
    elif len(set(diff_char)) == 2:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has term {in_1_not_2[0]} instead of term {in_2_not_1[0]}"    
   
    return False, 'NA'

## this is to compare the sympy parsed string, using string edit distance. 
## can help to capture some initial mistakes
def form_check(str1, str2):
    str1 = re.sub(r'[\(\) ]+','',str(str1))
    str2 = re.sub(r'[\(\) ]+','',str(str2))
    sorted_str1 = ''.join(sorted(str(str1)))
    sorted_str2 = ''.join(sorted(str(str2)))
    counter_str1 = Counter(sorted_str1)
    counter_str2 = Counter(sorted_str2)

    lower_sorted_str1 = ''.join(sorted(str(str1).lower()))
    lower_sorted_str2 = ''.join(sorted(str(str2).lower()))
    lower_counter_str1 = Counter(lower_sorted_str1)
    lower_counter_str2 = Counter(lower_sorted_str2)

    # Characters in str1 but not in str2
    in_1_not_2 = sorted(list((counter_str1 - counter_str2).elements()))
    
    # Characters in str2 but not in str1
    in_2_not_1 = sorted(list((counter_str2 - counter_str1).elements()))

    diff_char = []
    uniq_char = set(counter_str1.keys()).union(set(counter_str2.keys()))

    diff_char_lower = []
    uniq_char_lower = set(lower_counter_str1.keys()).union(set(lower_counter_str2.keys()))

    for char in uniq_char:
        if counter_str1[char] != counter_str2[char]:
            diff_count = abs(counter_str1[char] - counter_str2[char])
            diff_char.extend([char] * diff_count)
    
    for char in uniq_char_lower:
        if lower_counter_str1[char] != lower_counter_str2[char]:
            diff_count = abs(lower_counter_str1[char] - lower_counter_str2[char])
            diff_char_lower.extend([char] * diff_count)
    
    diff_char.sort()

    ## capture _ early, else in AST edit distance wil be much larger
    if diff_char and set(diff_char).issubset(set(['_'])):
        if in_1_not_2 and set(in_1_not_2).issubset(set(['_'])):
            return True, f"The student's response has excess term {', '.join(list(set(in_1_not_2)))}"
        if in_2_not_1 and set(in_2_not_1).issubset(set(['_'])):
            return True, f"The student's response has missing term {', '.join(list(set(in_2_not_1)))}"
    
    elif len(diff_char) == 1:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has excess term {', '.join(in_1_not_2)}"
        else:
            return True, f"The student's response has missing term {', '.join(in_2_not_1)}"
    elif len(set(diff_char)) == 2:
        if len(in_1_not_2) == 1:
            return True, f"The student's response has {in_1_not_2[0]} instead of {in_2_not_1[0]}"    
        elif re.search(r'[A-Za-z0-9]',('').join(diff_char)) and re.search(r'[\*\/]',(', ').join(diff_char)):
            if in_1_not_2 == diff_char:
                return True, f"The student's response has excess term {', '.join(diff_char)}"    
            elif in_2_not_1 == diff_char:
                return True, f"The student's response has missing term {', '.join(diff_char)}"    
    ## capture **2 early, else in AST edit distance wil be much larger
    elif len(diff_char) == 3 and re.search(r'[A-Za-z0-9]',('').join(diff_char)) and re.search(r'[\*]{2}',('').join(diff_char)):
        if in_1_not_2 == diff_char:
            return True, f"The student's response has excess term **{re.search(r'[A-Za-z0-9]',('').join(diff_char))}"
        elif in_2_not_1 == diff_char:
            return True, f"The student's response has missing term **{re.search(r'[A-Za-z0-9]',('').join(diff_char))}"

   
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

def store_results(storage, message, to_mod, treeA, treeB, raw_A, raw_B, row):
    storage.append({"message": message, 
                "row" : row,
                'raw_A': raw_A,
                'raw_B': raw_B,
                "to_mod": to_mod,
                "treeA": treeA,
                "treeB": treeB})
    
def recursive_extract_node(node, string):
    node.set_value(re.sub(r'\|.*','',node.value))
    if node.isLeaf:
        string += node.value
        return string
    elif node.type in ('operator','function'):
        range_len = len(node.children)
        for i in range(range_len):
            if i == 0:
                string += '('
            string = recursive_extract_node(node.children[i], string)
            if i != range_len - 1: 
                string += node.value
            elif i == range_len - 1:
                string += ')'
    return string


def generate_message(ops):
    if ops[0] == 'I':
        if ops[2].type in ['numeric','variable']:
            return f"The student's response is missing term {re.sub(r'\|.*','',ops[2].value)}."
        else:
            return f"The student's response is missing term {recursive_extract_node(ops[2],'')[1:-1]}. "
    elif ops[0] == 'R':
         if ops[1].type in ['numeric','variable']:
            return f"The student's response has excess term {re.sub(r'\|.*','',ops[1].value)}. "
         else:
            return f"The student's response has excess term {recursive_extract_node(ops[1],'')[1:-1]}. "
    else:
        if ops[2].type in ['numeric','variable']:
            ins_term_str =  f'{re.sub(r'\|.*','',ops[2].value)}'
        else:
            ins_term_str = recursive_extract_node(ops[2],'')[1:-1]
        if ops[1].type in ['numeric','variable']:
            rem_term_str = f'{re.sub(r'\|.*','',ops[1].value)}'
        else:
            rem_term_str = recursive_extract_node(ops[1],'')[1:-1]
        
        return f"The student's response has the term {rem_term_str} instead of the term {ins_term_str}. "

def generate_mult_msg(to_mod):
    uniq_msg = list(set([generate_message(to_mod[i]) for i in range(len(to_mod))]))
    msg = ''
       
    for i in range(len(uniq_msg)):
            msg += f'({i+1}) {uniq_msg[i]} '

    
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
        if i[0] == 'R' and i[1] and i[1].type in ('operator'):
            remove_children(i[1], to_mod)
        elif i[0] == 'I' and i[2] and i[2].type in ('operator'):
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

        ## Compare the sympy parsed string to catch some scenarios
        form_check_bool, form_check_msg = form_check(expr_a, expr_b)
        
        # # catch the brackets first.
        if form_check_msg_raw in ["The student's response has excess ), (", "The student's response has missing ), ("]:
            i["recommendedFeedback"] =  "(1) " + form_check_msg_raw
            # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
        elif len(to_mod) == 1:
            if to_mod[0][0] == 'U' and to_mod[0][1].value == 'a' and to_mod[0][2].value == 'b':
                i["recommendedFeedback"] =  "(1) Unable to be parsed!" 
            elif to_mod[0][0] == 'U' and to_mod[0][4] == 1 and to_mod[0][1].type == to_mod[0][2].type and to_mod[0][1].type == 'numeric':
                atol = params.get('atol',0)*2
                rtol = max(params.get("rtol", 0.05)*2,0.1)
                is_correct = None
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
                    i["recommendedFeedback"] =  f"(1) The student's response is a factor of {log_ratio} away from the answer."
            else:
                i["recommendedFeedback"] =  generate_mult_msg(to_mod)
            # if message_store != []:
            #     store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
        elif form_check_bool_raw:
            i["recommendedFeedback"] =  "(1) " + form_check_msg_raw
            # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
            # message_store.append(form_check_msg_raw)
        elif form_check_bool:
            i["recommendedFeedback"] =  '(1) ' +form_check_msg
            # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
        elif len(to_mod) <= 2:
            i["recommendedFeedback"] =  generate_mult_msg(to_mod)
            # store_results(result_store, message_store, to_mod, A, B, raw_A, raw_B, counter)
    
    return commonMistakes
