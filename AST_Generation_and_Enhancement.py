import os
import networkx as nx
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
from anytree import AnyNode

from Triads import traids
from multiprocessing import Pool
from functools import partial
import json
import time


edges={'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}

nodetypedict = {'MethodDeclaration': 0, 'Modifier': 1, 'FormalParameter': 2, 'ReferenceType': 3, 'BasicType': 4,
     'LocalVariableDeclaration': 5, 'VariableDeclarator': 6, 'MemberReference': 7, 'ArraySelector': 8, 'Literal': 9,
     'BinaryOperation': 10, 'TernaryExpression': 11, 'IfStatement': 12, 'BlockStatement': 13, 'StatementExpression': 14,
     'Assignment': 15, 'MethodInvocation': 16, 'Cast': 17, 'ForStatement': 18, 'ForControl': 19,
     'VariableDeclaration': 20, 'TryStatement': 21, 'ClassCreator': 22, 'CatchClause': 23, 'CatchClauseParameter': 24,
     'ThrowStatement': 25, 'WhileStatement': 26, 'ArrayInitializer': 27, 'ReturnStatement': 28, 'Annotation': 29,
     'SwitchStatement': 30, 'SwitchStatementCase': 31, 'ArrayCreator': 32, 'This': 33, 'ConstructorDeclaration': 34,
     'TypeArgument': 35, 'EnhancedForControl': 36, 'SuperMethodInvocation': 37, 'SynchronizedStatement': 38,
     'DoStatement': 39, 'InnerClassCreator': 40, 'ExplicitConstructorInvocation': 41, 'BreakStatement': 42,
     'ClassReference': 43, 'SuperConstructorInvocation': 44, 'ElementValuePair': 45, 'AssertStatement': 46,
     'ElementArrayValue': 47, 'TypeParameter': 48, 'FieldDeclaration': 49, 'SuperMemberReference': 50,
     'ContinueStatement': 51, 'ClassDeclaration': 52, 'TryResource': 53, 'MethodReference': 54,
     'LambdaExpression': 55, 'InferredFormalParameter': 56}

def getedge_flow(node,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child,src,tgt,edgetype,ifedge,whileedge,foredge)


def getedge_nextstmt(node,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append([edges['Nextstmt']])
    for child in node.children:
        getedge_nextstmt(child,src,tgt,edgetype)


def getedge_nexttoken(node,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,tokenlist):
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,tokenlist)
    gettokenlist(node,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append([edges['Nexttoken']])


def getedge_nextuse(node,src,tgt,edgetype,variabledict):
    def getvariables(node,variabledict):
        token=node.token
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[variablenode.id]
            else:
                variabledict[variable].append(variablenode.id)
        for child in node.children:
            getvariables(child,variabledict)
    getvariables(node,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append([edges['Nextuse']])


def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
              (w, u, 32))
    return sum(x for u, v, x in combos if v in G[u])


def traids(graph,triad_nodes):

    TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9,
                9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9,
                9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15,
                11, 15, 15, 16)

    TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                  '030T', '030C', '201', '120D', '120U', '120C', '210', '300')


    TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}

    G = graph

    m = {v: i for i, v in enumerate(G)}
    for v in G:
        vnbrs = set(G.pred[v]) | set(G.succ[v])
        for u in vnbrs:
            if m[u] > m[v]:
                unbrs = set(G.pred[u]) | set(G.succ[u])
                neighbors = (vnbrs | unbrs) - {u, v}

                for w in neighbors:
                    if m[u] < m[w] or (m[v] < m[w] < m[u] and
                                       v not in G.pred[w] and
                                       v not in G.succ[w]):
                        code = _tricode(G, v, u, w)
                        triad_nodes[TRICODE_TO_NAME[code]].add(tuple(sorted([u, v, w])))
    return triad_nodes

def get_token(node):
    token = ''

    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))


def createtree(root,node,nodelist,typedict,nodedict,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)

    nodelist.append(node)
    nodedict[id] = token

    if len(children) == 0:
        try:
           tokentype = typedict[token]
        except KeyError:
            if token not in nodetypedict:
                tokentype = 'String'
                #print(id)
            else:
                tokentype = token
        nodedict[id] = tokentype
    for child in children:
        if id==0:
            createtree(root,child, nodelist,typedict,nodedict,parent=root)
        else:
            createtree(root,child, nodelist,typedict,nodedict,parent=newnode)


def getnodeandedge_astonly(node,src,tgt):

    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)

        getnodeandedge_astonly(child,src,tgt)


def getnodeandedge(node,src,tgt,edgetype):

    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])

        getnodeandedge(child,src,tgt,edgetype)


def getpaths(dirname):
    paths=[]

    #dirname = './test/'
    #dirname = '/home/data4T/wym/fsl/traids/dataset/id2sourcecode/'
    for rt, dirs, files in os.walk(dirname):
        for file in files:
            paths.append(os.path.join(rt, file))
    return paths


def onefaast(path, outpath):
    print(path)
    programfile = open(path, encoding='utf-8')
    # print(os.path.join(rt,file))
    programtext = programfile.read()
    # programtext=programtext.replace('\r','')
    programtokens = javalang.tokenizer.tokenize(programtext)
    # print(list(programtokens))
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()
    programfile.close()

    file = open(path, encoding='utf-8')
    tokens = list(javalang.tokenizer.tokenize(file.read()))
    file.close()
    # 生成类型字典
    typedict = {}
    for token in tokens:
        token_type = str(type(token))[:-2].split(".")[-1]
        token_value = token.value
        if token_value not in typedict:
            typedict[token_value] = token_type
        else:
            if typedict[token_value] != token_type:
                print('!!!!!!!!')
    # 生成树
    nodelist = []
    nodedict = {}
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, programast, nodelist, typedict, nodedict)

    edgesrc = []
    edgetgt = []
    edge_attr = []

    ifedge = True
    whileedge = True
    foredge = True
    blockedge = True
    nexttoken = True
    nextuse = True
    getnodeandedge(newtree, edgesrc, edgetgt, edge_attr)
    # if nextsib==True:
    #     getedge_nextsib(newtree,edgesrc,edgetgt,edge_attr)
    getedge_flow(newtree, edgesrc, edgetgt, edge_attr, ifedge, whileedge, foredge)
    if blockedge == True:
        getedge_nextstmt(newtree, edgesrc, edgetgt, edge_attr)
    tokenlist = []
    if nexttoken == True:
        getedge_nexttoken(newtree, edgesrc, edgetgt, edge_attr, tokenlist)
    variabledict = {}
    if nextuse == True:
        getedge_nextuse(newtree, edgesrc, edgetgt, edge_attr, variabledict)

    edge_index = list(zip(edgesrc, edgetgt))
    edge, edgetype, nodelist = edge_index, edge_attr, nodedict

    TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                   '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
    triad_nodes = {name: set([]) for name in TRIAD_NAMES}

    G = nx.DiGraph()
    G.add_nodes_from([key for key in nodelist])
    G.add_edges_from(edge)
    traids(G, triad_nodes)
    triads = []
    getstring('021D', '4', triad_nodes, nodedict, triads)
    getstring('021U', '5', triad_nodes, nodedict, triads)
    getstring('021C', '6', triad_nodes, nodedict, triads)
    getstring('111D', '7', triad_nodes, nodedict, triads)
    getstring('111U', '8', triad_nodes, nodedict, triads)
    getstring('030T', '9', triad_nodes, nodedict, triads)
    getstring('030C', '10', triad_nodes, nodedict, triads)
    getstring('120D', '12', triad_nodes, nodedict, triads)
    getstring('120U', '13', triad_nodes, nodedict, triads)
    getstring('120C', '14', triad_nodes, nodedict, triads)
    #print(triads)
    name = path.split('/')[-1].split('.java')[0]
    txtname = outpath + name + '.txt'
    with open(txtname, 'w') as f:
        for line in triads:
            f.write(line + ',')


def getstring(name, id, triad_nodes, nodedict, triads):
    if len(triad_nodes[name]) != 0:
        if id == '4' or id == '12':
            for t in triad_nodes[name]:
                if tokenindex[nodedict[t[1]]] >= tokenindex[nodedict[t[2]]]:
                    triad = str(id + '/' + nodedict[t[0]] + '/' + nodedict[t[1]] + '/' + nodedict[t[2]])
                else:
                    triad = str(id + '/' + nodedict[t[0]] + '/' + nodedict[t[2]] + '/' + nodedict[t[1]])
                triads.append(triad)
        elif id == '5' or id == '13':
            for t in triad_nodes[name]:
                if tokenindex[nodedict[t[0]]] >= tokenindex[nodedict[t[1]]]:
                    triad = str(id + '/' + nodedict[t[0]] + '/' + nodedict[t[1]] + '/' + nodedict[t[2]])
                else:
                    triad = str(id + '/' + nodedict[t[1]] + '/' + nodedict[t[0]] + '/' + nodedict[t[2]])
                triads.append(triad)
        elif id == '10':
            for t in triad_nodes[name]:
                if tokenindex[nodedict[t[0]]] >= tokenindex[nodedict[t[1]]] and tokenindex[nodedict[t[0]]] >= tokenindex[nodedict[t[2]]]:
                    triad = str(id + '/' + nodedict[t[0]] + '/' + nodedict[t[1]] + '/' + nodedict[t[2]])
                elif tokenindex[nodedict[t[1]]] >= tokenindex[nodedict[t[0]]] and tokenindex[nodedict[t[1]]] >= tokenindex[nodedict[t[2]]]:
                    triad = str(id + '/' + nodedict[t[1]] + '/' + nodedict[t[2]] + '/' + nodedict[t[0]])
                elif tokenindex[nodedict[t[2]]] >= tokenindex[nodedict[t[0]]] and tokenindex[nodedict[t[2]]] >= tokenindex[nodedict[t[1]]]:
                    triad = str(id + '/' + nodedict[t[2]] + '/' + nodedict[t[0]] + '/' + nodedict[t[1]])
                triads.append(triad)
        else:
            for t in triad_nodes[name]:
                triad = str(id + '/' + nodedict[t[0]] + '/' + nodedict[t[1]] + '/' + nodedict[t[2]])
                triads.append(triad)


with open('1tokenindex_dict.json', 'r', encoding='utf8') as fp:
    tokenindex = json.load(fp)
if __name__ == '__main__':
    #onefaast('./test/2.java')
    start1 = time.time()

    paths = getpaths('./BCB/')
    # pool = Pool(8)
    # pool.map(partial(onefaast), paths)
    for path in paths:
        onefaast(path, './triadstxt/')

    end1 = time.time()
    t1 = end1 - start1
    print('gettxt time:')
    print(t1)

