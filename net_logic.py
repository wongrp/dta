

"""

KEY
p -- pocket 
r -- residue 
l -- ligand 
s -- surface 
v -- virtual nodes
c -- complex (pocket+ligand)
x -- complex (residue+ligand)
z -- complex (residue+pocket)

['p', 'r', 'l', 'c', 'x','z'] 
pp ll cpl cpp cll clp rl lr 
transformer, mlp 
"""
from net_l_schedule import get_l_schedule
from args import get_args 

args = get_args() 
MODES = {
    'plc': {
        'field': ['l', 'p', 'c'],
        'schedule': [['ll','pp','cc']] * args.layers
    },
    'rlx': {
        'field': ['r', 'l', 'x'],
        'schedule': [['rr','ll','xx']] * args.layers
    },
    'rlp': {
        'field': ['r', 'l', 'p'],
        'schedule': [['rr','ll','pp']] * args.layers
    },
    'rl': {
        'field': ['r', 'l'],
        'schedule': [['rr','ll']] * args.layers
    },
    'rlpz': 
    {
        'field': ['r', 'l','p','z'],
        'schedule': [['rr','ll', 'pp','zz']] * args.layers
    },
}

""" 
If a list has more than one element, there should be an option to turn it off.

proximity
ligand bond
hitting / commute times 
""" 
edge_sets = {'all': ['prox','lig_bond'],
            'rr': ['prox'],
            'll': ['prox','lig_bond'],
            'pp': ['prox'],
            'xx': ['prox'],
            'zz': ['prox'],
            'cc': ['prox'],
            }




# fusion_type = 'cross_att'  # https://arxiv.org/html/2402.17906v1
l_schedule = 'lmax1_ascending'
l_schedule = get_l_schedule(l_schedule,args.ns,args.ns,args.nv)

field = MODES[args.mode]['field']
schedule = MODES[args.mode]['schedule']




"""
- freeze weights
"""