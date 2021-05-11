from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)






DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# GDAS = Genotype(normal=[
#   ('dil_sepc_5x5', 0, 0.12614432),
#   ('dua_sepc_5x5', 1, 0.12614417),
#   ('skip_connect', 1, 0.12631783),
#   ('dua_sepc_5x5', 2, 0.12604922),
#   ('max_pool_3x3', 1, 0.12643974),
#   ('dua_sepc_5x5', 0, 0.12637283),
#   ('dua_sepc_5x5', 1, 0.12598419),
#   ('dil_sepc_3x3', 3, 0.12584162)], normal_concat=[2, 3, 4, 5],
#   reduce=[
#     ('dua_sepc_5x5', 0, 0.12726927),
#     ('dua_sepc_5x5', 1, 0.12616388),
#     ('dua_sepc_5x5', 2, 0.12644506),
#     ('dua_sepc_5x5', 0, 0.12632592),
#     ('dua_sepc_5x5', 0, 0.126043),
#     ('dua_sepc_5x5', 2, 0.12571044),
#     ('dua_sepc_5x5', 4, 0.1259345),
#     ('dua_sepc_5x5', 0, 0.12586214)], reduce_concat=[2, 3, 4, 5])

GDAS = Genotype(normal=[
  ('dua_sepc_5x5', 0, 0.12580751),
  ('dua_sepc_5x5', 1, 0.12563181),
  ('skip_connect', 1, 0.1255888),
  ('dua_sepc_5x5', 0, 0.12554474),
  ('skip_connect', 1, 0.125992),
  ('dil_sepc_5x5', 0, 0.12582529),
  ('dua_sepc_5x5', 0, 0.12667581),
  ('dil_sepc_5x5', 4, 0.12602767)],
  normal_concat = [2, 3, 4, 5], reduce=[
  ('dua_sepc_5x5', 0, 0.1271657),
  ('dua_sepc_3x3', 0, 0.12629774),
  ('dua_sepc_5x5', 1, 0.1264101),
  ('dua_sepc_3x3', 0, 0.12624726),
  ('dua_sepc_5x5', 1, 0.12614323),
  ('dua_sepc_5x5', 0, 0.12605816),
  ('dua_sepc_5x5', 4, 0.12598045),
  ('max_pool_3x3', 0, 0.1258678)],
  reduce_concat = [2, 3, 4, 5]
)

# GDAS = Genotype(normal=[
#   ('max_pool_3x3', 0, 0.12906693),
#   ('skip_connect', 1, 0.1276831),
#   ('max_pool_3x3', 0, 0.13194221),
#   ('skip_connect', 0, 0.12970467),
#   ('max_pool_3x3', 0, 0.13137783),
#   ('skip_connect', 0, 0.13007157),
#   ('max_pool_3x3', 0, 0.1333811),
#   ('skip_connect', 0, 0.13003603)],
#   normal_concat = [2, 3, 4, 5], reduce=[
#     ('max_pool_3x3', 0, 0.12812603),
#     ('max_pool_3x3', 1, 0.12732728),
#     ('max_pool_3x3', 0, 0.12968308),
#     ('max_pool_3x3', 1, 0.12849186),
#     ('max_pool_3x3', 0, 0.13088289),
#     ('max_pool_3x3', 1, 0.12931313),
#     ('max_pool_3x3', 0, 0.13195509),
#     ('max_pool_3x3', 1, 0.12923086)],
#   reduce_concat = [2, 3, 4, 5]
# )


{'normal': [], 'normal_concat': [2, 3, 4, 5], 'reduce': [], 'reduce_concat': [2, 3, 4, 5]}



DARTS = GDAS








