import logging
#%%
import yaml



logger = logging.getLogger(__name__)
data = dict(
    A = 'a',
    B = dict(
        C = 'c',
        D = 'd',
        E = 'e',
    )
)

data = {}




    
def read_txt_to_dict(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    columns = [line.strip().split('$') for line in lines]
    if not columns or len(columns) < 2:
        return {}
    headers = [h.strip() for h in columns[0]]
    data_rows = columns[1:]
    # Remove spaces from all entries
    cleaned_rows = [[entry.strip().replace(' ', '') for entry in row] for row in data_rows]
    transposed = list(zip(*cleaned_rows))
    col_dict = {header: list(col) for header, col in zip(headers, transposed)}
    return col_dict

txt_dict = read_txt_to_dict('/Users/prioletp/PhD/public_codes/pyGrater/parameters/material_list.txt')
logger.info(txt_dict['Nickname'])

Weight_par = []
Weight_per1 = [] #txt_dict['Weight_per1']
Weight_per2 = [] #txt_dict['Weight_per2']

for elem in txt_dict['Weight_par']:
    Weight_par.append(float(elem))
for elem in txt_dict['Weight_per1']:
    Weight_per1.append(float(elem))
for elem in txt_dict['Weight_per2']:
    Weight_per2.append(float(elem))

weights = []
if len(Weight_par) == len(Weight_per1) == len(Weight_per2):
    logger.info('All weights have the same length')
    for index in range(len(Weight_par)):
        weights.append([Weight_par[index], Weight_per1[index], Weight_per2[index]])
else:
    logger.info('Weights do not have the same length')
# print(weights)

dictionary_for_yaml = {}
for index, nickname in enumerate(txt_dict['Nickname']):
    dictionary_for_yaml[nickname] = weights[index]

logger.info(dictionary_for_yaml)
with open('weights.yaml', 'w') as outfile:
    yaml.dump(dictionary_for_yaml, outfile, default_flow_style=False)
    
# %%
