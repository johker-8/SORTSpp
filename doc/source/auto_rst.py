import glob
import os

py_list = glob.glob('../../*.py')

dev_lst = [ x for x in py_list if x.split('/')[-1][:3] == 'DEV']
source_lst = [ x for x in py_list if x.split('/')[-1][:3] != 'DEV']
source_lst_names = [ x.split('/')[-1].split('.')[0] for x in py_list if x.split('/')[-1][:3] != 'DEV']

print('source_lst')
print(source_lst)

print('dev_lst')
print(dev_lst)

template = '{}\n'
template += '===========================\n'
template += '\n'
template += '.. automodule:: {}\n'
template += '   :members:\n'
template += '   :undoc-members:\n'
template += '   :show-inheritance:\n'

file_names = []
for source in source_lst_names:
    file_names.append(source + '.rst')
    with open(file_names[-1],'w') as f:
        f.write(template.format(
            source,
            source,
            ))


for file in file_names:
    print(file.split('.')[0])
    