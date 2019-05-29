from string import Formatter
'''
class UnseenFormatter(Formatter):
    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            sep = key.find('|')
            if sep != -1:
                key_kw = key[sep + 1:]
                key_arg = key[:sep]

            if key_arg in kwds:
                return kwds[key_kw]
            else:
                return super(UnseenFormatter, self).get_value(key_arg, args, kwds)
        else:
            print(key,type(key))
            return super(UnseenFormatter, self).get_value(key, args, kwds)
'''

def extract_format_strings(form):
    '''Extracts the formatting string curly braces posisitons
    
    string = "{2|number_of_sheep} sheep {0|has} run away"
    form_v = extract_format_strings(string)
    print(form_v)
    for x,y in form_v:
        print(string[x:y])

    gives:
        2|number_of_sheep
        0|has
    '''

    form = ' '+form
    form_v = []
    ind = 0
    while ind < len(form) and ind >= 0:
        ind_open = form.find('{',ind+1)
        ind = ind_open
        if ind_open > 0:
            ind_close = form.find('}',ind+1)
            if ind_close > 0:
                form_v.append((ind_open, ind_close-1))
            ind = ind_close

    return form_v

def extract_format_keys(form):
    form_inds = extract_format_strings(form)

    keys = []
    for start, stop in form_inds:
        form_arg = form[start:stop]

        if form_arg.index('|') > 0:
            keys.append(tuple(form_arg.split('|')))
        else:
            raise Exception('Log call formating only works with key | formating')

    return form_inds, keys

def construct_formatted_format(form, kwargs):
    form_inds, keys = extract_format_keys(form)

    form_str = ''
    cntr = 0
    last_ind = 0

    for start,stop in form_inds:
        form_str += form[last_ind : start]

        ind, key = keys[cntr]

        if key.find('.') > 0:
            key_check = key[:key.find('.')]
        else:
            key_check = key

        if key_check in kwargs:
            form_str += key
        else:
            form_str += ind

        last_ind = stop
        cntr += 1
    form_str += form[last_ind :]

    return form_str

string = "{2.x|number_of_sheep.x} sheep {0|has} run away"

form_v = extract_format_strings(string)
print(form_v)
for x, y in form_v:
    print(string[x:y])

print(string)
print('becomes')
print(construct_formatted_format(string,{'number_of_sheep': 5}))


def log_call(form, logger):

    def log_call_decorator(method):
        def logged_fn(*args, **kwargs):

            form_str = construct_formatted_format(form, kwargs)

            logger.always('{}: {}'.format(
                method.__name__, 
                form_str.format(*args, **kwargs)
                ))
            return method(*args, **kwargs)

        return logged_fn
    return log_call_decorator


'''
string = "{2|number_of_sheep} sheep {0|has} run away"
other_dict = {'number_of_sheep' : 1}
args = [3.15, 4]

fmt = UnseenFormatter()
print(fmt.format(string, *args, **other_dict))

class a:
    def __init__(self):
        x = 5

A = a()

string = "{1.x|number_of_sheep.x} sheep {0|has} run away"
other_dict = {}
args = [3.15,A]
print(fmt.format(string, *args, **other_dict))

string = "{1.x|number_of_sheep.x} sheep {0|has} run away"
other_dict = {'number_of_sheep' : A}
args = [3.15]
print(fmt.format(string, *args, **other_dict))
'''