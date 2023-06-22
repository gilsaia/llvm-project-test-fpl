import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_names = ['Union', 'Empty', 'Equal', 'Complement', 'Intersect', 'Subtract']
func_line_map = {'Union': 3, 'Empty': 2, 'Equal': 3,
                 'Complement': 2, 'Intersect': 3, 'Subtract': 3}
isl_case_map = {'Union': 2, 'Empty': 1, 'Equal': 2,
                'Complement': 1, 'Intersect': 2, 'Subtract': 2}
fpl_case_map = {'Union': 3, 'Empty': 1, 'Equal': 2,
                'Complement': 2, 'Intersect': 3, 'Subtract': 3}
fpl_result_map = {'Union': 1, 'Empty': 0, 'Equal': 0,
                  'Complement': 1, 'Intersect': 1, 'Subtract': 1}
fpl_number_map = {'Union': 0, 'Empty': 1, 'Equal': 1,
                  'Complement': 0, 'Intersect': 0, 'Subtract': 0}


def getexample(func, ids, line):
    file_name = 'PresburgerSet'+func
    res = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for id in ids:
            res.append(lines[id*line+1:id*line+1+line])
    return res


def parseOneRelation(numbers, id):
    dims = numbers[id]
    id += 1
    symbols = numbers[id]
    id += 1
    maps = numbers[id]
    id += 1
    res = []
    res.append(f"{dims} {symbols} {maps}\n")
    for i in range(maps):
        eqs = numbers[id]
        id += 1
        ineqs = numbers[id]
        id += 1
        res.append(f'{eqs} {ineqs}\n')
        for _ in range(eqs):
            line = ''
            for _ in range(dims+symbols+1):
                line += f'{numbers[id]} '
                id += 1
            line += '\n'
            res.append(line)
        for _ in range(ineqs):
            line = ''
            for _ in range(dims+symbols+1):
                line += f'{numbers[id]} '
                id += 1
            line += '\n'
            res.append(line)
    return res, id


def getISLRelation(func):
    file_name = './isl_relation/PresburgerSet'+func+'_isl_relation'
    res = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if len(line) == 1:
                continue
            numbers = line.rstrip().split(' ')
            for number in numbers:
                res.append(int(number))
    size = int(res[0])
    id = 1
    result = []
    for _ in range(size):
        case = []
        for _ in range(isl_case_map[func]):
            onecase, id = parseOneRelation(res, id)
            case.append(onecase)
        result.append(case)
    return result


def getFPLRelation(func):
    file_name = './fpl_relation/PresburgerSet'+func+'_fpl_relation'
    res = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if len(line) == 1:
                continue
            numbers = line.rstrip().split(' ')
            for number in numbers:
                res.append(int(number))
    size = int(res[0])
    id = 1
    result = []
    for _ in range(size):
        case = []
        for _ in range(fpl_case_map[func]):
            onecase, id = parseOneRelation(res, id)
            case.append(onecase)
        for _ in range(fpl_result_map[func]):
            case.pop()
        for _ in range(fpl_number_map[func]):
            id += 1
        result.append(case)
    return result


def basiccompare(func_name):
    fpl_file_name = 'PresburgerSet'+func_name+'_fpl_info.csv'
    isl_file_name = 'PresburgerSet'+func_name+'_isl_info.csv'

    fpl_info = pd.read_csv(fpl_file_name, index_col=0)
    isl_info = pd.read_csv(isl_file_name, index_col=0)
    info = fpl_info
    info = info.rename(columns={'size': 'fpl_size', 'time': 'fpl_time'})
    info['isl_size'] = isl_info['size']
    info['isl_time'] = isl_info['time']
    info['diff_time'] = info['fpl_time']-info['isl_time']
    info['diff_size'] = info['fpl_size']-info['isl_size']
    print(info)
    return info


def fig_size(func_name, info):
    plt.scatter(info['fpl_size'], info['isl_size'], s=5)
    plt.xlabel('fpl')
    plt.ylabel('isl')
    plt.savefig(func_name+'_size.png')
    plt.clf()


def fig_time(func_name, info):
    plt.scatter(info['fpl_time'], info['isl_time'], s=5)
    plt.xlabel('fpl')
    plt.ylabel('isl')
    plt.savefig(func_name+'_time.png')
    plt.clf()


def fig_size_time(func_name, info):
    plt.scatter(info['fpl_size'], info['fpl_time'],
                s=5, marker='o', label='fpl')
    plt.scatter(info['isl_size'], info['isl_time'],
                s=5, marker='^', label='isl')
    # plt.hist(info['fpl_size'], bins=20, log=True, alpha=0.7, linewidth=1.2)
    plt.xlabel('size')
    plt.ylabel('time')
    plt.legend()
    # sns.histplot(data=info, x='fpl_size')
    plt.savefig(func_name+'_size_time.png')
    plt.clf()


def fig_diff_time_size(func_name, info):

    plt.scatter(info['fpl_size'], info['diff_time'],
                s=5, label='diff time vs fpl')
    plt.scatter(info['isl_size'], info['diff_time'],
                s=5, label='diff time vs isl')
    plt.xlabel('size')
    plt.ylabel('diff_time')
    plt.legend()
    plt.savefig(func_name+'_difftime_size.png')
    plt.clf()


def fig_diff_time(func_name, info):
    plt.hist(info['diff_time'], bins=20, log=True)
    plt.xlabel('diff time')
    plt.ylabel('count')
    plt.savefig(func_name+'_difftime.png')
    plt.clf()


def fig_diff_size(func_name, info):
    plt.hist(info['diff_size'], bins=20, log=True)
    plt.xlabel('diff size')
    plt.ylabel('count')
    plt.savefig(func_name+'_diffsize.png')
    plt.clf()


def gen_top(func_name, info: pd.DataFrame, attr_name, isl_cases, fpl_cases):
    isl_size_info = info.sort_values(attr_name, ascending=False)
    isl_ids = isl_size_info[:20].index
    res = getexample(func_name, isl_ids, func_line_map[func_name])
    res_path = func_name+f'_{attr_name}_top20.txt'
    # print(isl_size_info[attr_name][isl_ids[2]])
    with open(res_path, 'w') as f:
        for i in range(20):
            f.write(
                f'Id:{isl_ids[i]} {attr_name}:{isl_size_info[attr_name][isl_ids[i]]} \n')
            f.write(f'Origin Relation: \n')
            for line in res[i]:
                f.write(line)
            f.write(f'ISL Relation: \n')
            case = isl_cases[isl_ids[i]]
            for onecase in case:
                for line in onecase:
                    f.write(line)
            f.write(f'FPL Relation: \n')
            case = fpl_cases[isl_ids[i]]
            for onecase in case:
                for line in onecase:
                    f.write(line)


if __name__ == '__main__':
    for fun in file_names:
        info = basiccompare(fun)
        fig_size(fun, info)
        fig_time(fun, info)
        fig_size_time(fun, info)
        fig_diff_time_size(fun, info)
        fig_diff_time(fun, info)
        fig_diff_size(fun, info)
        isl_cases = getISLRelation(fun)
        fpl_cases = getFPLRelation(fun)
        gen_top(fun, info, 'isl_size', isl_cases, fpl_cases)
        gen_top(fun, info, 'fpl_size', isl_cases, fpl_cases)
        gen_top(fun, info, 'diff_size', isl_cases, fpl_cases)
        gen_top(fun, info, 'isl_time', isl_cases, fpl_cases)
        gen_top(fun, info, 'fpl_time', isl_cases, fpl_cases)
        gen_top(fun, info, 'diff_time', isl_cases, fpl_cases)
