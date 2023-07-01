import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_names = ['Union', 'Empty', 'Equal', 'Complement', 'Intersect', 'Subtract']
# file_names = ['Union', 'Subtract']
func_line_map = {'Union': 3, 'Empty': 2, 'Equal': 3,
                 'Complement': 2, 'Intersect': 3, 'Subtract': 3}
isl_case_map = {'Union': 3, 'Empty': 1, 'Equal': 3,
                'Complement': 1, 'Intersect': 3, 'Subtract': 3}
fpl_case_map = {'Union': 3, 'Empty': 1, 'Equal': 2,
                'Complement': 2, 'Intersect': 3, 'Subtract': 3}
fpl_simplify_case_map = {'Union': 3, 'Empty': 1, 'Equal': 3,
                         'Complement': 1, 'Intersect': 3, 'Subtract': 3}
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
        # for _ in range(fpl_result_map[func]):
            # case.pop()
        for _ in range(fpl_number_map[func]):
            id += 1
        result.append(case)
    return result


def getFPLSimplifyRelation(func):
    file_name = './fpl_simplify_relation/PresburgerSet'+func+'_fpl_simplify_relation'
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
        for _ in range(fpl_simplify_case_map[func]):
            onecase, id = parseOneRelation(res, id)
            case.append(onecase)
        result.append(case)
    return result


def basiccompare(func_name):
    fpl_file_name = 'PresburgerSet'+func_name+'_fpl_info.csv'
    fpl_simplify_file_name = 'PresburgerSet'+func_name+'_fpl_simplify_info.csv'
    isl_file_name = 'PresburgerSet'+func_name+'_isl_info.csv'

    fpl_info = pd.read_csv(fpl_file_name, index_col=0)
    fpl_simplify_info = pd.read_csv(fpl_simplify_file_name, index_col=0)
    isl_info = pd.read_csv(isl_file_name, index_col=0)
    info = fpl_info
    info = info.rename(
        columns={'size': 'fpl_size', 'time': 'fpl_time', 'result_size': 'fpl_result_size'})
    info['isl_size'] = isl_info['size']
    info['isl_time'] = isl_info['time']
    info['isl_result_size'] = isl_info['result_size']
    info['fpl_simplify_size'] = fpl_simplify_info['size']
    info['fpl_simplify_time'] = fpl_simplify_info['time']
    info['fpl_simplify_result_size'] = fpl_simplify_info['result_size']
    info['diff_time'] = info['fpl_time']-info['isl_time']
    info['diff_size'] = info['fpl_size']-info['isl_size']
    info['diff_simplify_time'] = info['fpl_simplify_time']-info['isl_time']
    info['diff_simplify_size'] = info['fpl_simplify_size']-info['isl_size']
    info['diff_optim_time'] = info['fpl_time']-info['fpl_simplify_time']
    info['diff_optim_size'] = info['fpl_size']-info['fpl_simplify_size']
    print(info)
    return info


def fig_size(func_name, info, image_path):
    plt.scatter(info['fpl_size'], info['isl_size'], s=5)
    plt.xlabel('fpl')
    plt.ylabel('isl')
    plt.savefig(image_path+func_name+'_size.png')
    plt.clf()


def fig_time(func_name, info, image_path):
    plt.scatter(info['fpl_time'], info['isl_time'], s=5)
    plt.xlabel('fpl')
    plt.ylabel('isl')
    plt.savefig(image_path+func_name+'_time.png')
    plt.clf()


def fig_size_time(func_name, info, image_path):
    plt.scatter(info['fpl_size'], info['fpl_time'],
                s=5, marker='o', label='fpl')
    plt.scatter(info['isl_size'], info['isl_time'],
                s=5, marker='^', label='isl')
    # plt.hist(info['fpl_size'], bins=20, log=True, alpha=0.7, linewidth=1.2)
    plt.xlabel('size')
    plt.ylabel('time')
    plt.legend()
    # sns.histplot(data=info, x='fpl_size')
    plt.savefig(image_path+func_name+'_size_time.png')
    plt.clf()


def fig_diff_time_size(func_name, info, image_path):

    plt.scatter(info['fpl_size'], info['diff_time'],
                s=5, label='diff time vs fpl')
    plt.scatter(info['isl_size'], info['diff_time'],
                s=5, label='diff time vs isl')
    plt.xlabel('size')
    plt.ylabel('diff_time')
    plt.legend()
    plt.savefig(image_path+func_name+'_difftime_size.png')
    plt.clf()


def fig_diff_time(func_name, info, image_path):
    plt.hist(info['diff_time'], bins=20, log=True)
    plt.xlabel('diff time')
    plt.ylabel('count')
    plt.savefig(image_path+func_name+'_difftime.png')
    plt.clf()


def fig_diff_size(func_name, info, image_path):
    plt.hist(info['diff_size'], bins=20, log=True)
    plt.xlabel('diff size')
    plt.ylabel('count')
    plt.savefig(image_path+func_name+'_diffsize.png')
    plt.clf()


def fig_time_line(func_name, info, image_path):
    isl_times = info['isl_time'].sort_values()
    fpl_times = info['fpl_time'].sort_values()
    fpl_simplify_times = info['fpl_simplify_time'].sort_values()

    x = np.arange(0, isl_times.size, 1)
    full_size = round(isl_times.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, fpl_times, '--', label='fpl')
    plt.plot(x, fpl_simplify_times, '--', label='fpl_optim')
    plt.plot(x, isl_times, '--', label='isl')
    plt.yscale('log')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(f'{func_name} test case, sorted by run time seprately')
    plt.ylabel('Run time (ns)')
    plt.savefig(image_path+func_name+'_time.png')
    plt.clf()


def fig_time_by_isl_line(func_name, info, image_path):
    new_info = info.sort_values(by='fpl_time')
    isl_times = new_info['isl_time']
    fpl_times = new_info['fpl_time']
    fpl_simplify_times = new_info['fpl_simplify_time']
    print(new_info)

    x = np.arange(0, isl_times.size, 1)
    full_size = round(isl_times.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, fpl_times, '--', label='fpl')
    plt.plot(x, fpl_simplify_times, '--', label='fpl_optim')
    plt.plot(x, isl_times, '--', label='isl')
    plt.yscale('log')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(f'{func_name} test case, sorted by run time in fpl')
    plt.ylabel('Run time (ns)')
    plt.savefig(image_path+func_name+'_time_by_isl.png')
    plt.clf()


def fig_size_line(func_name, info, image_path):
    isl_sizes = info['isl_size'].sort_values()
    fpl_sizes = info['fpl_size'].sort_values()
    fpl_simplify_sizes = info['fpl_simplify_size'].sort_values()

    x = np.arange(0, isl_sizes.size, 1)
    full_size = round(isl_sizes.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, fpl_sizes, '--', label='fpl')
    plt.plot(x, fpl_simplify_sizes, '--', label='fpl_optim')
    plt.plot(x, isl_sizes, '--', label='isl')
    plt.yscale('symlog')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(f'{func_name} test case, sorted by constraint system size')
    plt.ylabel('Constraint system size')
    plt.savefig(image_path+func_name+'_size.png')
    plt.clf()


def fig_result_size_line(func_name, info, image_path):
    isl_sizes = info['isl_result_size'].sort_values()
    fpl_sizes = info['fpl_result_size'].sort_values()
    fpl_simplify_sizes = info['fpl_simplify_result_size'].sort_values()

    x = np.arange(0, isl_sizes.size, 1)
    full_size = round(isl_sizes.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, fpl_sizes, '--', label='fpl')
    plt.plot(x, fpl_simplify_sizes, '--', label='fpl_optim')
    plt.plot(x, isl_sizes, '--', label='isl')
    plt.yscale('symlog')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(f'{func_name} test case, sorted by calculate result size')
    plt.ylabel('Calculate result size')
    plt.savefig(image_path+func_name+'_result_size.png')
    plt.clf()


def fig_diff_size_line(func_name, info, image_path):
    diff_sizes = info['diff_size'].sort_values()
    diff_optim_sizes = info['diff_simplify_size'].sort_values()

    x = np.arange(0, diff_sizes.size, 1)
    full_size = round(diff_sizes.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, diff_sizes, '--', label='fpl vs isl')
    plt.plot(x, diff_optim_sizes, '--', label='fpl_optim vs isl')
    plt.yscale('log')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(
        f'{func_name} test case, sorted by constraint system difference\n (larger means more space is occupied)')
    plt.ylabel('Constraint system diffeerence')
    plt.savefig(image_path+func_name+'_diff_size.png')
    plt.clf()


def fig_diff_time_line(func_name, info, image_path):
    diff_times = info['diff_time'].sort_values()
    diff_optim_times = info['diff_simplify_time'].sort_values()

    x = np.arange(0, diff_times.size, 1)
    full_size = round(diff_times.size/1000)*1000
    half_size = full_size/2
    plt.figure(figsize=(8, 6))
    plt.plot(x, diff_times, '--', label='fpl vs isl')
    plt.plot(x, diff_optim_times, '--', label='fpl_optim vs isl')
    plt.yscale('log')
    plt.legend()
    plt.xticks([0, half_size, full_size])
    plt.xlabel(
        f'{func_name} test case, sorted by run time difference\n (larger means more time is used)')
    plt.ylabel('Run time difference (ns)')
    plt.savefig(image_path+func_name+'_diff_time.png')
    plt.clf()


def gen_top(func_name, info: pd.DataFrame, attr_name, isl_cases, fpl_cases, res_path):
    isl_size_info = info.sort_values(attr_name, ascending=False)
    isl_ids = isl_size_info[:20].index
    res = getexample(func_name, isl_ids, func_line_map[func_name])
    res_path = res_path+func_name+f'_{attr_name}_top20.txt'
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


def gen_simplify_top(func_name, info: pd.DataFrame, attr_name, isl_cases, fpl_simplify_cases, res_path):
    isl_size_info = info.sort_values(attr_name, ascending=False)
    isl_ids = isl_size_info[:20].index
    res = getexample(func_name, isl_ids, func_line_map[func_name])
    res_path = res_path+func_name+f'_{attr_name}_simplify_top20.txt'
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
            f.write(f'FPL Simplify Relation: \n')
            case = fpl_simplify_cases[isl_ids[i]]
            for onecase in case:
                for line in onecase:
                    f.write(line)


if __name__ == '__main__':
    for fun in file_names:
        info = basiccompare(fun)
        image_path = 'result/'
        top_path = 'topcase/'
        # fig_size(fun, info, image_path)
        # fig_time(fun, info, image_path)
        # fig_size_time(fun, info, image_path)
        # fig_diff_time_size(fun, info, image_path)
        # fig_diff_time(fun, info, image_path)
        # fig_diff_size(fun, info, image_path)
        # isl_cases = getISLRelation(fun)
        # fpl_cases = getFPLRelation(fun)
        # fpl_simplify_cases = getFPLSimplifyRelation(fun)
        # gen_top(fun, info, 'isl_size', isl_cases, fpl_cases)
        # gen_top(fun, info, 'fpl_size', isl_cases, fpl_cases)
        # gen_top(fun, info, 'diff_size', isl_cases, fpl_cases)
        # gen_top(fun, info, 'isl_time', isl_cases, fpl_cases)
        # gen_top(fun, info, 'fpl_time', isl_cases, fpl_cases)
        # gen_top(fun, info, 'diff_time', isl_cases, fpl_cases)
        # fig_time_line(fun, info, image_path)
        # fig_time_by_isl_line(fun, info, image_path)
        fig_size_line(fun, info, image_path)
        # fig_result_size_line(fun, info, image_path)
        # fig_diff_size_line(fun, info, image_path)
        # fig_diff_time_line(fun, info, image_path)
        # gen_top(fun, info, 'diff_optim_time', isl_cases, fpl_cases, top_path)
        # gen_top(fun, info, 'diff_size', isl_cases, fpl_cases, top_path)
        # gen_simplify_top(fun, info, 'diff_simplify_size', isl_cases,
        #                  fpl_simplify_cases, top_path)
