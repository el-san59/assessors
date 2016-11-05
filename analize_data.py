import pandas as pd
import matplotlib.pyplot as plt
import json
import random


data = pd.read_table('assessors_data')
names = {}
pd.read_csv('reformed_assessors_data.csv', index_col='id')


def txt_to_int(feature):
    values = sorted(list(set(data[feature])))
    d = dict(zip(values, range(len(values))))
    names[feature] = d
    data[feature] = data[feature].map(lambda x: d[x])


def set_random_values(feature):
    with_data = data.loc[data[feature] != '']
    with_data = with_data[feature]
    empty = data.loc[data[feature] == '']
    empty = empty[feature]
    if feature in [13, 15]:
        with_data = with_data.map(float)
        value = round(with_data.sum()/len(with_data), 4)
        print(value)
        for i in empty.index:
            data.set_value(i, feature, value)
    elif feature in [1, 5, 16, 17]:
        with_data = with_data.map(int)
        value = with_data.sum() // len(with_data)
        print(value)
        for i in empty.index:
            data.set_value(i, feature, value)
    else:
        for i in empty.index:
            value = with_data[random.choice(with_data.index)]
            data.set_value(i, feature, value)
    print(list(data[feature]))


def convert_data():
    global data
    data = data.fillna('')
    data.columns = range(20)
    data = data.applymap(lambda x: x.strip().lower() if type(x) == str else x)
    # score for test
    set_random_values(1)
    # open questions
    data.set_value(84, 2, 'средние')
    data.set_value(82, 2, '')
    data.set_value(113, 2, '')
    data.set_value(118, 2, 'средние')
    set_random_values(2)
    txt_to_int(2)
    # test S/N
    data.set_value(39, 3, '')
    set_random_values(3)
    txt_to_int(3)
    # sex
    set_random_values(4)
    txt_to_int(4)
    # year of birth
    set_random_values(5)
    # education
    data[6] = data[6].map(lambda x: 'сред' if x=='среднее' else x)
    set_random_values(6)
    txt_to_int(6)
    # speciality
    set_random_values(7)
    txt_to_int(7)
    # job
    set_random_values(8)
    txt_to_int(8)
    # attempts
    set_random_values(9)
    # impression admin
    set_random_values(10)
    txt_to_int(10)
    # source
    set_random_values(11)
    txt_to_int(11)
    # recruit
    data.set_value(54, 12, 'нет')
    data.set_value(72, 12, 'нет')
    data.set_value(76, 12, 'нет')
    data.set_value(77, 12, 'нет')
    data.set_value(59, 12, 'да')
    set_random_values(12)
    txt_to_int(12)
    # Gset exam accuracy
    data.set_value(145, 13, '')
    data.set_value(39, 13, '')
    data.set_value(42, 13, '')
    data.set_value(46, 13, '')
    set_random_values(13)
    # % Gset exam
    data.set_value(39, 14, '')
    values = list(data[14])
    for i in range(len(values)):
        if '/' in values[i]:
            f, s = values[i].split('/')
            data.set_value(i, 14, float(f) / float(s))
    empty = data.loc[data[14] == '']
    for i in empty.index:
        data.set_value(i, 14, 1)
    # Gset training accuracy
    data.set_value(39, 15, '')
    data.set_value(42, 15, '')
    data.set_value(46, 15, '')
    set_random_values(15)
    # time spent on training
    data.set_value(39, 16, '')
    data.set_value(42, 16, '')
    data.set_value(46, 16, '')
    values = list(data[16])
    ftr = [3600, 60, 1]
    for i in range(len(values)):
        if ' ' in values[i]:
            values[i] = values[i].split()[1]
        if ':' in values[i]:
            data.set_value(i, 16, sum([a * b for a, b in zip(ftr, map(int, values[i].split(':')))]))
    set_random_values(16)
    # time spent on exam
    data.set_value(39, 17, '')
    data.set_value(42, 17, '')
    data.set_value(46, 17, '')
    values = list(data[17])
    ftr = [3600, 60, 1]
    for i in range(len(values)):
        if ' ' in values[i]:
            values[i] = values[i].split()[1]
        if ':' in values[i]:
            data.set_value(i, 17, sum([a * b for a, b in zip(ftr, map(int, values[i].split(':')))]))
    set_random_values(17)
    return data


def show_dependence(feature):
    df = data[[feature, 12]]
    yes = df.loc[df[12] == 1].groupby([feature]).count()
    no = df.loc[df[12] == 0].groupby([feature]).count()
    no.columns = ['no']
    yes.columns = ['yes']
    res = no.join(yes, how='outer')
    res.plot.bar()
    plt.show()


def save_convert_data():
    new_data = convert_data()
    with open('names.json', 'w') as f:
        json.dump(names, f)
    new_data = new_data.ix[:, 1:17]
    new_data = new_data.applymap(lambda x: float(x))
    print(names)
    new_data.to_csv('reformed_assessors_data.csv', index_label='id')


if __name__ == "__main__":
    # convert_data()
    # print(data)
    save_convert_data()












