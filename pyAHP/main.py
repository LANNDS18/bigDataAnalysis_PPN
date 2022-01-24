import numpy as np


def _2darr(y):
    return np.array([[1, y], [1 / y, 1]])


def _5darr(*df):
    if len(df) != 25:
        raise Exception("bad-request")
    a = np.array(df).reshape((5, 5))
    return a


def construct(*df, person_name):
    for element in df[0]:
        small = _2darr(element)
        person_name.append(small)
    return person_name


def avg(*df):
    df = df[0]
    res = 1
    for n in df:
        res *= n
    return np.power(res, 1 / len(df))


person = []

wxh = [_5darr(1.0000, 4.0000, 3.0000, 7.0000, 6.0000,
              0.2500, 1.0000, 0.2500, 2.0000, 3.0000,
              0.3333, 4.0000, 1.0000, 6.0000, 3.0000,
              0.1429, 0.5000, 0.1667, 1.0000, 2.0000,
              0.1667, 0.3333, 0.3333, 0.5000, 1.0000)]
mn_wxh = [8, 0.1429, 0.3333, 0.1429, 0.1667]
person.append(construct(mn_wxh, person_name=wxh))

lzx = [_5darr(1.0000, 4.0000, 8.0000, 8.0000, 4.0000,
              0.2500, 1.0000, 7.0000, 7.0000, 3.0000,
              0.1250, 0.1429, 1.0000, 1.0000, 0.1667,
              0.1250, 0.1429, 1.0000, 1.0000, 0.1667,
              0.2500, 0.3333, 6.0000, 6.0000, 1.0000)]
mn_lzx = [1, 0.1667, 1, 0.1667, 0.125]
person.append(construct(mn_lzx, person_name=lzx))

wgy = [_5darr(1.0000, 1.0000, 0.1667, 0.2500, 0.2000,
              1.0000, 1.0000, 0.1429, 0.3333, 0.3333,
              6.0000, 7.0000, 1.0000, 6.0000, 4.0000,
              4.0000, 3.0000, 0.1667, 1.0000, 0.3333,
              5.0000, 3.0000, 0.2500, 3.0000, 1.0000)]
mn_wgy = [5, 8, 0.1429, 0.1429, 0.1111]
person.append(construct(mn_wgy, person_name=wgy))

jhf = [_5darr(1.0000, 3.4464, 6.4466, 6.5534, 3.5535,
              0.2902, 1.0000, 4.4465, 3.5534, 3.4466,
              0.1551, 0.2249, 1.0000, 1.0000, 2.4462,
              0.1526, 0.2814, 1.0000, 1.0000, 0.4088,
              0.2814, 0.2901, 0.4088, 2.4464, 1.0000)]
mn_jhf = [6, 0.2, 0.24, 0.1667, 0.125]
person.append(construct(mn_jhf, person_name=jhf))

yzh = [_5darr(1.0000, 3.0000, 0.1667, 0.3333, 0.1429,
              0.3333, 1.0000, 0.1667, 0.2000, 0.1429,
              6.0000, 6.0000, 1.0000, 5.0000, 0.5000,
              3.0000, 5.0000, 0.2000, 1.0000, 0.1429,
              7.0000, 7.0000, 2.0000, 7.0000, 1.0000)]
mn_yzh = [0.1429, 0.125, 0.1429, 0.1667, 0.1667]
person.append(construct(mn_yzh, person_name=yzh))

fxl = [_5darr(1.0000, 3.0000, 7.0000, 0.1667, 0.1667,
              0.3333, 1.0000, 3.0000, 0.1429, 0.1667,
              0.1429, 0.3333, 1.0000, 0.1429, 0.1429,
              6.0000, 7.0000, 7.0000, 1.0000, 1.0000,
              6.0000, 6.0000, 7.0000, 1.0000, 1.0000)]
mn_fxl = [8, 7, 0.125, 0.125, 6]
person.append(construct(mn_fxl, person_name=fxl))

'''
jqq = [_5darr(1.0000, 0.3324, 3.0364, 3.0721, 2.9383,
              3.0085, 1.0000, 6.9768, 6.9682, 4.9732,
              0.3293, 0.1433, 1.0000, 1.0000, 2.9111,
              0.3255, 0.1435, 1.0000, 1.0000, 0.3436,
              0.3403, 0.2011, 0.3435, 2.9104, 1.0000)]
mn_jqq = [3, 0.3333, 0.2, 0.2, 0.3333]
person.append(construct(mn_jqq, person_name=jqq))
'''

wyn = [_5darr(1.0000, 3.0000, 5.0000, 7.0000, 6.0000,
              0.3333, 1.0000, 0.3333, 3.0000, 3.0000,
              0.2000, 3.0000, 1.0000, 6.0000, 4.0000,
              0.1429, 0.3333, 0.1667, 1.0000, 2.0000,
              0.1667, 0.3333, 0.2500, 0.5000, 1.0000)]
mn_wyn = [7, 0.1667, 0.1667, 0.2500, 0.1429]
person.append(construct(mn_wyn, person_name=wyn))

zay = [_5darr(1.0000, 4.0000, 6.0000, 5.0000, 0.3333,
              0.2500, 1.0000, 2.0000, 5.0000, 0.1667,
              0.1667, 0.5000, 1.0000, 3.0000, 0.2000,
              0.2000, 0.2000, 0.3333, 1.0000, 0.1667,
              3.0000, 6.0000, 5.0000, 6.0000, 1.0000)]
mn_zay = [6, 0.1667, 0.2, 0.1667, 5]
person.append(construct(mn_zay, person_name=zay))

cx = [_5darr(1.0000, 3.0000, 8.0000, 7.0000, 5.0000,
             0.3333, 1.0000, 8.0000, 6.0000, 4.0000,
             0.1250, 0.1250, 1.0000, 0.3333, 0.1429,
             0.1429, 0.1667, 3.0000, 1.0000, 0.2500,
             0.2000, 0.2500, 7.0000, 4.0000, 1.0000)]
mn_cx = [5, 0.25, 0.1667, 0.1667, 6]
person.append(construct(mn_cx, person_name=cx))

zzb = [_5darr(1.0000, 3.0000, 5.0000, 6.0000, 6.0000,
              0.3333, 1.0000, 3.0000, 5.0000, 5.0000,
              0.2000, 0.3333, 1.0000, 6.0000, 4.0000,
              0.1667, 0.2000, 0.1667, 1.0000, 0.5000,
              0.1667, 0.2000, 0.2500, 2.0000, 1.0000)]
mn_zzb = [4, 4, 0.25, 0.25, 4]
person.append(construct(mn_zzb, person_name=zzb))


ll = [_5darr(1.0000, 2.0000, 7.0000, 4.0000, 6.0000,
             0.5000, 1.0000, 7.0000, 6.0000, 5.0000,
             0.1429, 0.1429, 1.0000, 2.0000, 3.0000,
             0.2500, 0.1667, 0.5000, 1.0000, 2.0000,
             0.1667, 0.2000, 0.3333, 0.5000, 1.0000)]
mn_ll = [3, 0.1667, 0.25, 0.2, 0.1667]
person.append(construct(mn_ll, person_name=ll))


zb = [_5darr(1.0000, 3.0000, 6.0000, 4.0000, 6.0000,
             0.3333, 1.0000, 5.0000, 6.0000, 4.0000,
             0.1667, 0.2000, 1.0000, 1.0000, 3.0000,
             0.2500, 0.1667, 1.0000, 1.0000, 3.0000,
             0.1667, 0.2500, 0.3333, 0.3333, 1.0000)]
mn_zb = [7.0000, 7.0000, 0.1250, 0.1429, 0.1250]
person.append(construct(mn_zb, person_name=zb))

lxr = [_5darr(1.0000, 0.5000, 7.0000, 3.0000, 1.0000,
              2.0000, 1.0000, 6.0000, 6.0000, 1.0000,
              0.1429, 0.1667, 1.0000, 0.2000, 0.2500,
              0.3333, 0.1667, 5.0000, 1.0000, 1.0000,
              1.0000, 1.0000, 4.0000, 1.0000, 1.0000)]
mn_lxr = [0.1429, 0.3333, 0.2, 0.125, 0.1429]
person.append(construct(mn_lxr, person_name=lxr))

ljz = [_5darr(1.0000, 6.0000, 4.0000, 8.0000, 7.0000,
              0.1667, 1.0000, 0.3333, 2.0000, 5.0000,
              0.2500, 3.0000, 1.0000, 4.0000, 4.0000,
              0.1250, 0.5000, 0.2500, 1.0000, 3.0000,
              0.1429, 0.2000, 0.2500, 0.3333, 1.0000)]
mn_ljz = [7, 5, 0.125, 1, 6]
person.append(construct(mn_ljz, person_name=ljz))

ljy = [_5darr(1.0000, 3.0000, 5.0000, 5.0000, 0.3333,
              0.3333, 1.0000, 3.0000, 5.0000, 0.2000,
              0.2000, 0.3333, 1.0000, 3.0000, 0.2000,
              0.2000, 0.2000, 0.3333, 1.0000, 0.2000,
              3.0000, 5.0000, 5.0000, 5.0000, 1.0000)]
mn_ljy = [0.1111, 7.0000, 8.0000, 0.1429, 0.1111]
person.append(construct(mn_ljy, person_name=ljy))


np.set_printoptions(suppress=True, precision=4)
print("-----微众银行数字普惠金融识别------\n")
for i in range(6):
    print(f"This is No.{i + 1} Matrix")
    calculate = []
    for j in range(len(person)):
        calculate.append(person[j][i])
    print(avg(calculate))
    print("----------------\n")
