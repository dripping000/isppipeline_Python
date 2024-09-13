import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import traceback

import json


AWB_STAT_INT = (4096.0)
stats_size_width = 64
stats_size_height = 48


def format_bit(x, num=6):
    return round(x, num)


fig, ax = plt.subplots()
po_annotation1 = []
def plot_annotations(plt, point_x, point_y, text, po_annotation):
    point, = plt.plot(point_x, point_y, 'o', c='darkgreen', markersize=2)
    # 标注偏移量
    offset1 = 30
    offset2 = 30
    # 标注框
    bbox1 = dict(boxstyle="round", fc="lightgreen", alpha=0.6)
    # 标注箭头
    arrowprops1 = dict(arrowstyle="->", connectionstyle="arc3,rad=0.")
    # 标注
    annotation = plt.annotate(text, xy=(point_x, point_y), xytext=(offset1, offset2), textcoords="offset points",
                            bbox=bbox1, arrowprops=arrowprops1, size=15)
    # 默认鼠标未指向时不显示标注信息
    annotation.set_visible(False)
    po_annotation.append([point, annotation])


# 定义鼠标响应函数
def on_move(event):
    visibility_changed = False
    for point, annotation in po_annotation1:
        should_be_visible = (point.contains(event)[0] == True)

        if should_be_visible != annotation.get_visible():
            visibility_changed = True
            annotation.set_visible(should_be_visible)

    if visibility_changed:
        plt.draw()


def plot_list_wbgain(plt, list, quantization=1.0, color='y'):
    # print("\n")
    for i in range(len(list)):
        print(i, list[i])
        print(i, 1/(list[i][0]/quantization), 1/(list[i][1]/quantization))

        if i <= (len(list) - 2):
            plt.plot([1/(list[i][0]/quantization), 1/(list[i+1][0]/quantization)], [1/(list[i][1]/quantization), 1/(list[i+1][1]/quantization)], color=color, linestyle='-')
        if i == (len(list) - 1):
            plt.plot([1/(list[i][0]/quantization), 1/(list[0][0]/quantization)], [1/(list[i][1]/quantization), 1/(list[0][1]/quantization)], color=color, linestyle='-')


def plot_list_1_wbgain(plt, list, quantization=1.0, color='y'):
    # print("\n")
    for i in range(len(list)):
        print(i, list[i])
        print(i, (list[i][0]/quantization), (list[i][1]/quantization))

        if i <= (len(list) - 2):
            plt.plot([(list[i][0]/quantization), (list[i+1][0]/quantization)], [(list[i][1]/quantization), (list[i+1][1]/quantization)], color=color, linestyle='-')
        if i == (len(list) - 1):
            plt.plot([(list[i][0]/quantization), (list[0][0]/quantization)], [(list[i][1]/quantization), (list[0][1]/quantization)], color=color, linestyle='-')


# 判断是否与水平射线相交
def is_intersect(p1, p2, point):
    if min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
        if (p2[1] - p1[1]) == 0:
            if min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]):
                return True
        else:
            x = int(p1[0] + (point[1] - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0]))
            if point[0] < x:
                return True
    return False

def point_in_polygon(point, polygon):
    # 统计相交点的个数
    count = 0
    # 遍历多边形的所有边
    for i in range(len(polygon)):
        # 边的两个顶点
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        # 判断边是否与水平射线相交
        if is_intersect(p1, p2, point):
            count += 1
    # 判断点的位置
    if count % 2 == 1:
        return True  # 在多边形内部
    else:
        return False  # 在多边形外部


def trigger_weight(x, trigger_x, trigger_y):
    for i in range(len(trigger_x)):
        if x >= trigger_x[i] and x <= trigger_x[i+1]:
            weight = (trigger_y[i+1] - trigger_y[i]) / (trigger_x[i+1] - trigger_x[i]) * (x - trigger_x[i]) + trigger_y[i]
            break

    return weight


if __name__ == '__main__':
    with open('./Resource/isp_config_cannon_.json', 'r') as f:
        data = json.load(f)

    # 打印JSON数据
    # print(data)

    r_stat = data['wb_gain']['R_awb_stats']
    g_stat = data['wb_gain']['G_awb_stats']
    b_stat = data['wb_gain']['B_awb_stats']


    plt.style.use('ggplot')

    stat_index = 0  # Debug 多帧数据时，用于选择第几帧数据
    stat_num = 1  # Debug stat_index开始，连续显示stat_num帧数据

    Dark_Stats_Threshold = 0  # Debug 77
    Sat_Stats_Threshold = 65535  # Debug 13107


    SA_1_weight_global = 1.0

    polygon_SA1 = [(0.33, 0.80), (0.42, 0.80), (0.42, 0.67), (0.33, 0.67)]
    polygon_SA1 = [(int(point[0]*AWB_STAT_INT), int(point[1]*AWB_STAT_INT)) for point in polygon_SA1]

    SA_1_Level1_2ndTri_x = [0,0.04,     0.1,0.3,    0.4,0.6,    0.7,1]
    SA_1_Level1_2ndTri_y = [0,0,        0.8,0.8,    1,1,        1,1]


    for i in range(stat_num):
        po_annotation1 = []

        plt.clf()
        SA_1_rgain_sum = 0
        SA_1_bgain_sum = 0
        SA_1_num = 0
        SA_1_rgain = 0
        SA_1_bgain = 0

        golden_num = 0

        wb_rgain = 0
        wb_bgain = 0


        stat_index_ = stat_index + i

        rgain = []
        bgain = []

        Dark_Stats_Count = 0
        Sat_Stats_Count = 0
        g_min = 65535
        g_max = 0

        for index, r in enumerate(r_stat):
            # print(r_stat.index(r))
            # print("------", index)

            r_ = int(r_stat[index])
            g_ = int(g_stat[index])
            b_ = int(b_stat[index])
            # print(r, g_, b_)

            if g_ == 0.0:
                print(index, r_, g_, b_)
                g_ = 1

            rgain_min = int(0 * AWB_STAT_INT)
            rgain_max = int(2 * AWB_STAT_INT)
            bgain_min = int(0 * AWB_STAT_INT)
            bgain_max = int(2 * AWB_STAT_INT)
            rgain_ = np.clip(int(r_/g_*AWB_STAT_INT), rgain_min, rgain_max)
            bgain_ = np.clip(int(b_/g_*AWB_STAT_INT), bgain_min, bgain_max)

            V = math.floor(index / 64)
            H = index - V*64

            if g_ < g_min:
                g_min = g_
            if g_ > g_max:
                if g_ > 65535:
                    print(g_, V, H)
                    pass
                else:
                    g_max = g_

            # if (index%2 == 0):  # (stats_size_width/2)*stats_size_height
            if (1):  # stats_size_width*stats_size_height
                if r_ < Dark_Stats_Threshold or g_ < Dark_Stats_Threshold or b_ < Dark_Stats_Threshold:
                    Dark_Stats_Count += 1
                    print('Dark_Stats_Count', g_, V, H)

                elif r_ > Sat_Stats_Threshold or g_ > Sat_Stats_Threshold or b_ > Sat_Stats_Threshold:
                    Sat_Stats_Count += 1
                    print('Sat_Stats_Count', g_, V, H)
                
                else:
                    plot_annotations(plt, rgain_, bgain_, str(V)+","+str(H)+": "+str(format_bit(rgain_))+","+str(format_bit(bgain_))+","+str(format_bit(r_))+","+str(format_bit(g_))+","+str(format_bit(b_)), po_annotation1)

                    rgain.append(rgain_)
                    bgain.append(bgain_)

                    point = (rgain_, bgain_)

                    if (point_in_polygon(point, polygon_SA1) == True):
                        SA_1_rgain_sum += rgain_
                        SA_1_bgain_sum += bgain_
                        SA_1_num += 1

                        # print(rgain_, bgain_, V, H)
                        # data = str(rgain_) + " " + str(bgain_) + " " + str(V) + " " + str(H) + "\n"
                        # with open("./tmp.txt", 'a') as file:
                        #     file.write(data)

                    # data = str(rgain_) + " " + str(bgain_) + " " + str(V) + " " + str(H) + "\n"
                    # with open("./tmp.txt", 'a') as file:
                    #     file.write(data)

        plt.scatter(rgain, bgain, c='m', s=2)
        plt.title(str(stat_index_)+": ", fontdict={'fontsize': 10})


        if SA_1_num == 0.0:
            SA_1_num = 1.0
        
        SA_1_rgain = int(SA_1_rgain_sum / SA_1_num)
        SA_1_bgain = int(SA_1_bgain_sum / SA_1_num)

        SA_num_sum = stats_size_width*stats_size_height

        SA_1_Level1 = SA_1_num / SA_num_sum
        SA_1_Level1_CONFIDENCE = trigger_weight(SA_1_Level1, SA_1_Level1_2ndTri_x, SA_1_Level1_2ndTri_y)

        SA_1_weight = SA_1_Level1_CONFIDENCE
        SA_1_weight = 1.0

        wb_rgain = SA_1_rgain * SA_1_weight
        wb_bgain = SA_1_bgain * SA_1_weight

        plt.plot(SA_1_rgain, SA_1_bgain, 'o', C='y', markersize=5)


    r_g = [0.253272, 0.407010, 0.442099, 0.527150, 0.515729, 0.586529, 0.781099, 1.021902, 1.401907]
    b_g = [0.939229, 0.773839, 0.728039, 0.578373, 0.447339, 0.448376, 0.363937, 0.297893, 0.187933]
    r_g = [int(x*AWB_STAT_INT) for x in r_g]
    b_g = [int(x*AWB_STAT_INT) for x in b_g]

    plt.scatter(r_g, b_g, c='c', s=10, label='AWB')
    plt.plot(r_g, b_g, color='c', linestyle='-')

    polygon_SA = polygon_SA1
    plot_list_1_wbgain(plt, polygon_SA, quantization=1.0, color='c')


    # 鼠标移动事件
    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

    # 显示图形
    plt.show()

    pass
