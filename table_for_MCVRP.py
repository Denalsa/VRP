##分割路径保存到excel文件中
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy
import random
import numpy as np
import more_itertools as mi

####序_读取数据
def dmnd_dt(filepath):                                              #按照需求点大小输出两份表格,创建初始化的路径

    df = pd.read_excel(filepath)
    df=df.set_index('id')                                           #修改索引
    df_20 = df.loc[df['demand'] <= 20,:]
    df_80 = df.loc[df['demand'] > 20,:]

    nodes_seq_s=copy.deepcopy(df_20.index.to_list())
    nodes_seq_b=copy.deepcopy(df_80.index.to_list())

    random.shuffle(nodes_seq_s)
    random.shuffle(nodes_seq_b)    

    return df,nodes_seq_s,nodes_seq_b
df,nodes_seq_s,nodes_seq_b=dmnd_dt(r'D:\VRP\test_GA.xlsx')
##################

####引入计算fit函数
def calDsts(x_route): 

    o_x=[]
    o_y=[]
    d_x=[]
    d_y=[]
    distance=0
    for i in range(len(x_route)):
        o_x.append(df.loc[x_route[i],'x_coord'])
        o_y.append(df.loc[x_route[i],'y_coord'])
        d_x.append(df.loc[x_route[i],'d_x'])
        d_y.append(df.loc[x_route[i],'d_y'])

    while i < len(x_route)-1:                                           #最后一个i已经比len(route_b)小1，所以要求i比len(route_b)-1还小1       
        d1=math.sqrt((o_x[i]-d_x[i])**2 + (o_y[i]-d_y[i])**2)           #配送距离

        d2=math.sqrt((d_x[i]-o_x[i+1])**2+(d_y[i]-o_y[i+1])**2)         #订单距离
        d=d1+d2

    if i == len(o_x)-1:
        d1=math.sqrt((o_x[i]-d_x[i])**2 + (o_y[i]-d_y[i])**2)           #配送距离
        d=d1

    distance +=d
    return distance       
##########


####处理最优染色体，分成大小车两种,输出各种fit
def chr_fit():              
    chrome=[3, 20, 10, 22, 23, 16, 19, 24, 31, 26, 30, 25, 15, 13, 5, 11, 9, 0, 0, 0, 0, 0, 0, 0, 18, 7, 21, 29, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 27, 0, 0, 6, 2, 8, 17, 0, 0, 0, 4, 12, 0, 0, 1]

    chrome_to_routes =[i for i in mi.split_at(chrome, lambda x : x == 0) if i]
   
    return chrome_to_routes                                          #return  fit_N_pop
#############

####画图辅助函数
def plt_arrow(x_begin,y_begin,x_end,y_end):
    plt.arrow(x_begin, y_begin, x_end - x_begin, y_end - y_begin,
             length_includes_head=True,                                 # 增加的长度包含箭头部分
             head_width = 1, head_length =1.618) 
################

####定义画图
def route(routes):         #这里的N_begin是0,3,6   N_end是3，6，9
    plt.figure(figsize=(7,7))
    d =  1 
    for i in range(int(len(routes))):   
        plt.subplot(3,3,d)
        plt.title((f"配送车辆：{(d)}"))
        plt.xlabel('经度')
        plt.ylabel('纬度')     
        for j in range(int(len(routes[i]))):
            print(len(routes[i]))

            if j == 0:                                                                                      #画图约束                                                                 
                x_begin,y_begin =0,0                                                                        #连接第一单起点
                x_end, y_end =df.loc[routes[i][j],'x_coord'],df.loc[routes[i][j],'y_coord']                 ####
                plt_arrow(x_begin, y_begin, x_end, y_end)

                x_begin,y_begin=df.loc[routes[i][j],'x_coord'],df.loc[routes[i][j],'y_coord']             #配送第一单路线
                x_end, y_end =df.loc[routes[i][j],'d_x'],df.loc[routes[i][j],'d_y']                       ############
                plt_arrow(x_begin, y_begin, x_end, y_end)
                if j==int(len(routes[i])-1):
                    pass
                else:
                    x_begin,y_begin =df.loc[routes[i][j],'d_x'],df.loc[routes[i][j],'d_y']                      #连接下一单起点
                    x_end, y_end =df.loc[routes[i][j+1],'x_coord'],df.loc[routes[i][j+1],'y_coord']
                    plt_arrow(x_begin, y_begin, x_end, y_end)

            elif 0<j<int(len(routes[i])-1):                                                                                               #画图约束

                x_begin,y_begin=df.loc[routes[i][j],'x_coord'],df.loc[routes[i][j],'y_coord']             #配送路线
                x_end, y_end =df.loc[routes[i][j],'d_x'],df.loc[routes[i][j],'d_y']                        ############
                plt_arrow(x_begin, y_begin, x_end, y_end)
                if j==int(len(routes[i])-1):
                    pass
                else:
                    x_begin,y_begin=df.loc[routes[i][j],'d_x'],df.loc[routes[i][j],'d_y']                  #连接下一单起点
                    x_end, y_end =df.loc[routes[i][j+1],'x_coord'],df.loc[routes[i][j+1],'y_coord']
                    plt_arrow(x_begin, y_begin, x_end, y_end)
        d += 1
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
    plt.show()
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(train_data['outtemp'][0:101])
        # plt.subplot(2,1,2)
        # plt.plot(train_data['temperature'][0:101])
    
        
    exit()
##############


####主程序画图
def main_progress():
    chrome_to_routes=chr_fit()
    #画车图
    route(chrome_to_routes)

    #画散点图

main_progress()