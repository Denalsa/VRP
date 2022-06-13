from itertools import count
import pandas as pd
import random
import copy
import math
import numpy as np
import more_itertools as mi
import time
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

####一、1-1 创建初始种群
def Create_Init_pop(pop_size):                                                     #给定种群大小，染色体长度
    # dmnd_num=len(nodes_seq_s)+len(nodes_seq_b)                      #需求点数量

    Init_pop=[]
    for j in range(pop_size):                                       #种群的长度为变量
        chromsome=[0]
        for i in nodes_seq_s:
            chromsome.append(i)
            chromsome.append(0)
        # random.shuffle(chromsome)
        # Init_pop.append(chromsome)
        for k in nodes_seq_b:
            chromsome.append(k)
            chromsome.append(0)
        random.shuffle(chromsome)
        Init_pop.append(chromsome)
                                              
    return Init_pop
##################

####      1-2  (辅助函数) 计算任一路径距离。[1,2,3],每单的配送距离+配送点与下一点的距离。1.找出配送点2.计算距离3.返回距离
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
#################

####二、计算适应度：（2-1计算染色体适应度，运用锦标赛）改变种群为分裂了的路径列表，计算fit，合并每条染色体fit，形成列向量。
def chr_fit(Init_pop):              

    Init_pop_routes=[]
    pop_fit_list=[]

    for chrome in Init_pop:                                                     #对所有染色体
        chrome_to_routes =[i for i in mi.split_at(chrome, lambda x : x == 0) if i]
        Init_pop_routes.append(chrome_to_routes)                                 #顺便添加这条染色体路径到Init_pop_routes
        chrome_fit=[]
        route_fit=[]

        b_routes=[]
        s_routes=[]
        for route in chrome_to_routes:                                           #判断该路径的车型，并分类到相应车型的列表里
            for i in range(len(route)):
                if route[i] in nodes_seq_b:                                      
                    b_routes.append(route)   
        ###嵌套列表去重                                       
        b_routes=[list(t) for t in set(tuple(_) for _ in b_routes)]                #去重
        b_routes.sort(key=chrome_to_routes.index)
        s_routes=[item for item in chrome_to_routes if item not in b_routes]       #取补集
        s_routes.sort(key=chrome_to_routes.index)

        #计算大车fit
        for i in range(len(b_routes)):                                              #这里是否有逻辑错误
            b_routes_fit=1.3*calDsts(b_routes[i])
            route_fit.append(b_routes_fit)
        #计算小车fit
        for i in range(len(s_routes)):
            s_routes_fit=0.9*calDsts(s_routes[i])
            route_fit.append(s_routes_fit)
        routes_fit=sum(route_fit)
        chrome_fit.append(routes_fit)
        pop_fit_list.append(chrome_fit[0])

    ####选优                                       
    fit_N_pop={'chrome':Init_pop,'cost':pop_fit_list}                    #坑，这个Init_pop须传迭代后的值，这一步创建含有fit值的dataframes

    fit_N_pop=pd.DataFrame(data=fit_N_pop)

    fit_N_pop=fit_N_pop.assign(fit=lambda x: (sum(x.cost)-x.cost)/sum(x.cost))    #求fit

    fit_N_pop.sort_values(by='fit',inplace=True,ascending=False)                        #按fit排序，从小到大
    fit_N_pop.reset_index(inplace=True)                                 #重制索引

    return fit_N_pop                                           #return  fit_N_pop
#########################

####  2-3（主要）交叉操作的函数,辅助作用
def pmx(parent1, parent2, length, r_a_b=None, ):                            #length为矩阵的行数
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        if r_a_b is None:
            a, b = np.random.choice(length, 2, replace=False)
            min_a_b, max_a_b = min([a, b]), max([a, b])
            r_a_b = range(min_a_b, max_a_b)                                 #range 位置
        r_left = np.delete(range(length), r_a_b)
        left_1, left_2 = child1[r_left], child2[r_left]
        middle_1, middle_2 = child1[r_a_b], child2[r_a_b]
        child1[r_a_b], child2[r_a_b] = middle_2, middle_1
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = np.argwhere(middle_1 == j)[0, 0]
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = np.argwhere(middle_1 == value)[0, 0]
                        value = middle_2[index]
                    else:
                        break
                mapping[0].append(i)
                mapping[1].append(value)
            elif i in middle_2:
                pass
            else:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[np.argwhere(left_1 == i)[0, 0]] = j
            elif i in left_2:
                left_2[np.argwhere(left_2 == i)[0, 0]] = j
            if j in left_1:
                left_1[np.argwhere(left_1 == j)[0, 0]] = i
            elif j in left_2:
                left_2[np.argwhere(left_2 == j)[0, 0]] = i
        child1[r_left], child2[r_left] = left_1, left_2
        return child1, child2
#############################

####     2-2   1.计算route适应度，选择优秀路径 2.交叉前面1/2基因
def rou_fit(fit_N_pop):                                                     #坑 fit为表格

    nick_fit_N_pop=[]                                                       ####处理一下dataframe结构的pop
    for i in range(len(fit_N_pop)):
        nick_chrome=fit_N_pop.loc[i,'chrome']
        nick_fit_N_pop.append(nick_chrome)
    
    Init_pop_routes=[]
    best_routes=[]
    for chrome in nick_fit_N_pop:                                            #对所有染色体
        chrome_to_routes =[i for i in mi.split_at(chrome, lambda x : x == 0) if i]
        Init_pop_routes.append(chrome_to_routes)                             #顺便添加这条染色体路径到Init_pop_routes
        
        b_routes=[]
        s_routes=[]
        
        for route in chrome_to_routes:                                           #路径分类#判断该路径的车型，并分类到相应车型的列表里
            for i in range(len(route)):
                if route[i] in nodes_seq_b:                                      
                    b_routes.append(route)   
                                              
        b_routes=[list(t) for t in set(tuple(_) for _ in b_routes)]                ###嵌套列表去重 #去重
        b_routes.sort(key=chrome_to_routes.index)
        s_routes=[item for item in chrome_to_routes if item not in b_routes]       #取补集
        s_routes.sort(key=chrome_to_routes.index)
              
        Bche_df=pd.DataFrame(data=None,columns=['chrome_route','cost'])            #铺垫求解chrome每条路径fit  
        Sche_df=pd.DataFrame(data=None,columns=['chrome_route','cost'])
        Bche_df.loc[:,'chrome_route']=b_routes
        Sche_df.loc[:,'chrome_route']=s_routes
        
        for i in range(len(b_routes)):                                              #计算大车花费,
            b_routes_fit=0.5*calDsts(b_routes[i])
            Bche_df.loc[i,'cost']=b_routes_fit
        
        for i in range(len(s_routes)):                                              #计算小车花费
            s_routes_fit=0.5*calDsts(s_routes[i])
            Sche_df.loc[i,'cost']=s_routes_fit

        chrome_route_fit=pd.concat([Bche_df,Sche_df])
        chrome_route_fit=chrome_route_fit.assign(fit=lambda x: (sum(x.cost)-x.cost)/sum(x.cost))    #求fit
        chrome_route_fit.sort_values(by='fit',inplace=True)                         #按fit从小到大排序
        chrome_route_fit.reset_index(inplace=True)                                  #整合完的一张新表，route_fit
     
        last_row=len(chrome_route_fit)-1                                            ####取最后一行
        best_route=chrome_route_fit.loc[last_row,'chrome_route']

        best_routes.append(best_route)                                              #只需要最大fit时，这条染色体的route

    new_fit_N_pop = fit_N_pop.assign(best_rou = best_routes)      
    cross_time=math.floor(len(new_fit_N_pop)/4)                                    ##左：交叉次数；右：奇偶都可以，向下取整

    new_fit_N_pop_list=new_fit_N_pop.loc[:,'chrome'].tolist()

    #这种交叉避免了if判断
    for i in range(cross_time):                                                    ###交叉到原表长度（test1交换基因中不能有0，r_a_b只能是下标）
        j=2*i+1
        k=2*i+2
        parent1=np.array(new_fit_N_pop.loc[j,'chrome'])
        parent2=np.array(new_fit_N_pop.loc[k,'chrome'])
        
        length=parent1.shape[0]
        pass_r_a_b=new_fit_N_pop.loc[i,'best_rou']                                                                            #选取最优路径
        child1,child2=pmx(parent1, parent2, length, r_a_b=[i for i, x in enumerate(parent1) if x in pass_r_a_b])              ###调用函数
        
        child1=child1.tolist()
        child2=child2.tolist()
        new_fit_N_pop_list.append(child1)
        new_fit_N_pop_list.append(child2)
    del new_fit_N_pop_list[2*cross_time+1:4*cross_time+1]                          #删除没有参与交叉的基因，个数为cross_time   

    cla_fit_df=chr_fit(new_fit_N_pop_list)

    new_fit_N_pop_list=cla_fit_df.loc[:,'chrome'].to_list()
    return new_fit_N_pop_list                                                      ####new_fit_N_pop_list有
##############################

###变异
def mutate(pm,new_fit_N_pop_list):
    for i in range(math.floor(1/2*len(new_fit_N_pop_list)),len(new_fit_N_pop_list)):        ##变异后面1/4

        if pm<random.random():
            list_1=[i for i, x in enumerate(new_fit_N_pop_list[i]) if x!=0]                       #非零数下标集合
            list_0=[j for j, x in enumerate(new_fit_N_pop_list[i]) if x==0]                       #零的下标集合

            sub_1=random.randrange(len(list_1))
            sub_0=random.randrange(len(list_0))

            new_fit_N_pop_list[i][sub_1],new_fit_N_pop_list[i][sub_0]=new_fit_N_pop_list[i][sub_0],new_fit_N_pop_list[i][sub_1]

    cla_fit_df=chr_fit(new_fit_N_pop_list)

    best_cost=cla_fit_df.loc[0,'cost']
    best_sample=cla_fit_df.loc[0,'chrome'] 

    return cla_fit_df,best_cost,best_sample

        
####主程序
def run(pop_size,pm,epoch):
    start = time.time()
    best_cost_list=[]
    for i in range(epoch):
        
        if i ==0:
            Init_pop=Create_Init_pop(pop_size)
            fit_N_pop=chr_fit(Init_pop)
            new_fit_N_pop_list=rou_fit(fit_N_pop)
            iteral_pop,best_cost,best_sample=mutate(pm,new_fit_N_pop_list) 
            # best_cost_list.append(best_cost) 
            # print((f"历代最优花费：{(best_cost)}"))                                            #打印历代最佳cost

        else:
            new_fit_N_pop_list=rou_fit(iteral_pop)
            iteral_pop,best_cost,best_sample=mutate(pm,new_fit_N_pop_list)
            
            # print((f"历代最优花费：{(best_cost)}"))                                            #打印历代最佳cost
        best_cost_list.append(best_cost) 
    print((f"最优花费：{(best_cost)}s")) 
    print((f"最优染色体：{(best_sample)}"))                                          #打印历代最佳cost
    end = time.time()
    print((f"迭代用时：{(end-start)}")) 
    return epoch,best_cost_list

#####################


import matplotlib.pyplot as plt
def plt_perfomance():
    epoch,best_cost_list=run(50,0.6,100)                                        #主程序
    x = list(range(1,epoch+1))                                                    #数据
    y = best_cost_list                                                          ######
    
    plt.plot(x, y)
    plt.xlabel('迭代次数')
    plt.ylabel('历代花费')
    plt.title('迭代情况')
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'
    plt.show()

plt_perfomance()