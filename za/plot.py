import seaborn as sns
from pandas import DataFrame
import torch
import pandas as pd
import matplotlib.pyplot as plt
'''
y=[96.33961351238096 ,96.52295292333143 ,96.5355387437139 ,96.58980599038596,96.82020940000001,96.91696076701699,97.0891589248043,97.14340613346879,96.86489512375216,96.79500320779872]
x=[i+1 for i in range(10)]
std=[0.0027961003409536674 ,0.004686533513565735 ,0.005340190348674572 ,0.002237189184079053,0.004048590995635874,0.003629155411364523,0.001974249755098094,0.0038963516842780562,0.0030351938104344225,0.003106780381191848]
res=[]

y=[91.98918330755432 ,93.66978558980257 ,94.3812305492154 ,94.68782651402105,94.81530351368344,94.93710981848364,95.06521425483715,95.07703743715978,95.11346898296313,95.06830356267942]
x=[i+1 for i in range(10)]
std=[0.0015591813989865066 ,0.0015245298697664203 ,0.0015495551198472818 ,0.0008251277042423067,0.001690966172846567,0.001484180701086264,0.0015726319787333866,0.0012625776713141613,0.0011613754016032386,0.002329088247687012]
res=[]
res=[]
y=[0.946 ,0.955, 0.957, 0.959 ,0.961 ,0.959, 0.961 ,0.959, 0.961, 0.962]
std=[0.000,0.001,0.001,0.001,0.001,0.001,0.001,0.002,0.001,0.001]
'''
'''
sns.set(rc={'figure.figsize':(11.7,8.27)})
res=[]
#y=[96.33961351238096 ,96.52295292333143 ,96.5355387437139 ,96.58980599038596,96.82020940000001,96.91696076701699,97.0891589248043,97.14340613346879,96.86489512375216,96.79500320779872]
#y=[0.946,0.955,0.958,0.960,0.960,0.961,0.961,0.960,0.961,0.961]
y=[0.9442211819999999 ,0.9532103019999999 ,0.953317936 ,0.9565915760000001 ,0.957352594 ,0.958327264 ,0.958728086 ,0.96030558 ,0.958540378 ,0.9581458279999999 ]

x=[i+1 for i in range(10)]
#x.append(11)
#x.append(12)
#std=[0.0027961003409536674 ,0.004686533513565735 ,0.005340190348674572 ,0.002237189184079053,0.004048590995635874,0.003629155411364523,0.001974249755098094,0.0038963516842780562,0.0030351938104344225,0.003106780381191848]
#std=[0.0008898958503645388,0.0014363749032393846,0.0021291288254363596,0.0011726802116792012,0.001943222759290375,0.0008804896972617153,0.0020108280829807253]
std=[0.0008898958503645388,0.0014363749032393846,0.0021291288254363596,0.0011726802116792012,0.001943222759290375,0.0008804896972617153,0.0020108280829807253,0.0004897315647168229,0.0006974579214375527,0.0014091597877515527]
for i in range(10):
    res.append([x[i],100*y[i],100*std[i]])
data = DataFrame(res, columns=['Number of Factors', 'AUC(%)','std'])
#ax = sns.lineplot(x="rho", y="wait_time_mean", hue="c", style="service_type", data=df)
#ax.fill_between(df["rho"], y1=df["wait_time_mean"] - df["wait_time_std"], y2=df["wait_time_mean"] + df["wait_time_std"], alpha=.5)
sns.set(style="darkgrid")
sns.set(font_scale = 3.5)
a=sns.scatterplot(x="Number of Factors",
                y="AUC(%)",
                data=data,color="r")
# 在图上绘制线段
a=sns.lineplot(x="Number of Factors",
             y="AUC(%)",
             ci=None,
             data=data,color="r")
a.fill_between(data['Number of Factors'] , y1=data["AUC(%)"]- data["std"], y2=data["AUC(%)"] + data["std"],alpha=.3,color="r")
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8,9,10,15,20])
#a.fill_between(x, y1=[y[i]-std[i] for i in range(10)], y2=[y[i]+std[i] for i in range(10)], alpha=.2,color="r")
#plt.legend(loc='lower right', title='Dataset')
#for tick in a.xaxis.get_major_ticks():
#    tick.label.set_fontsize(20)

#for tick in a.yaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
#a.set_xlabel("X Label",fontsize=20)
#a.set_ylabel("Y Label",fontsize=20)
fig = a.get_figure()
fig.savefig("figure1/nfactor_engb.pdf",bbox_inches='tight' , pad_inches=0.01)
'''


sns.set(font_scale=1.8)
#data=torch.load('feature/chame_out_32_data.pt')
data=torch.load('feature/feature_out_32_5_naive1.pt')
df=pd.DataFrame(data.detach().cpu().numpy())
cor=df.corr()
plt.figure(figsize=(10,8))
corre=sns.heatmap(cor,cmap="YlGnBu")
fig = corre.get_figure()
fig.savefig("figure/feature_corr_5fatcor_naive.pdf",bbox_inches='tight' , pad_inches=0.01)
#fig.savefig("figure/feature_corr_5fatcor_data.pdf",bbox_inches='tight' , pad_inches=0.01)






'''

#de 93.06  93.7
#chameleon 0.960 99.1
#squirrel 0.952 98.2
#engb 0.9506617211671545 96.0

#list1 = [96.2,93.7,99.1,98.2,95.066,93.06,96.0,95.2,95.4,92.9,98.7,97.7,94.6,92.5,98.7,97.6]  
list1 = [96.2,93.7,99.1,98.2,94.6,92.4,96.0,95.2,95.4,92.9,98.7,97.7,94.7,93.1,98.7,97.6]  
list2 = ['DisenLink','DisenLink','DisenLink','DisenLink','DisenLink-{}'.format(chr(945)),'DisenLink-{}'.format(chr(945)),'DisenLink-{}'.format(chr(945)),'DisenLink-{}'.format(chr(945)),'DisenLink-s','DisenLink-s','DisenLink-s','DisenLink-s','DisenLink-r','DisenLink-r','DisenLink-r','DisenLink-r']
list3=['ENGB','DE','Chameleon','Squirrel','ENGB','DE','Chameleon','Squirrel','ENGB','DE','Chameleon','Squirrel','ENGB','DE','Chameleon','Squirrel']
std=[0.1,0.1,0.1,0.1,0.2,0.1,0.3,0.1,0.23,0.2,0.1,0.2,0.1,0.1,0.12,0.19]
res=[]
for i in range(16):
    res.append([list1[i],list2[i],list3[i],std[i]])
data = DataFrame(res, columns=['AUC(%)', 'Aggragation','Datasets','SD'])
sns.set(style="darkgrid")
sns.set(font_scale = 1.3)
a1=sns.barplot(x = 'Datasets',
            y = 'AUC(%)',
            hue = 'Aggragation',
            data = data)

x_coords = [p.get_x() + 0.5*p.get_width() for p in a1.patches]
y_coords = [p.get_height() for p in a1.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=std, fmt="none", c= "k")

#a1.errorbar( x=data['Datasets'], y='AUC(%)',yerr='SD', ls='Aggragation', lw=3, color='black')
a1.set(ylim=(80, 100))
plt.legend(loc=[0.00,0.805], fontsize='xx-small',frameon=False)
#plt.legend(loc='best',fontsize='xx-small',alpha  = 0.3)
#frameon=False
fig = a1.get_figure()
fig.savefig("figure1/ablation.pdf",bbox_inches='tight' , pad_inches=0.01)
year=[92.7,93.7,94.4,94.6,95.4,94.9,95.4,95.3,95.3,95.0,96.0,95.8,95.7,96.1,96.0,96.3,95.8,95.7,95.7,96.3,96.5,97.2,97.4,96.4,96.8,96.9,96.7]
reed98=[95.0,96.0,95.8,95.7,96.1,96.0,96.3,95.8,95.7]
cora=[95.7,96.3,96.5,97.2,97.4,96.4,96.8,96.9,96.7]
'''




'''
#reed=[95.0,96.0,95.8,95.7,96.1,96.0,96.3,95.8,95.7]
#year=[92.7,93.7,94.4,94.6,95.4,94.9,95.4,95.3,95.3,  95.7,96.3,96.5,97.2,97.4,96.4,96.8,96.9,96.7 ]
year=[0.8731357396666667 ,0.9246626679999999, 0.9362664695000001 ,0.9430594895000001,0.9467455494999999,0.9487154094999999,0.949888088,0.9523220350000001,0.9509836905,0.9485984324999999,0.9470117663333333,
0.921 ,0.941, 0.942 ,0.950 ,0.955 ,0.957, 0.959, 0.960, 0.959 ,0.958, 0.957, 
0.941, 0.956 ,0.958, 0.963 ,0.964 ,0.967 ,0.968 ,0.968 ,0.971, 0.968, 0.966] 
std=[0.005882599380180775,0.0038513343103093517,0.0043788569522016495,0.0031109804402237495,0.003288298409287806,0.003809390582747412,0.003058482291613936,0.0030256204815715046,0.0035621421585156823,0.0023592147088912383,0.0029457976119080537,
0.011,0.004,0.003,0.002,0.002,0.001,0.002,0.001,0.000,0.001,0.001,
0.004 ,0.003,0.005,0.004,0.003,0.003,0.002,0.004,0.004,0.002,0.002]
x=[0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
y=[]
for i in range(len(year)):
    year[i]=100*year[i]
for i in range(len(year)):
    std[i]=100*std[i]

#for i in range(11):
#    y.append([x[i],year[i],std[i],'ArXiv-year'])
for i in range(11):
    y.append([x[i],year[11+i],std[11+i],'ENGB'])
#for i in range(11):
#    y.append([x[i],year[22+i],std[22+i],'Cora'])
data = DataFrame(y, columns=[r'$ \beta $', 'AUC(%)','std','Datesets'])
sns.set(style="darkgrid")
sns.set(font_scale = 2)
a=sns.scatterplot(x=r'$ \beta $',
                y='AUC(%)',
                data=data,legend=False,color="r")
# 在图上绘制线段
a=sns.lineplot(x=r'$ \beta $',
                y='AUC(%)',
                data=data,color="r")
a.fill_between(data[r'$ \beta $'][0:11] , y1=data["AUC(%)"][0:11] - data["std"][0:11], y2=data["AUC(%)"][0:11] + data["std"][0:11],color="r",alpha=.3)
#a.fill_between(data[r'$ \beta $'][11:22] , y1=data["AUC(%)"][11:22] - data["std"][11:22] , y2=data["AUC(%)"][11:22]  + data["std"][11:22] ,color="r",alpha=.3)
#a.fill_between(data[r'$ \beta $'][22:33] , y1=data["AUC(%)"][22:33] - data["std"][22:33], y2=data["AUC(%)"][22:33]  + data["std"][22:33] ,color="g",alpha=.3)
#a.set_xticks(data[r'$ \beta $'].values)
#plt.legend(loc=0)
fig = a.get_figure()
fig.savefig("figure1/beta_engb.pdf",bbox_inches='tight' , pad_inches=0.01)
'''