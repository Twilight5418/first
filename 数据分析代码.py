#数据处理包导入
import numpy as np
import pandas as pd

#画图包导入
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
sns.set()

#日期处理包导入
import calendar
from datetime import datetime 

#读取数据,记得改读取路径
BikeData = pd.read_csv(r"D:\homework\bike-sharing-demand\train.csv")

#提取“date”
BikeData["date"] = BikeData.datetime.apply(lambda x: x.split()[0])

#提取"hour"
BikeData["hour"]=BikeData.datetime.apply(lambda x: x.split()[1].split(":")[0])
BikeData['ihour']=BikeData['hour'].astype(int)

dateString = BikeData.datetime[1].split()[0]

#提取"weekday"
BikeData["weekday"] = BikeData.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])


#提取"month"
BikeData["month"] = BikeData.date.apply(lambda dateString:  calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month] )

#季节映射处理
BikeData["season_label"] = BikeData.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

#天气映射处理
BikeData["weather_label"] = BikeData.weather.map({1:"sunny",2:"cloudy",3:"rainly",4:"bad-day"})

#是否是节假日映射处理
BikeData["holiday_map"] = BikeData["holiday"].map({0:"non-holiday",1:"hoiday"})

#可视化查询缺失值
#缺失值可视化.png
msno.matrix(BikeData,figsize=(12,5))

plt.show()
#先数据分析，为之后准确展示租车数据与什么因素相关打好基础

#利用数据可视化分析共享单车的租用情况与哪些因素有关
#租车人数与温度，体感温度，租车类型，湿度，风速的相关性可视化.png
correlation = BikeData[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(correlation)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(correlation, mask=mask,vmax=.8, square=True,annot=True)

plt.show()


# 租车人数，按不同的因素划分的分布情况
# 设置绘图格式和画布大小
#租车人数与季节，时间段，是否工作日的关系的可视化图表.png
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)

# 添加第一个子图， 租车人数分布的箱线图
sns.boxplot(data=BikeData,y="count",orient="v",ax=axes[0][0])

#添加第二个子图，租车人数季节分布的箱线图
sns.boxplot(data=BikeData,y="count",x="season",orient="v",ax=axes[0][1])

#添加第三个子图，租车人数时间分布的箱线图
sns.boxplot(data=BikeData,y="count",x="hour",orient="v",ax=axes[1][0])

#添加第四个子图，租车人数工作日分布的箱线图
sns.boxplot(data=BikeData,y="count",x="workingday",orient="v",ax=axes[1][1])

# 设置第一个子图坐标轴和标题
axes[0][0].set(ylabel='Count',title="Box Plot On Count")

# 设置第二个子图坐标轴和标题
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

# 设置第三个子图坐标轴和标题
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

# 设置第四个子图坐标轴和标题
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

plt.show()


x=BikeData[['temp']]
y=BikeData[['count']]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.scatter(x,y)
plt.xlabel('temp')
plt.ylabel('count')
plt.show()

x=BikeData[['atemp']]
y=BikeData[['count']]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.scatter(x,y)
plt.xlabel('atemp')
plt.ylabel('count')
plt.show()

x=BikeData[['humidity']]
y=BikeData[['count']]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.scatter(x,y)
plt.xlabel('humidity')
plt.ylabel('count')
plt.show()

#分析什么样的温度和湿度情况下租车的人数最多
#由于温度，体感温度和湿度数据变化较多，范围较广，因此将温度和湿度离散化，便于分析
BikeData["humidity_band"] = pd.cut(BikeData['humidity'],5)
BikeData["temp_band"] = pd.cut(BikeData["temp"],5)
BikeData["atemp_band"] = pd.cut(BikeData["atemp"],5)

#假期字段映射处理
BikeData["holiday_map"] = BikeData["holiday"].map({0:"non-holiday",1:"hoiday"})

#温度和湿度与租车人数的可视化.png
sns.FacetGrid(data=BikeData,row="humidity_band",aspect=2).\
map(sns.barplot,'temp_band','count','holiday_map',palette='deep',errorbar=None).\
add_legend()

plt.show()

#分析什么样的体感温度和湿度情况下租车的人数最多

#体感温度和湿度与租车人数的可视化.png
sns.FacetGrid(data=BikeData,row="humidity_band",aspect=2).\
map(sns.barplot,'atemp_band','count','holiday_map',palette='deep',errorbar=None).\
add_legend()

plt.show()



#分析不同季节下每小时平均租车人数如何变化
#不同季节下不同时间段与租车人数的可视化.png
sns.FacetGrid(data=BikeData,aspect=1.5).\
map(sns.pointplot,'hour','count','season_label',palette="deep",errorbar=None).\
add_legend()

plt.show()

#分析不同月份下每小时平均租车人数如何变化
#不同月份下不同时间段与租车人数的可视化.png
sns.FacetGrid(data=BikeData,aspect=1.5).\
map(sns.pointplot,'hour','count','month',palette="deep",errorbar=None).\
add_legend()

plt.show()

#分析不同天气情况下，每个月的平均租车人数如何变化
#不同月份下不同天气与租车人数的可视化.png
sns.FacetGrid(data=BikeData,aspect=1.5).\
map(sns.pointplot,'month','count','weather_label',palette="deep",errorbar=None).\
add_legend()

plt.show()

#分析按星期数划分，每小时的平均租车人数如何变化
#同一星期内租车人数随时间的大致变化的可视化.png
sns.FacetGrid(data=BikeData,aspect=1.5).\
map(sns.pointplot,'hour','count','weekday',palette="deep",errorbar=None).\
add_legend()

plt.show()


