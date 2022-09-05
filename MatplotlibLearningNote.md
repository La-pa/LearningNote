# Matplotlib学习笔记

## 作图基本操作

```python
#import为导包的意思
import matplotlib.pyplot as plt		#matplotlib的pyplot模块，缩写为plt
import numpy as np					#numpy缩写为np（太长了）
import matplotlib as mpl			#matplotlib缩写为mpl
```

### 生成数据

$$
y = x^2 \quad\quad\\
y = 3x+3  
$$



```python
x = np.linspace(-5, 5, 50)		#在-5到5的长度上生成50个点
y1 = x ** 2						
y2 = 3*x + 3
```

### 绘制图像

```python
plt.figure(num=2)				#建立画布，函数内可写参数来区分不同画布
plt.title('title')				#图像的名称
plt.plot(x, y1)					#默认模式，注意x和y的顺序
plt.plot(x, y2, color="r", linewidth=1.0,linestyle="--")			
								#'r'表示红色线条，'--'表示线条样式为虚线
plt.show()						#展示画布
```

### 修改坐标轴

###### 坐标轴的描述

```python
plt.xlabel('x')					#设置坐标轴名称
plt.ylabel('y')

plt.xlim((-2, 4))				#调节坐标轴取值范围
plt.ylim((-5, 15))				#注意要有两个括号
```

###### 修改坐标轴的位置

* 即调整原点在图像的位置

 ```python
# gca = "get current axis" 在坐标轴的具体位置
ax = plt.gca()
# spine是山脊的意思，在这里表示轴
ax.spines['right'].set_color("none")			# none表示消失
ax.spines['top'].set_color("none")
  
ax.xaxis.set_ticks_position("bottom") 			#将底下的轴设置x轴
ax.yaxis.set_ticks_position("left")				#将左边的轴设置y轴
ax.spines['bottom'].set_position(('data',0))	#将x轴移动到y=0的位置
ax.spines['left'].set_position(('data',0))		#将y轴移动到x=0的位置
 ```

###### 修改坐标尺的最小分度

```python
new_ticks = np.linspace(-2, 4, 13)			#在-2~4之间取13个距离相等的点
# print(new_ticks)							#打印数据，防止出现错误
plt.xticks(new_ticks)
plt.yticks([-5, -2.5, 2.5, 10, 15],
           [r'$realy\ bad$', "$bad$","$normal$", "$good$", "$realy\ good$"])
```



### 图例

```python
plt.plot(x, y1, label='up')					#给线条取名
plt.plot(x, y2, color="r", linewidth=1.0,
         linestyle="--", label='down')
plt.legend(loc='lower right')				#显示图例
```

###### loc具体方位

```
best
upper right
upper left
lower left
lower right
right
center left
center right
lower center
upper center
center
```



### 注解

```python
x0 = 3.8
y0 = x0 ** 2
plt.scatter(x0,y0,s=70,color='blue')
#在（x0,y0）坐标标记一个粗70蓝色的点
plt.plot([x0,x0],[y0,0],'k--',linewidth=2.5)
#过点（x0,y0)画一条垂直于x轴的黑色虚线

plt.annotate(r'$x^2=%s$'%y0, 				#引号内为文本，%s可以替代
             xy=(x0,y0),					#指需要标记的点在(x0,y0)
             xycoords='data', 				#是根据坐标上的值来对应
             xytext=(+30,-30), 				#偏移量
             textcoords='offset points',	#相对于(x0,y0)偏移
             fontsize=20,arrowprops=
             dict(arrowstyle='->',connectionstyle='arc3,rad=0.2'))
											#箭头样式

#在（-2,5)的位置写一段文本，字体为25，颜色为红色
plt.text(-2,5,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',fontdict={'size':25,'color':'r'})
```



### 完整代码

```python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:17:40 2022

@author: Jiang
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 50)
y1 = x ** 2
y2 = 3*x + 3

plt.figure(num=3)

plt.xlim((-2, 4))
plt.ylim((-5, 15))

new_ticks = np.linspace(-2, 4, 13)
# print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-5, -2.5, 2.5, 10, 15],
           [r'$realy\ bad$', "$bad$",
            "$normal$", "$good$", "$realy\ good$"])

# gca = "get current axis" 在坐标轴的具体位置
ax = plt.gca()

# spine是山脊的意思，在这里表示坐标轴
# none表示消失
# TODO 问题set_color是自己定义的变量吗

ax.spines['right'].set_color("none")
ax.spines['top'].set_color("none")

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))


plt.plot(x, y1, label='up')
plt.plot(x, y2, color="r", linewidth=1.0,
         linestyle="--", label='down')
plt.legend(loc='lower right')

x0 = 3.8
y0 = x0 ** 2
plt.scatter(x0,y0,s=70,color='blue')
plt.plot([x0,x0],[y0,0],'k--',linewidth=2.5)

plt.annotate(r'$x^2=%s$'%y0, xy=(x0,y0),
             xycoords='data', xytext=(+30,-30),
             textcoords='offset points',
             fontsize=20,arrowprops=
             dict(arrowstyle='->',
                  connectionstyle='arc3,rad=0.2'))

plt.text(-2,5,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size':25,'color':'r'})

plt.show()
```



## 不同图表

### 散点图

```python
import matplotlib.pyplot as plt
import numpy as np

n = 1024										# 生成n个数据
X = np.random.normal(0,1,n)						# 取随机数
Y = np.random.normal(0,1,n)						
T = np.arctan2(Y,X)								# 用于计算每个点的颜色
plt.scatter(X,Y,s=75,c=T,alpha=0.5)
##plt.scatter(np.arange(5),np.arange(5))
plt.xlim(-1.5,1.5)								# 设置取值范围
plt.ylim((-1.5,1.5))
plt.xticks(())									# 隐藏x轴的刻度值
plt.yticks(())
plt.show()
```



### 柱状图

```python
import matplotlib.pyplot as plt
import numpy as np

n = 12									# 生成12个数据
X = np.arange(n)						# 设置x值
										# 生成随机数
Y1 = (1 - X / float(n))*np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n))*np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
# 数据柱颜色为#9999ff，方向向上，边框为白色
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
# 数据柱颜色为#ff9999，方向向下，边框为白色

# 在数据柱的上方写上数据
for x, y in zip(X, Y1):
    # zip: 将X和Y1同时分别赋值给x，y
    # ha: horizontal alignment 	水平对齐
    # va: vertical alignment	垂直对齐
    plt.text(x+0, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    plt.text(x+0, -y-0.05, '-%.2f' % y, ha='center', va='top')

plt.xlim(-0.5,n)						# 设置x的取值范围
plt.ylim(-1.25, 1.25)					# 设置y的取值范围
plt.xticks(())							# 隐藏坐标轴刻度
plt.yticks(())
           
plt.show()
```



### 等高线

```python
import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    # 用于计算高度
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)				# 在x轴上生成256个点
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)					# 生成网格

# use plt.contourf to filling contours  来填充等高线颜色
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)
# cmap: color map 色谱

# 8:用来切分等高线的区域，值越大，等高线越密

# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

# adding label
# 用于设置等高线的值
# inline = ture 表示值在登高线里面
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()
```

### 数据图形化

```python
import matplotlib.pyplot as plt
import numpy as np

# image data
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405,0.651536351379])
.reshape(3,3)

"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
# interpolation: 调节清晰度
# cmap: color map 色谱
# origin: lower 按照从小到大的顺序排列

plt.colorbar(shrink=0.9)
# 将色值框的大小调成原来的0.9倍

plt.xticks(())
plt.yticks(())
plt.show()
```



### 3D数据

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D		# 用于3D的包

fig = plt.figure()							# 将画布的平面设置为3d的底面
ax = Axes3D(fig)	
# X, Y value
X = np.arange(-4, 4, 0.25)					# arange(起点,终点,步长)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)					# 建立网格图
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)								# 设置z的值

# 画3d图像
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'),edgecolor = 'black')
# rainbow: 彩虹色 
# rsride: 步长

"""
===========  ================================================
Argument      Description
===========  ================================================
*X*, *Y*, *Z* Data values as 2D arrays
*rstride*     Array row stride (step size), defaults to 10
*cstride*     Array column stride (step size), defaults to 10
*color*       Color of the surface patches
*cmap*        A colormap for the surface patches.
*facecolors*  Face colors for the individual patches
*norm*        An instance of Normalize to map values to colors
*vmin*        Minimum value to map
*vmax*        Maximum value to map
*shade*       Whether to shade the facecolors
===========  ================================================
"""

# I think this is different from plt12_contours
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# 将图像在z轴方向的投影，投在z=-2的平面上

"""
==========  ================================================
Argument    Description
==========  ================================================
*X*, *Y*,   Data values as numpy.arrays
*Z*
*zdir*      The direction to use: x, y or z (default)
*offset*    If specified plot a projection of the filled contour on this position in plane normal to zdir
==========  ================================================
"""

ax.set_zlim(-2, 2)						# 限制z轴的取值范围
plt.show()
```



## 图片的展示

### 将一个画布分成几个部分

```python
import matplotlib.pyplot as plt

# example 1:
###############################
plt.figure(figsize=(6, 4))
# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])

plt.subplot(2, 2, 3)
plt.plot([0, 1], [0, 3])

plt.subplot(2, 2, 4)
plt.plot([0, 1], [0, 4])

plt.tight_layout()          # 自动调节plt.subplot的间距

# example 2:
###############################
plt.figure(figsize=(6, 4))
# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 1, 1)
# figure splits into 2 rows, 1 col, plot to the 1st sub-fig
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)
# figure splits into 2 rows, 3 col, plot to the 4th sub-fig
plt.plot([0, 1], [0, 2])

plt.subplot(2, 3, 5)
# figure splits into 2 rows, 3 col, plot to the 5th sub-fig
plt.plot([0, 1], [0, 3])

plt.subplot(2, 3, 6)
# figure splits into 2 rows, 3 col, plot to the 6th sub-fig
plt.plot([0, 1], [0, 4])


plt.tight_layout()
plt.show()
```

### 分格显示

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# method 1: subplot2grid
##########################
# 这种方法比较好理解
plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # stands for axes
ax1.plot([1, 2], [1, 2])
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')
ax5 = plt.subplot2grid((3, 3), (2, 1))

# method 2: gridspec
#########################
plt.figure()
gs = gridspec.GridSpec(3, 3)
# use index from 0
ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:, 2])
ax9 = plt.subplot(gs[-1, 0])
ax10 = plt.subplot(gs[-1, -2])

# method 3: easy to define structure
####################################
f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.scatter([1,2], [1,2])

plt.tight_layout()
plt.show()
```



## 书写习惯

* 在输入参数的时候，如果是用变量带数值的形式，eg:plt.figure(num=3)

  等号两边尽量不要有空格，缩短代码宽度

  但是除此之外都要有空格

  以防字符之间太密集看不清楚，以至于出现错误

## 问题

* 图表中的字体如何设置
  * latex中的字体是什么
    * 英文要是Time New Roman
    * 中文要是宋体
    * 数学公式，latex默认是cm
    * 具体论文要采用sitx
* python的语法习惯是要查一下
* 要下载一下pycharm
* 掌握一下Python中MATLAB的一下操作如何使用
  * 如矩阵的运算
  * 函数如何标识
* 还要找一下相关的书籍
* TODO Python中单引号和双引号的区别
  * 没有区别
  * 最好单词用单引号
  * 句子用双引号

