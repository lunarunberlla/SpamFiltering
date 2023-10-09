# Naive Bayes

## Introduction:

​		[贝叶斯定理](https://baike.baidu.com/item/贝叶斯定理/1185949?fromModule=lemma_inlink)由英国数学家贝叶斯 ( Thomas Bayes 1702-1761 ) 发展，用来描述两个条件[概率](https://baike.baidu.com/item/概率?fromModule=lemma_inlink)之间的关系。假设现在我们有两个事件A和B

$$
P(A|B)=\frac {P(A,B)} {P(B)}
$$

$$
P(B|A)=\frac {P(A,B)}{P(A)}=\frac {P(A|B)*P(B)}{P(A)}
$$

由加法公式可得：

$$
P(A)=ΣP(A|X_i)_{(X表示一个事件集合中所有的与A相关的但是非A事件集合，X_i表示X集合中的一个元素)}
$$

那么我们的:

$$
P(B|A)=\frac {P(A,B)}{P(A)}=\frac {P(A|B)*P(B)}{ΣP(A|X_i)}
$$

## Derivation：

### Conditional probability:

​			首先我们来介绍一下条件概率，条件概率这个东西很简单啊。举一个简单的例子，就比如今天买一朵玫瑰花的概率是P(A),<sub>(A表示今天我买玫瑰花这一个事件)</sub>,那么在我买了一朵玫瑰花的条件下，我还会再买一朵玫瑰花嘛？可能会，也可能不会，但是，我们不得不承认，今天第一次买玫瑰花这一个事件确实对今天我第二次买玫瑰花产生了一定的影响，这很容易理解。因为我想如果没有特别特殊原因的话，我不会一天去买两朵玫瑰花，当然，除非我有两个女朋友，早上我去见了第一个女朋友，买了一束话，晚上又见了另一个女朋友，又买了一束花。好了，言归正转，现在我们要考虑我买第二朵玫瑰花发生的概率了，首先我们用这样一个符号来记录我们要得到的概率：

$$
P(B|A)
$$

很自然的，A代表的是我们今天第一次买玫瑰花的这个事件，B代表的是今天我们第二次买玫瑰花的这个事件。上面的值就代表的是，今天我在第一天买了玫瑰花的情况下，我们要第二次买到这个玫瑰花的概率。

### Joint prob：

​			这串英文翻译过来就是叫联合概率，那么什么是联合概率呢？其实很容易理解，就像我们上面的玫瑰花一样，假设现在通过对我好几千年的观察，发现，我一天买一朵玫瑰花的概率是:<sub>(A为我每天买玫瑰花这个事件，a为具体的实例，i代表朵数)</sub>

$$
P(A=a_1)
$$

在这几千年的观察中，观察者还发现了一个事情，就是，我每天都要买几瓶可口可乐。那么我每天买几瓶可口可乐呢？经过观察统计发现，我每天买i瓶可乐的概率为：<sub>(B为我每天买可乐这个事件，b为具体的实例，i代表瓶数)</sub>

$$
P(B=b_i)
$$

哎，这样一来，新的问题就出现了：今天我买了两朵玫瑰花，同时又买了五瓶可乐这个事情发生的概率有多大。首先这个概率值我们记做：

$$
P(A=a_2,B=b_5)
$$


### Product Rule：

​			按照我们通常的理解就是买可乐和买玫瑰花，这两个事件是毫无相干的，就像风马牛一样，不相及。也就是这两个事件的相互独立的，这个时候那么我们要的概率值就是：

$$
P(A=a_2,B=b_5)=P(A=a_2)*P(B=b_5)
$$

可是，我个人感觉这两个事件并不是毫无相干的。因为，例如我今天买了999多玫瑰花去向我的女神表白，但是被拒绝了，在回来的路上十分的抑郁，剪不断，理还乱。这个时候，突然一抬头，一辆大卡车，出现在了我的面前，欧玛噶。我见了上帝，今天是喝不到可乐了。你看买玫瑰花，和我喝多少可乐是不是还是有联系的，对吧？虽然有一些牵强，但是还是有联系的，这个时候就不能按照上面那个公式来计算了，那么我们应该怎么计算呢？还是通过观察，经过长达九九八十一千年的观察，观察者发现，我每天买两朵花的概率是P（A=a<sub>2</sub>）,在我买完两朵玫瑰花后，又买五瓶可乐的概率是P(B=b<sub>5</sub>|A=a<sub>2</sub>)那么我买玫瑰花之前，我想知道我买了两朵玫瑰花同时又买了五瓶可乐的概率就是：

$$
P(A=a_2,B=b_5)=P(B=b_5|A=a_2)*P(A=a_2)
$$

这个公式看起来可能有些抽象，但是其实没有那么难，他的本质还是两个事件的概率乘积，但是，对于两个有关联的事件，一个事件的概率会随着另一个事件的改变而改变，我们只是找到正确的概率值相乘而已。同样的，上面的公式我们找了B事件随着A事件的改变而改变的概率值，当然我们也可以尝试这找到事件A随着事件B概率的变换而变换的值，那么

$$
P(A=a_2,B=b_5)=P(B=b_5)*P(A=a_2|B=b_5)=P(B=b_5|A=a_2)*P(A=a_2)
$$

当然，对于多个事件，我们依然可以这样进行计算。假设现在我们有一个事件集合X，x<sub>i</sub>为该事件集合的具体摸一个事件，那么我们考虑多个随机事件的情况：

$$
P(X=x_1,X=x_2,X=x_3,...,X=x_i)=P(X=x_1)*P(X=x_2,X=x_3,...,X=x_i|X=x_1)
$$


$$
=P(X=x_1)*P(X=x_2|X=x_1)P(X=x_3,...,X=x_i|X=x_1,X=x_2)
$$

$$
=P(X=x_1)P(X=x_2|X=x_1)P(X=x_3,...,X=x_i|X=x_1,X=x_2)...P(X=x_i|X=x_1,...,X=x_{i-1})
$$



### Bayes:

​			有了上面的一些内容，下面我们再来想这样一件事情。假设现在我已经知道了，我在买了两朵玫瑰花的情况下又买了两瓶可乐的概率为P(B=b<sub>5</sub>|A=a<sub>2</sub>)，即：

$$
P(B=b_5|A=a_2)
$$

那么我们能不能知道我在买了五瓶可乐的情况下，买了两多玫瑰花的概率呢？就是我们想要求得：

$$
P(A=a_2|B=b_5)
$$

我想这不会太难的，根据上面我们推导的乘法规则里的第三个式子：

$$
P(A=a_2,B=b_5)=P(B=b_5)*P(A=a_2|B=b_5)=P(B=b_5|A=a_2)*P(A=a_2)
$$

不难发现：

$$
P(A=a_2|B=b_5)=\frac{P(A=a_2,B=b_5)}{P(B=b_5)}=\frac{P(B=b_5|A=a_2)P(A=a_2)}{P(B=b_5)}
$$

我们已经知道了P(B=b<sub>5</sub>|A=a<sub>2</sub>)的值，那么我们只要求得P（A=a<sub>2</sub>)和P（B=b<sub>5</sub>）的值便可以计算出来我们想要的值。这便是贝叶斯公式，实际上它是很简单的。但是通常情况下我们碰到的情况可能要比这复杂很多。因为事务之间的联系不可能只有玫瑰花和可乐之间的联系，还有好多好多的情况。就比如我们要判断一对夫妻未来生出来的孩子长得漂亮还是不漂亮，那么我们可以参考的依据就有很多了，比如他爸爸的眼睛大小，他妈妈的眼睛大小，他爸爸的海拔，他妈妈的海拔等等有诸多的因素。所以往往我们要的这个公式是很长的：

$$
P(Y=y_i|x_1,x_2,x_3,...,x_i)=\frac{P(x_1,x_2,x_3,...,x_i|Y=y_i)P(Y=y_i)}{P(x_1,x_2,x_3,...,x_i)}
$$

我们按照乘法规则将他们展开：

$$
P(Y=y_i|x_1,x_2,x_3,...,x_i)=\frac{P(x_1|x_2,x_3...,x_i,Y=y_i)P(x_2|x_3,x_4...,x_i,Y=y_i)...P(x_i|Y=y_i)P(Y=Y_i)}{P(x_1,x_2,x_3,...,x_i)}
$$

这样看来，这个贝叶斯公式的计算难度还是很大，因为我们无法忽略这些X=(x<sub>1</sub>，x<sub>2</sub>，...，x<sub>i</sub>)内部之间的联系，就像可能1.6m的女生要找1.8m的男性一样，这个内部纯在着某种联系，因为1.6m的女性找一个1.5m的男性说啥豆不太合适，所以P(X=x<sub>i</sub>)这样的东西是很难算出来的。那么怎么办呢？



### Naive Bayes：

​			为了解决上面的问题，我们就引出来朴素贝叶斯这个方法。所谓的朴素就是忽略掉X=(x<sub>1</sub>，x<sub>2</sub>，...，x<sub>i</sub>)内部之间的联系。也就是x<sub>1</sub>，x<sub>2</sub>，...，x<sub>i</sub>他们之间是相互独立的：

$$
P(x_1,x_2,x_3,...,x_i)=P(x_1)P(x_2)P(x_3)...P(x_i)
$$

同样的：

$$
P(x_1|x_2,x_3...,x_i,Y=y_i)=P(x_1|y_i)
$$

所以我们将贝叶斯公式变为朴素贝叶斯公式就是：

$$
P(Y=y_i|x_1,x_2,x_3,...,x_i)=\frac{P(x_1|Y=y_i)P(x_2|Y=y_i)...P(x_i|Y=y_i)P(Y=Y_i)}{P(x_1,x_2,x_3,...,x_i)}
$$

在机器学习里面，我们要判断事件Y是属于哪一个类别，我们可以分别计算出：

$$
P(Y=y_1|x_1,x_2,x_3,...,x_i)
$$

$$
P(Y=y_2|x_1,x_2,x_3,...,x_i)
$$

$$
...
$$

$$
P(Y=y_i|x_1,x_2,x_3,...,x_i)
$$



然后将他们的概率值进行比较，取概率值最大的为我们预测的结果。在计算的过程中不难发现，他们的分母都是一样的，都是P(x<sub>1</sub>，x<sub>2</sub>，...，x<sub>i</sub>)，所以其实我们只需要计算分子部分作比较就可以了，就是这一部分：

$$
P(x_1|Y=y_i)*P(x_2|Y=y_i)...P(x_i|Y=y_i)*P(Y=y_i)
$$

这便是我们朴素贝叶斯公式的推导全过程了。下面我们来讨论一些再计算的时候可能出现的问题。

## Data Processing:

### Continuous Value Processing:

#### Gaussian Distribution：

​			在进行处理之前，我们回想一下我们的高斯分布，也就是正态分布。第一次接触高斯分布是在高中的时候，但是那个时候，仅仅停留在这么应用高斯分布去解题，没有实际的理解过他。那个时候，给出一组数据去计算他的均值和方差，然后巴拉巴拉一大堆。至于为什么是那样没有思考过。但是在大学上概率论的时候，思考了很长时间。终于弄明白他是什么意思了。其实说白了就是大多数都是事情的发生都是在中间的，至于那些极大值或者极小值的发生都是偶然发生的。就像我们买彩票，大多数情况都是中不了奖的。但是也有极少数可以中大奖，极其极其的少数中两次以上的大奖，这个就是符合高斯分布的。所谓的分布，就是给定一个x值，可以计算出这个x值发生的概率值。大致就是这样，也许表达的不是很清楚，这个其实是一种感觉上的东西，目前还没有想到该怎么表达。下面我们给出他的公式：

$$
X\leadsto N(\mu,\sigma^2)
$$

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp^{\frac{-(x-\mu)^2}{2\sigma^2}}
$$



#### Application:

​			当我们在数据集中遇到连续值的时候，我们就认为这个连续值是符合高斯分布的。因为，万事万物基本上都符合高斯分布，其实无论是二项分布，泊松分布还是指数分布，他们的图像和高斯分布都是高度相似的。我们将连续值按照标签进行划分，将划分的每一个集合都算一个高斯分布出来。在测试集合中将数值代入该高斯分布得到的函数值就是我们要求的概率值。举一个例子，现在我们在Y=y<sub>1</sub>的条件下，有一群连续的属性值(x<sub>1</sub>，x<sub>2</sub>，...，x<sub>i</sub>)。该属性下测试样本的值为X。那么我们计算：

$$
\mu=\frac{Σx_i}{Σ\Pi(x)}
$$

$$
\sigma^2=Σ(x_i-\mu)^2
$$

这样我们认为：

$$
X\leadsto N(\mu,\sigma^2)
$$

那么：

$$
P(X|Y=y_1)=f(X)=\frac{1}{\sigma\sqrt{2\pi}}\exp^{\frac{-(X-\mu)^2}{2\sigma^2}}
$$



### Missing Value Handling:

​			在有些时候，我们的测试样本中的某一些属性值，在我们的训练样本中是没有的，那么对应这个属性值的条件概率就变成了0，假设这个属性值是X=x<sub>m</sub>,其在Y=y<sub>m</sub>下是没有的，那么：

$$
P(x_m|y_m)=0
$$

对应的：

$$
P(x_1|y_m)*P(x_2|y_m)...P(x_3|y_m)...P(x_i|y_m)=0
$$

这显然是不合理的。对于这种情况，我们采用拉普拉斯平滑来解决。

#### Laplacian smoothing：

​			拉普拉斯平滑，其实也很简单。假设现在`我们有一个随机变量X，其取值范围为（1，2，3，...，i）在进行了k次观察后，其对应的实验结果为Y=(y<sub>1</sub>，y<sub>2</sub>，...，y<sub>k</sub>),那么P(Y=y<sup>m</sup>)的极大似然估计值就是：

$$
P(Y=y^m)=\frac{Σ\Pi(Y=y^m)}{k}
$$

这很容易理解，就是Y=y<sup>m</sup>的数量除以观测的次数。这个是没有做拉普拉斯平滑之前的数值，做拉普拉斯平滑之后就变成了：(α为平滑系数，通常为1)

$$
P(Y=y^m)=\frac{Σ\Pi(Y=y^m)+1\alpha}{k+i\alpha}
$$

那怎么理解这个拉普拉斯平滑呢？说白了其实就是为了避免零概率的出现，我们在原来的分子上加1，同时为了减小分子加1对数据整体统计规律产生的影响，我们在其分母部分加上该随机变量可能的取值的个数。至于为什么，我还没有参悟。



## Result:

### Polynomial:

![2](./img/2.png)

#### Data:

![1](./img/1.png)

#### Analysis:



<h4>分类结果混肴矩阵:</h4>   <!--标题-->
<table border="1" width="500px" cellspacing="10">
<tr>
  <th align="left">真实情况\预测结果</th>
  <th align="center">正例</th>
  <th align="right">反例</th>
</tr>
<tr>
  <td>正例</td>
  <td>TP=277</td>
  <td>FN=47</td>
</tr>
<tr>
  <td>反例</th>
  <td>FP=12</td>
  <td>TN=351</td>
</tr>
</table>


$$
Accuracy={(TP+TN)\over(TP+TN+FN+FP)}=0.9141193595342066 
$$

$$
Precision={TP\over(TP+FP)}=0.9584775086505191
$$

$$
Recall={TP\over(TP+FN)}=0.8549382716049383
$$

### Bernoulli:

![3](./img/3.png)

#### Data:

![1](./img/1.png)

#### Analysis:



<h4>分类结果混肴矩阵:</h4>   <!--标题-->
<table border="1" width="500px" cellspacing="10">
<tr>
  <th align="left">真实情况\预测结果</th>
  <th align="center">正例</th>
  <th align="right">反例</th>
</tr>
<tr>
  <td>正例</td>
  <td>TP=363</td>
  <td>FN=0</td>
</tr>
<tr>
  <td>反例</th>
  <td>FP=114</td>
  <td>TN=210</td>
</tr>
</table>


$$
Accuracy={(TP+TN)\over(TP+TN+FN+FP)}=0.834061135371179
$$

$$
Precision={TP\over(TP+FP)}=0.7610062893081762
$$

$$
Recall={TP\over(TP+FN)}=1.0
$$

### The difference between scaled and unscaled：（复制作业的结论）



​			1：从实验结果上来看，他们的预测结果都是正确的。并没有受到特征缩放的影响。分析其原因：

​						a:该数据集的数据都是连续性的数据。对于连续性的数据，我们采用高斯分布的概率密度函数来映射出一个相应的概率值。值得注意的是，这里的概率密度函数值并不是真正意义上的概率值，因为他是一个概率密度函数我们只有对于某一个区域求积分才能算得其概率。而对于某一点的概率值它总是为零的。我们只是用概率密度的函数值来代替了其概率，并不是真正的概率。

​						b:使用高斯分布的时候，我们需要求得的参数是特征值的均值和方差。对于数据来说当数据进行缩放以后，其方差和均值都会显著减小。高斯分布的形状也会发生变化，但是这一过程并不会改变数据的内部大小关系，因此高斯分布的特征，相对于未缩放的数据来说并没有完全丢失。而是在原有的分布特征下，进行了一些调整，仍然还保留了原来的分布的关键信息。



​		 2：我们再来看其两个预测结果的比值：

​						未缩放的数据，预测的结果的比值明显要比缩放的数据预测的比值要大。看起来未缩放的数据在比较过程中更加的具有差异性，但是观察其概率取值，均是小于1大于0的取值，对于这样的数值其比值的大小，并不能有效衡量其差异性，因为特征的不规则性导致在数据的方差较大，函数自变发生变化时，函数值变化不会很明显，这很容易理解。而对于缩放后的数据来说，其预测的概率值都是大于1的。因为我们是用的高斯分布的概率密度函数值代替了其概率， 所以概率值大于1很正常。同样的，由于特征的缩放，方差减小，导致其概率密度函数变得极其苗条，两个数值之间的变化差异性极其明显。

​			3：总结：对于朴素贝叶斯算法来说，无论特征缩放还是不缩放其各有好坏。不缩放特征可以很大程度上保障其数据的原有的内涵不被破坏。缩放特征可以更加有效的突出对于结果的差异性质，也扩充了其包容性。

## Summary:

​		整体来说这个算法不是很难的，具体利用伯努利分布的朴素贝叶斯算法，在课后作业里面已经实现过一次了，在哪里做了特征的缩放，高斯分布啊上面的，等多种对比，所以的话这次垃圾邮件分类对于使用伯努利模型还是调用了sklearn里面的包。但是多项式模型的话，还是自己写了一次。感觉多项式模型还是好用一点。从实验的结果就能看出来了，各个方面的数据都是多项式模型要更好一点的。当然了，能有这样的理解，还是离不开老师讲的好呀，能碰到这样一位好老师，我哭死。

## Code:

```python
import os
import wordninja
import pandas
from colorama import Fore, init
from sklearn.naive_bayes import BernoulliNB
'''由于数据处理部分没有调包啊，所以这个部分运行非常的慢，再加上有好多数据集'''
init()
###################################多项式分布#######################################################
class Email:           ## 多项式分布
    def __init__(self,filepath='./email/',testpath='./test/',C=1000):   ###传入训练数据集的路径和测试数据集的路径 这个C是为了放在数值溢出而进行缩放的
        self.filepath=filepath
        self.testpath=testpath
        self.C=C
    def dataset(self):                                           #### 构建训练数据集
        '''采用的是多项式模型，以单词为颗粒，所以把所有的单词分类整合在一起就行'''
        SourceData=[]
        cu=0
        for item in os.listdir(self.filepath):
           content=[]
           for value in os.listdir(self.filepath+item+'/'):
               cu=cu+1
               print(Fore.GREEN,"正在加载第{}个文件...".format(cu))
               with open(self.filepath+item+'/'+value, 'r+', encoding='utf8',errors='ignore') as f:
                    content =content+ wordninja.split(f.read())   ####采用了wordninja库，将邮件进行分词
           SourceData.append(content)
        print(Fore.YELLOW,"文件加载完成")
        return SourceData                                        ##### 这里是一个二维的列表，一个是非垃圾邮件单词的集合，一个是垃圾邮件单词的集合

    def classfies(self):                                         #### 构建每一个单词分别在不同类别中出现的概率
        PrediectValue={}
        Data=Email.dataset(self)
        #防止概率值太小都乘以2000，不影响结果
        '''先传入P（X） P（Y）'''
        PrediectValue.update({'PX1':((len(Data[0])+1)*self.C/(len(Data[0])+len(Data[1])+2))}) #做拉普拉斯平滑
        PrediectValue.update({'PX0': ((len(Data[1]) + 1)*self.C / (len(Data[0]) + len(Data[1]) +2))})  # 做拉普拉斯平滑
        ''' 使用pandas.value_counts这个库进行概率计算，大幅度缩小训练时间'''
        PrediectValue1=(pandas.value_counts(Data[0])+1)*self.C/(len(Data[0])+len(set(Data[1]+Data[0])))
        PrediectValue0=(pandas.value_counts(Data[1])+1)*self.C/(len(Data[1])+len(set(Data[1]+Data[0])))

        return PrediectValue,PrediectValue1,PrediectValue0,Data

    def  User(self):
        PrediectValue, PrediectValue1, PrediectValue0,Data=Email.classfies(self)
        classflyerror=[]
        P1List=list(PrediectValue1.index)
        P0List=list(PrediectValue0.index)
        TN,TP,FN,FP=0,0,0,0
        count = 1
        cue=len(os.listdir(self.testpath+os.listdir(self.testpath)[0]+'/'))
        for item in os.listdir(self.testpath):   ###加载测试数据

           for value in os.listdir(self.testpath+item+'/'):
               P1 = PrediectValue['PX1']
               P0=  PrediectValue['PX0']
               with open(self.testpath+item+'/'+value, 'r+', encoding='utf8',errors='ignore') as f:
                    for i in wordninja.split(f.read()):
                        '''计算正例的值'''
                        if i not in P1List:
                            P1=P1*(1*self.C /(len(Data[0])+len(set(Data[1]+Data[0])))) #如果测试样本不在里面，做拉普拉斯平滑，取对数变换，
                        else:
                            P1=P1*PrediectValue1[i]                                             #如果测试样本在里面，计算概率值，取对数变换
                        '''计算负例的值'''
                        if i not in P0List:
                            P0=P0*(1*self.C/(len(Data[1])+len(set(Data[0]+Data[1]))))
                        else:
                            P0=P0*PrediectValue0[i]
                    '''比较样本属于哪一个类的概率大'''
                    if P1 >= P0:                ##比较样本属于哪一个类的概率大
                        if count<=cue:
                            TN=TN+1
                        else:
                            FN=FN+1
                            classflyerror.append(self.testpath + item + '/' + value)
                        print(Fore.BLUE,"该样本为非垃圾邮件")
                    else:
                        if count>cue:
                            TP=TP+1
                        else:
                            FP=FP+1
                            classflyerror.append(self.testpath + item + '/' + value)
                        print(Fore.RED, "该样本为垃圾邮件")
                    print(P0,P1)
               count=count+1
        print(Fore.BLUE,"一共测试{}个文件".format(TN+TP+FN+FP))
        print(Fore.GREEN,"TN={} TP={} FN={} FP={}  Recall={} Accuract={} Precision={}".format(TN,TP,FN,FP,TP/(TP+FN),(TP+TN)/(TP+TN+FN+FP),TP/(TP+FP)))

class Email2:                            ##伯努利分布
    def __init__(self,filepath='./test/',testpath='./test/',C=1000):   ###传入训练数据集的路径和测试数据集的路径
        self.filepath=filepath
        self.testpath=testpath
        self.C=C
    def dataset(self):                                           #### 构建训练数据集
        '''采用的是伯努利模型，以文章为颗粒'''
        SourceData=[]
        cu=0
        cc00 = []
        for item in os.listdir(self.filepath):
           content=[]
           for value in os.listdir(self.filepath+item+'/'):
               cu=cu+1
               print(Fore.GREEN,"正在加载第{}个文件...".format(cu))
               with open(self.filepath+item+'/'+value, 'r+', encoding='utf8',errors='ignore') as f:
                    data=wordninja.split(f.read())
                    content.append(data)   ####采用了wordninja库，将邮件进行分词
                    cc00=cc00+data
           SourceData.append(content)
        X,Y=[],[]
        '''将文章转换为向量'''
        for i in range(len(SourceData)):
                for item in SourceData[i]:
                    list1=[]
                    for j in sorted(set(cc00)):
                        if j in item:
                            list1.append(1)
                        else:
                            list1.append(0)
                    if i==0:
                        Y.append('1')
                    else:
                        Y.append('0')
                    X.append(list1)
        print(Fore.YELLOW,"文件加载完成")

        return X,Y,cc00                                       ##### 这里是一个二维的列表

    def classfies(self):                                         #### 调用sklearn，训练模型
       X,Y,cc00=Email2.dataset(self)
       model=BernoulliNB()
       model.fit(X,Y)
       return model,cc00

    def  User(self):
        model,cc00=Email2.classfies(self)
        SourceData = []

        for item in os.listdir(self.testpath):   ###加载测试数据
           content = []
           for value in os.listdir(self.testpath+item+'/'):
               with open(self.testpath+item+'/'+value, 'r+', encoding='utf8',errors='ignore') as f:
                   data = wordninja.split(f.read())
                   content.append(data)  ####采用了wordninja库，将邮件进行分词

           SourceData.append(content)

        X, Y = [], []
        '''将文章转换为向量'''
        for i in range(len(SourceData)):
            for item in SourceData[i]:
                list1 = []
                for j in sorted(set(cc00)):
                    if j in item:
                        list1.append(1)
                    else:
                        list1.append(0)
                X.append(list1)
                if i == 0:
                    Y.append('1')
                else:
                    Y.append('0')

        result=model.predict(X)
        TP,TN,FP,FN=0,0,0,0
        for i in range(len(Y)):
            if Y[i]=='1':
                if result[i]=='1':
                    TP=TP+1
                else:
                    FN=FN+1
            else:
                if result[i]=='0':
                    TN=TN+1
                else:
                    FP=FP+1

        print(Fore.BLUE, "一共测试{}个文件".format(TN + TP + FN + FP))
        print(Fore.GREEN,
              "TN={} TP={} FN={} FP={}  Recall={} Accuract={} Precision={}".format(TN, TP, FN, FP, TP / (TP + FN),
                                                                                   (TP + TN) / (TP + TN + FN + FP),
                                                                                   TP / (TP + FP)))
if __name__ == '__main__':
    A=Email2()
    A.User()
```

​		

