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
    def __init__(self,filepath='./email/',testpath='./test/',C=1000):   ###传入训练数据集的路径和测试数据集的路径
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
    A=Email()
    A.User()