# Student Intervention System
建立学生干预系统

![模型选择](https://github.com/JCancy/Student-Intervention-System/picture/choose algorithm.PNG)

![github](Student-Intervention-System/picture/choose algorithm.PNG)

![git1](https://github.com/JCancy/Student-Intervention-System/blob/master/picture/choose%20algorithm.PNG)

![moxing](Student-Intervention-System/picture/choose algorithm.PNG)

主要内容：使用pandas，numpy和pyplot探索数据，通过将非数值数据转换为数值数据对数据进行预处理，使用适当的监督学习算法（如朴素贝叶斯，逻辑回归和支持向量机）进行预测，并计算和比较F1分数。

文章来源：翻译自《Building a Student Intervention System》
http://www.ritchieng.com/machine-learning-project-student-intervention
作者简介：本文作者Ritchie Ng是来自新加坡国立大学（NUS）的深度学习研究人员、全球优秀学者，专门从事深度学习、计算机视觉和自然语言处理方面的研究，其研究项目涉及TensorFlow，PyTorch，TensorLayer，Keras，OpenCV，scikit-learn，CUDA，cuDNN，Python以及C / C++等相关内容。
作者网站：http://www.ritchieng.com

## 1 项目内容：建立学生干预系统
在这个项目中，我们的目标是根据已提供的学生个人数据以及往届学生的毕业情况，找出学生各项指标和学生能否毕业之间的关系。划分出训练集和测试集，对建立的模型进行训练与测试，根据各项指标将学生分为“可能毕业”与“可能不会毕业”两类，从而确定是否需要对学生进行提前干预。

本项目使用的数据集在student-data.csv中，数据包括学生的基本信息（性别、年龄、健康状况等）、学生的家庭信息（家庭大小、父母受教育程度、父母工作情况等）、学生的学习情况（学校、每周学习时间、旷课次数、挂科次数等）以及学生的课余生活状况（是否有课外活动、放学后的课余时间、工作日及周末的酒精消费情况等）共计31个指标。其中，最后一列‘passed’（学生是否毕业）将是我们的目标标签，其他所有列是每个学生的特征指标。

为了了解我们的项目内容，确定数据集的基本情况，获知往届学生的毕业率，我们将首先计算出我们所关注的内容：学生总人数、每个学生的特征总数、毕业的学生人数、未毕业的学生人数以及毕业率等，以此来了解学生数据的基本状况，以便我们在后续的数据准备中确定特征和目标列，并对特征列进行预处理，以实现训练集和测试集的拆分，为建立模型做准备。

### 1.1 数据探索
运行下面的代码，加载必要的Python库和学生数据。请注意， 最后一列‘passed’（学生是否毕业）将是我们的目标标签，其他所有列是每个学生的特征指标。

In [1]:
```python 
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
Student data read successfully!
```

In [2]:
```python
# Further Exploration using .head()
student_data.head()
```
Out[2]:
school	sex	age	address	famsize	Pstatus	Medu	Fedu	Mjob	Fjob	...	internet	romantic	famrel	freetime	goout	Dalc	Walc	health	absences	passed
0	GP	F	18	U	GT3	A	4	4	at_home	teacher	...	no	no	4	3	4	1	1	3	6	no
1	GP	F	17	U	GT3	T	1	1	at_home	other	...	yes	no	5	3	3	1	1	3	4	no
2	GP	F	15	U	LE3	T	1	1	at_home	other	...	yes	no	4	3	2	2	3	3	10	yes
3	GP	F	15	U	GT3	T	4	2	health	services	...	yes	yes	3	2	2	1	1	5	2	yes
4	GP	F	16	U	GT3	T	3	3	other	other	...	no	no	4	3	2	1	2	5	4	yes
5 rows × 31 columns

In [3]:
```python
# This is a 395 x 31 DataFrame
student_data.shape
```
Out[3]:
(395, 31)

In [4]:
```python
# Type of data is a pandas DataFrame
# Hence I can use pandas DataFrame methods
type(student_data)
```
Out[4]:
pandas.core.frame.DataFrame
 
 
从上述代码的运行结果可以看出，该数据集共包含395名学生的31个指标，由于数据集类型是pandas数据类，我们可以直接使用pandas对该数据集进行处理。

为了确定我们有多少学生的信息，并了解这些学生的毕业率。在后面代码中，需要计算以下内容：
* 学生总人数n_students。
* 每个学生的特征总数n_features。
* 毕业的学生人数n_passed。
* 未毕业的学生人数n_failed。
* 毕业率grad_rate，使用百分比（%）。


In [5]:
```python
# TODO: Calculate number of students
n_students = student_data.shape[0]

# TODO: Calculate number of features
n_features = student_data.shape[1] - 1

# TODO: Calculate passing students
# Data filtering using .loc[rows, columns]
passed = student_data.loc[student_data.passed == 'yes', 'passed']
n_passed = passed.shape[0]

# TODO: Calculate failing students
failed = student_data.loc[student_data.passed == 'no', 'passed']
n_failed = failed.shape[0]

# TODO: Calculate graduation rate
total = float(n_passed + n_failed)
grad_rate = float(n_passed * 100 / total)

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)
```
Total number of students: 395
Number of features: 30
Number of students who passed: 265
Number of students who failed: 130
Graduation rate of the class: 67.09%
 
从输出结果可以看到，总学生数为395，特征指标为30（‘passed’被我们作为了目标标签），成功毕业的人数为265人，未能毕业的人数为130人，毕业率为67.09%。

#### 问题1 - 分类与回归
在这个项目中，我们的目标是确定可能不能毕业而需要及早干预的学生。这是哪种类型的监督学习，分类还是回归？为什么？

#### 答案：
* 这应该是一个分类问题。
* 这是因为它有两个离散的结果，这是典型的分类问题：
	1. 需要早期干预的学生。
	2. 不需要及早干预的学生。
* 我们可以用二进制结果进行相应的分类，如：
	1. 是，1，需要早期干预的学生。
	2. 否，0，不需要早期干预的学生。
* 由于我们并不想预测得到一个连续的结果，因此这不是一个回归问题。

## 2 数据准备
在这一部分，我们要准备用于建模、训练和测试的数据。

### 2.1 确定特征和目标列
通常情况下，我们获得的数据包含非数字类型的特征。这是一个比较麻烦的问题，因为大多数机器学习算法更希望使用数字类型的数据进行计算。

运行下面的代码将学生数据分成特征和目标列，查看是否有非数字类型的特征。
 
 
In [6]:
```python
# Columns
student_data.columns
```
Out[6]:
Index([u'school', u'sex', u'age', u'address', u'famsize', u'Pstatus', u'Medu', u'Fedu', u'Mjob', u'Fjob', u'reason', u'guardian', u'traveltime', u'studytime', u'failures', u'schoolsup', u'famsup', u'paid', u'activities', u'nursery', u'higher', u'internet', u'romantic', u'famrel', u'freetime', u'goout', u'Dalc', u'Walc', u'health', u'absences', u'passed'], dtype='object')

In [7]:
```python
# We want to get the column name "passed" which is the last 
student_data.columns[-1]
```
Out[7]:
'passed'

In [8]:
```python
# This would get everything except for the last element that is "passed"
student_data.columns[:-1]
```
Out[8]:
Index([u'school', u'sex', u'age', u'address', u'famsize', u'Pstatus', u'Medu', u'Fedu', u'Mjob', u'Fjob', u'reason', u'guardian', u'traveltime', u'studytime', u'failures', u'schoolsup', u'famsup', u'paid', u'activities', u'nursery', u'higher', u'internet', u'romantic', u'famrel', u'freetime', u'goout', u'Dalc', u'Walc', u'health', u'absences'], dtype='object')

 
我们在此获得了除目标标签“passed”以外的所有列标签。接下来进行特征列的提取，首先将数据转换为一个表，将数据分为特征数据和目标数据。

 In [9]:
```python
# Extract feature columns
# As seen above, we're getting all the columns except "passed" here but we're converting it to a list
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
# As seen above, since "passed" is last in the list, we're extracting using [-1]
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()
```
Feature columns:
['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

Target column: passed

Feature values:
  school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
0     GP   F   18       U     GT3       A     4     4  at_home   teacher
1     GP   F   17       U     GT3       T     1     1  at_home     other
2     GP   F   15       U     LE3       T     1     1  at_home     other
3     GP   F   15       U     GT3       T     4     2   health  services
4     GP   F   16       U     GT3       T     3     3    other     other

    ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
0   ...       yes       no        no       4         3     4    1    1      3
1   ...       yes      yes        no       5         3     3    1    1      3
2   ...       yes      yes        no       4         3     2    2    3      3
3   ...       yes      yes       yes       3         2     2    1    1      5
4   ...       yes       no        no       4         3     2    1    2      5

  absences
0        6
1        4
2       10
3        2
4        4

[5 rows x 30 columns]
 
从打印结果可以看出，数据已经转换成表的形式，“passed”为我们的目标列，其余指标列为特征列。

### 2.2 特征列预处理
可以看到，有几个非数字列需要转换。其中许多列只是yes/ no，例如internet。这些可以转换为1/ 0（二进制）值。
对于其他列，如Mjob和Fjob，具有两个以上的值，称为分类变量。处理此类列的建议方法是，创建与所有可能值的数量相同的列（例如Fjob_teacher，Fjob_other，Fjob_services等），并将1分配给其中一列，0分配给其他所有列。

有时这些生成的列称为虚拟变量，我们将使用pandas.get_dummies()函数来进行这一转换。运行下面的代码来执行这一部分所讨论的预处理程序。
 
 
In [10]:
```python
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)

        # Collect the revised columns
        output = output.join(col_data)

    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
```
Processed feature columns (48 total features):
['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
 
通过以上代码，我们对数据进行了简单的处理，并成功将非数字二进制变量转换为了二进制变量，分类变量转换为虚拟变量。

### 2.3 拆分训练数据和测试数据
到目前为止，我们已将所有的分类特征转换为数值类型。下一步，我们将数据（特征和其相应的标签）分为训练集和测试集。在下面的代码中，需要实现以下内容：
* 随机重组，将数据（X_all，y_all）拆分为训练子集和测试子集。
	* 使用300个训练点（约75％）和95个测试点（约25％）。
	* 为所用的函数设置random_state（如果提供了的话）。
	* 将结果储存在X_train，X_test，y_train，和y_test中。
**特别提示：**数据评估对训练集/测试集的影响
在处理新的数据集时，较好的做法是评估其具体特征，并使用根据这些特征制定的交叉验证方法，在本例中有两个要点：
* 我们的数据集较小。
* 我们的数据集略有些不平衡。（已经通过的学生比正在通过的学生更多）
**我们能做什么？**
* 我们可以利用K-折交叉验证来探索小数据集
* 如果我们需要处理严重不平衡的数据集，即使在这种情况下可能没有必要，我们也还是可以使用分层K-折和分层随机分流交叉验证法来解决我们的数据集不平衡性，因为分层保留了每个类别的样本百分比
	* http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
	* http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html

 
In [11]:
```python
# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split

In [12]:
```python
# For initial train/test split, we can obtain stratification by simply using stratify = y_all:
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify = y_all, test_size=95, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
```
Training set has 300 samples.
Testing set has 95 samples.

In [13]:
```python
# To double check stratification
print np.mean(y_train == 'no')
print np.mean(y_test == 'no')
```
0.33
0.326315789474
 
对于初始的训练集测试集分离，我们可以使用分层的方法得到分层数据。结果显示训练集有300个样本，测试集有95个样本，然后对分层结果进行双重检查。

## 3 模型训练与评估
在这一部分，我们将选择适合此问题并在scikit-learn中可用的3个监督学习模型。首先，根据对数据的了解和每个模型的优缺点讨论选择这三个模型的原因。然后，拟合不同大小的训练数据（100个数据点，200个数据点和300个数据点），并测量F 1分数。需要生成三个表（每个模型一个），用来显示训练集的大小、训练时间、预测时间、训练集的F 1分数，和测试集的F 1分数。

scikit-learn中有以下监督学习方法可供选择： 
* 高斯朴素贝叶斯（GaussianNB）
* 决策树
* 组合方法（Bagging，AdaBoost，Random Forest，Gradient Boosting）
* K-近邻（KNeighbors）
* 随机梯度下降（SGDC）
* 支持向量机（SMV）
* 逻辑回归

#### 问题2 – 模型应用
列出适合此问题的三个监督学习模型。对于所选择的每个模型
* 描述一下在可应用该模型的行业中的真实应用。（给出参考文献）
* 模型的优点是什么？何时表现良好？
* 模型的缺点是什么？何时表现不好？
* 根据你对数据的了解，是什么使该模型成为这个问题的良好备选项？
如何选择算法？
 
![github](https://github.com/JCancy/Student-Intervention-System/tree/master/picture/choose algorithm.PNG)  
 

#### 答案：
* 我们将选择3个监督学习模型。
	1. 朴素贝叶斯
	2. 逻辑回归
	3. 支持向量机
* 但在我们继续介绍3种监督学习模式之前，我们将讨论数据本身，因为它对于确定该模型是否成为当前问题的良好备选项是一个重要的因素。
##### 数据概述
###### 1. 分类偏斜：
* 可以看到，与未能毕业的学生相比，成功毕业的学生人数几乎是其两倍。
	* 成功毕业学生人数：265（多数类）
	* 未能毕业学生人数：130人（少数类）
* 这在我们分割数据时会产生问题。
	* 训练集可以用多数类聚集，测试集可以用多数类聚集。这将影响计算的精度。
	* 因此，应该强调如何拆分数据以及要选择的指标。
		* 数据分割：分层K折
		* 指标选择：准确率和召回率
###### 2. 数据缺失：
* 数据集中没有示例。
	* 395名学生
* 这将对需要更多数据的一些算法产生影响。
	* 一般来说，我们需要更多的数据，除非我们面临高偏差问题。
	* 在这种情况下，我们应该保持简单的算法。
###### 3. 特征过多：
* 对于这样一个有395名学生的小数据库，我们的特征数量惊人。
	* 48个特征
* 维度诅咒
	* 由于维度诅咒，对于添加的每个额外特征，我们需要增加示例的数量。
##### 模型说明
###### 1. 朴素贝叶斯
* 行业应用（Barbosa等，2014）
	* 鸡蛋分类
	* 这是朴素贝叶斯作为学习算法繁荣一个有趣的应用，将鸡蛋分为2组：
		1. 自由摆放的鸡蛋
			* 这些鸡蛋是能够自由滚动的鸡蛋
		2. 层架式摆放的鸡蛋
			* 这些鸡蛋是保存在一个小笼子里的鸡蛋。有些人可能称之为“unethical eggs”
	* 研究显示，在2组鸡蛋之间进行分类时，朴素贝叶斯提供了90％的高精度。
* 优点
	1. 由于比逻辑回归之类的判别模型收敛更快，因此需要更少的数据
* 缺点
	1. 要求观察值之间相互独立
		* 但实际上，即使违反了独立性假设，分类器也表现良好
	2. 表示简单，没有超参数调整的机会
* 问题的适合性
	* 该算法对此问题表现良好，因为数据具有以下属性：
			* 观察值数量少
			* 朴素的贝叶斯在小数据集上表现良好
###### 2. 逻辑回归
* 行业应用（Penderson等，2014）
	* 蛋白质序列分类
		* 识别并自动将蛋白质序列分为11个预定义类别之一
		* 逻辑回归的使用在未来生物信息学的应用中有巨大潜力
* 优点
	1. 有许多方法用来规范模型以容忍错误并避免过度拟合
	2. 与朴素贝叶斯不同，我们不用担心有相关性的特征
	3. 与支持向量机不同，我们可以使用在线梯度下降法轻松获取新数据
* 缺点
	1.需要观察值是相互独立的
	2.其目的是基于自变量进行预测，如果没有正确识别，逻辑回归几乎没有预测价值
* 问题的适合性
	* 许多特征可能是相关的
		* 与朴素贝叶斯不同，逻辑回归可以处理这个问题
		* 由于数据集具有许多特征，正则化以防止过度拟合
###### 3. 支持向量机（SVM）
* 行业应用（Di Pillo等，2016）：
	* 促销活动的销售预测
		* 最初使用的统计方法是如ARIMA和指数平滑这样的平滑方法 
		* 但如果出现销售数据的高无规律性，可能会失败
		* 因此SVM提供了一个很好的选择
* 优点
	1. SVM能正则化参数以容忍一些错误并避免过度拟合
	2. 内核技巧：用户可以通过设计内核来嵌入有关该问题的专业知识
	3. 如果选择参数C和γ，则样本推广效果好
		* 换句话说，即使训练样本有一些偏差，SVM也会更稳健
* 缺点
	1. 不好解释：支持向量机是黑箱
	2. 高计算成本：SVM的训练时间呈指数级增长
	3. 用户需要具有一定的专业知识才能使用内核函数
* 问题的适合性
	* 许多特征可能是相关的
		* 由于数据集具有许多特征，正则化以防止过度拟合
###### 参考文献
1.	Barbosa, R. M., Nacano, L. R., Freitas, R., Batista, B. L. and Barbosa, F. (2014), The Use of Decision Trees and Naïve Bayes Algorithms and Trace Element Patterns for Controlling the Authenticity of Free-Range-Pastured Hens’ Eggs. Journal of Food Science, 79: C1672–C1677. http://dx.doi.org/10.1111/1750-3841.12577.
2.	Pedersen, B. P., Ifrim, G., Liboriussen, P., Axelsen, K. B., Palmgren, M. G., Nissen, P., . . . Pedersen, C. N. S. (2014). Large scale identification and categorization of protein sequences using structured logistic regression. PloS One, 9(1), 1. http://dx.doi.org/10.1371/journal.pone.0085139
3.	Di Pillo, G., Latorre, V., Lucidi, S. et al. 4OR-Q J Oper Res (2016) 14: 309. http://dx.doi.org/10.1007/s10288-016-0316-0
##### 设置
运行下面的代码以初始化三个辅助函数，可以使用它们来训练和测试上面选择的三个监督学习模型。函数如下：
* train_classifier - 输入分类器和训练数据，并将分类器与数据进行匹配。
* predict_labels- 将合适的分类器、特征和目标标签作为输入，并使用F 1分数进行预测。
* train_predict- 输入分类器、训练数据和测试数据，并展示train_clasifier和predict_labels结果。
	* 此函数将分别汇报训练数据和测试数据的F 1分数。
 
In [14]:
```python
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print ""
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
 ```
 
其中，train_classifier函数用分类器拟合训练数据，prediction_labels函数使用基于F1分数的拟合分类器进行预测，train_predict函数使用基于F1分类器进行分类和预测。

### 3.1 模型效果指标
使用上面的预定义函数，导入你选择的三个监督学习模型，并为每个学习模型运行train_predict函数。记住，你需要对每个分类器进行训练和预测，以获得三种不同大小的训练集：100、200和300。因此，你应该为每个模型使用不同的大小的训练集得出以下9个不同输出。在下面的代码中，需要实现以下内容：
* 导入你在上部分中选择的三个监督学习模型。
* 初始化三种模型，并把它们存储在clf_A、clf_B和clf_C中。
	* 如果提供random_state，为每个模型使用此函数。
	* **注意：**为每个模型使用默认设置 - 在稍后的部分中你会调整一个特定的模型。
* 创建用于训练每个模型的不同训练集大小。
	* 不要重组并重新分配数据！新的训练点应来自X_train和y_train。
* 将每个模型与每个训练集大小相匹配，并对测试集进行预测（总共9个）。
  **注意：**以下代码之后提供了可以用于存储结果的三个表。

 
In [15]:
```python
# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = LogisticRegression(random_state=42)
clf_C = SVC(random_state=42)

# TODO: Set up the training set sizes
X_train_100 = X_train.iloc[:100, :]
y_train_100 = y_train.iloc[:100]

X_train_200 = X_train.iloc[:200, :]
y_train_200 = y_train.iloc[:200]

X_train_300 = X_train.iloc[:300, :]
y_train_300 = y_train.iloc[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

for clf in [clf_A, clf_B, clf_C]:
    print "\n{}: \n".format(clf.__class__.__name__)
    for n in [100, 200, 300]:
        train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)
```
GaussianNB:

Training a GaussianNB using a training set size of 100. . .
Trained model in 0.0012 seconds
Made predictions in 0.0003 seconds.
F1 score for training set: 0.7752.
Made predictions in 0.0003 seconds.
F1 score for test set: 0.6457.

Training a GaussianNB using a training set size of 200. . .
Trained model in 0.0026 seconds
Made predictions in 0.0004 seconds.
F1 score for training set: 0.8060.
Made predictions in 0.0002 seconds.
F1 score for test set: 0.7218.

Training a GaussianNB using a training set size of 300. . .
Trained model in 0.0008 seconds
Made predictions in 0.0004 seconds.
F1 score for training set: 0.8134.
Made predictions in 0.0003 seconds.
F1 score for test set: 0.7761.


LogisticRegression:

Training a LogisticRegression using a training set size of 100. . .
Trained model in 0.0020 seconds
Made predictions in 0.0002 seconds.
F1 score for training set: 0.8671.
Made predictions in 0.0001 seconds.
F1 score for test set: 0.7068.

Training a LogisticRegression using a training set size of 200. . .
Trained model in 0.0018 seconds
Made predictions in 0.0002 seconds.
F1 score for training set: 0.8211.
Made predictions in 0.0001 seconds.
F1 score for test set: 0.7391.

Training a LogisticRegression using a training set size of 300. . .
Trained model in 0.0034 seconds
Made predictions in 0.0003 seconds.
F1 score for training set: 0.8512.
Made predictions in 0.0002 seconds.
F1 score for test set: 0.7500.


SVC:

Training a SVC using a training set size of 100. . .
Trained model in 0.0011 seconds
Made predictions in 0.0008 seconds.
F1 score for training set: 0.8354.
Made predictions in 0.0007 seconds.
F1 score for test set: 0.8025.

Training a SVC using a training set size of 200. . .
Trained model in 0.0043 seconds
Made predictions in 0.0021 seconds.
F1 score for training set: 0.8431.
Made predictions in 0.0011 seconds.
F1 score for test set: 0.8105.

Training a SVC using a training set size of 300. . .
Trained model in 0.0066 seconds
Made predictions in 0.0066 seconds.
F1 score for training set: 0.8664.
Made predictions in 0.0019 seconds.
F1 score for test set: 0.8052.
 
 
从输出结果可以看出，不同算法、不同数据集大小所得的结果不同，为了进行对比，我们将结果存放到表格中。

### 3.2 表格结果
在Markdown中设计表格，编辑下面的单元格，可以在表中记录上述结果。
 
![classifer1](https://github.com/JCancy/Student-Intervention-System/tree/master/picture/classifer1.png) 

![逻辑回归](https://github.com/JCancy/Student-Intervention-System/tree/master/picture/classifer2.png "逻辑回归") 

![](https://github.com/JCancy/Student-Intervention-System/tree/master/picture/classifer3.png) 
 
从三张表中，可以很清晰的看出三种不同算法对应三种不同大小数据集所用的训练及预测的时间和F1分数，经过对比和分析，我们可以选则一个最合适的模型。

## 4 选择最佳模型
在最后一部分中，我们将从三个监督学习模型中选择用于学生数据的最佳模型。然后，对整个训练集（X_train和y_train）的模型进行网格搜索优化，通过调整至少一个参数来改善未经训练模型的F 1分数。

#### 问题3 – 选择最佳模型
根据之前的实验，用一到两段说明你所选择的最佳模型。基于已有的数据、有限的资源、代价和表现，哪种模式是最合适的？

#### 答案：
SVM的预测性能比朴素贝斯略好，但需要注意的是，数据增加时，SVM的计算时间将会比朴素贝叶斯所需时间增加的快得多，当我们学生数量更多时，代价将会呈指数级增长。另一方面，朴素贝叶斯的计算时间随着数据的增加而线性增长，代价不会快速上升。因此，考虑到其在一个小数据集和潜在的大型和不断增长的数据集上的表现，朴素贝叶斯是SVM一个很好的替代。

再比较朴素贝叶斯和逻辑回归。虽然结果显示，逻辑回归在预测效果方面比朴素贝叶斯稍差，但与朴素贝叶斯相比，逻辑回归模型的轻微调整很容易产生更好的预测效果，而朴素贝叶则不能对模型进行调整。因此，我们应该选择回归。

3个算法的大O符号
1. 朴素贝叶斯：
	* O(n)
2. 逻辑回归：
	* O(C^n)
3. 支持向量机：
	* O(n^3) 关于S型核
	* O(n^2) 关于空间复杂性

#### 问题4 – 模型的外行解释
用1-2段内容，向外行人士解释，最终选择的模型是如何起作用的。确保你描述的是模型的主要内容，例如模型如何训练以及模型如何进行预测。避免使用数学或技术术语，例如描述方程或讨论算法实现。

#### 答案： 
首先，模型学习学生的表现指标对学生是否毕业的影响。由于我们已经有已毕业和没有毕业的学生的现有数据，所以该模型可以做到这一点。根据学生的表现指标，该模型将输出每个表现指标的权重。

考虑到这一点，第二步是预测新生是否毕业。这一次，我们没有关于现有学生的信息，他们是否毕业或还在读书。但是，我们有一个从以前批次毕业学生中学习到的模型。新增学生的表现指标及其权重将被输入模型，模型将产生概率，根据“可能毕业”或“不太可能毕业”将学生进行分类。然后，我们可以对不太可能毕业的学生采取预防措施。

此外，如果我们想要结果更加安全可靠，以确保我们发现尽可能多的学生，我们可以提升“可能毕业”的严格程度，将一部分学生判定为“可能不毕业”，即使他们可能“能毕业”，来确定他们毕业的可能性并找到更多学生。这是因为将“可能毕业”的学生标记为“不太可能毕业”没有任何害处，但是如果将“不太可能毕业”的学生判定为“有可能毕业”而不进行干预的学生，可能会导致严重的后果。

### 4.1 模型调整（逻辑回归）
微调所选模型。使用网格搜索（GridSearchCV），其中至少有一个重要参数并调用至少3个不同的值，对整个训练集使用这一方法。在下面的代码中，需要实现以下内容：
* 导入sklearn.grid_search.gridSearchCV和sklearn.metrics.make_scorer。
* 为所选模型创建一个你希望调整的参数的字典。
	* 示例：parameters = {'parameter' : [list of values]}。
* 初始化已选择的分类器并将其存储在clf中。
* 用make_scorer创建F 1评分函数并存储在f1_scorer中。
	* 将pos_label参数设置为正确值！
* 使用f1_scorer评分方法在分类器clf上执行网格搜索，并将其存储在grid_obj中。
* 将网格搜索对象与训练数据（X_train，y_train）进行拟合，并将其存储在grid_obj中。

**特别提示：**
* 可以使用分层重组进行数据分割，它保留了每个类的样本百分比，并将其与交叉验证相结合。当数据集对两个目标标签之一有较强不平衡性时，这会非常有用。
* http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
 
 
In [16]:
```python
# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
```

In [17]:
```python
# Create the parameters list you wish to tune
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['sag']
max_iter = [1000]
param_grid = dict(C=C, solver=solver, max_iter=max_iter)

# Initialize the classifier
clf = LogisticRegression(random_state=42)

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Stratified Shuffle Split
ssscv = StratifiedShuffleSplit(y_train, n_iter=10, test_size=0.1)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, param_grid, cv=ssscv, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
```
Made predictions in 0.0005 seconds.
Tuned model has a training F1 score of 0.8040.
Made predictions in 0.0004 seconds.
Tuned model has a testing F1 score of 0.8050.
 
根据输出结果，我们得到训练F1分数为0.8040，测试F1分数为0.8050。

#### 问题5 – 最终的F1分数
经训练与测试后的最终模型的F1分数是什么？该分数与未经训练的模型相比如何？

#### 答案：
* 最终模型的F1分数为：
	* 训练F1分数：0.8040
	* 测试F1分数：0.8050
* 与未调整的模型相比，调整后模型的测试F1得分有所增加
	* 目前高于朴素贝叶斯F1分数


