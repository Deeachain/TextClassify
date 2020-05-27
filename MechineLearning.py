from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import jieba


'''数据预处理将文本转换为TF-IDF向量'''
vectorizer = TfidfVectorizer()

with open('data/o2o/train.csv', 'r') as f:
    next(f)
    lines = f.readlines()
x, y = [], []
for line in lines:
    x.append(line.split('\t')[1].strip())
    y.append(line.split('\t')[0].strip())

jieba.enable_parallel(8)
train_x, val_x = x[:int(len(x)*0.85)], x[int(len(x)*0.85):]
sent_words = [list(jieba.cut(sentence)) for sentence in train_x]
train_x = [" ".join(word) for word in sent_words]
train_x = vectorizer.fit_transform(train_x)
sent_words = [list(jieba.cut(sentence)) for sentence in val_x]
val_x = [" ".join(word) for word in sent_words]
val_x = vectorizer.transform(val_x)

# y_onehot = dict(zip(list(set(y)), range(len(list(set(y))))))
# for i, temp in enumerate(y):
#     y[i] = y_onehot[temp]
y = np.array(y)
train_y, val_y = y[:int(len(y)*0.85)], y[int(len(y)*0.85):]


'''sub数据集处理'''
with open('data/o2o/test_new.csv', 'r') as f:
    next(f)
    lines = f.readlines()
x, id = [], []
for line in lines:
    x.append(line.split(',')[1].strip())
    id.append(line.split(',')[0].strip())

jieba.enable_parallel(8)
sub_x = x
sent_words = [list(jieba.cut(sentence)) for sentence in sub_x]
sub_x = [" ".join(word) for word in sent_words]
sub_x = vectorizer.transform(sub_x)

'''朴素贝叶斯'''
from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB(alpha=0.1)
bayes.fit(train_x, train_y)
pred = bayes.predict(val_x)
print('Bayes F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('Bayes Acc Score is: ', accuracy_score(val_y, pred))

sub_y = bayes.predict(sub_x)
with open('sub/bayes_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''SVM'''
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(train_x, train_y)
pred = svm.predict(val_x)
print('SVM F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('SVM Acc Score is: ', accuracy_score(val_y, pred))

sub_y = svm.predict(sub_x)
with open('sub/svm_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''DecisionTree'''
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(train_x, train_y)
pred = dt.predict(val_x)
print('DecisionTree F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('DecisionTree Acc Score is: ', accuracy_score(val_y, pred))

sub_y = dt.predict(sub_x)
with open('sub/dt_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''RandomForest'''
from sklearn.ensemble import RandomForestClassifier

rft = RandomForestClassifier(n_estimators=150)
rft.fit(train_x, train_y)
pred = rft.predict(val_x)
print('RandomForest F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('RandomForest Acc Score is: ', accuracy_score(val_y, pred))

sub_y = rft.predict(sub_x)
with open('sub/rft_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''ADBoosting'''
from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier()
adb.fit(train_x, train_y)
pred = adb.predict(val_x)
print('ADBoosting F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('ADBoosting Acc Score is: ', accuracy_score(val_y, pred))

sub_y = adb.predict(sub_x)
with open('sub/adb_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''GBDT'''
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier()
gbdt.fit(train_x, train_y)
pred = gbdt.predict(val_x)
print('gbdt F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('gbdt Acc Score is: ', accuracy_score(val_y, pred))

sub_y = gbdt.predict(sub_x)
with open('sub/gbdt_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))


'''XGBoosting'''
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(train_x, train_y)
pred = xgb.predict(val_x)
print('XGBoosting F1 Score is: ', f1_score(val_y, pred, average='macro'))
print('XGBoosting Acc Score is: ', accuracy_score(val_y, pred))

sub_y = xgb.predict(sub_x)
with open('sub/xgb_sub.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(sub_y)):
        f.write('{},{}\n'.format(id[i], sub_y[i]))

