import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression

"""CurrencyRegression.py: Attempts 5 different methods of linear and logistic regression
to predict future values of currency exchange rates based on current and past values.
These models are to be examined in order to determine the possibility of their
 effective application in real world currency trading"""

__author__ = "David Heintz"

def logistic_regression(x1, y1, s1, x2, y2, s2):
    log_reg = LogisticRegression()
    i = 0
    l_train = []
    while i < len(s1):
        if y1[i] > s1[i]:
            l_train.append(1)
        else:
            l_train.append(0)
        i += 1
    log_reg.fit(x1,l_train)
    i = 0
    l_test = []
    while i < len(s2):
        if y2[i] > s2[i]:
            l_test.append(1)
        else:
            l_test.append(0)
        i += 1
    print("Logistic: {:.2f}".format(log_reg.score(x2,l_test)))
    l_pred = log_reg.predict(x2)
    i = 0
    right, wrong, avg_r, avg_w = 0, 0, 0, 0
    while i < len(l_pred):
        if l_pred[i] == l_test[i]:
            right += 1
            avg_r += abs(y2[i] - s2[i])
        else:
            wrong += 1
            avg_w += abs(y2[i] - s2[i])
        i += 1
    print("Correct v Incorrect Magnitude: " + str(avg_r - avg_w))
    print(confusion_matrix(l_test, l_pred))
    return log_reg.score(x2,l_test)

def hundred_split(country):
    i = 0
    arr = []
    while i < len(country) - 200:
        first = country[i]
        second = country[i + 49]
        third = country[i + 99]
        fourth = country[i + 199]
        arr.append([first,second,third,fourth])
        i += 1
    return arr

def make_bimonthly(country,dates):
    prev = 0
    start = 0
    m_count = 0
    first, second, third, last, n_end = 0, 0, 0, 0, 0
    nf, ns, nt, l = 0, 0, 0, 0
    arr = []
    i = 0
    while i < dates.size:
        date = dates[i]
        date = date.split('-')
        if(date[1] != prev):
            if(start > 1):
                arr.append([first, second, third, last, n_end])
            else:
                start += 1
            first = nf/5
            second = ns/5
            third = nt/5
            last = l
            nf = country[i]
            ns, nt = 0, 0
            m_count = 1
            prev = date[1]
        else: #date[1] == prev
            if m_count < 5:
                 nf += country[i]
            elif m_count < 10:
                ns += country[i]
            elif m_count < 15:
                nt += country[i]
                l = country[i]
            else:
                n_end = country[i]
            m_count += 1
        i += 1
    arr.append([first, second, third, last, n_end])
    return arr

def make_monthly(country, dates):
    prev = 0
    m_count = 0
    first, second, start, last = 0, 0, 0, 0
    arr = []
    i = 0
    while i < dates.size:
        date = dates[i]
        date = date.split('-')
        if(date[1] != prev):
            if(prev != 0):
                arr.append([first/5, second/5 , start, last])
            first = country[i]
            second = 0
            m_count = 1
            prev = date[1]
        else: #date[1] == prev
            if m_count < 5:
                first += country[i]
            elif m_count < 10:
                second += country[i]
                start = country[i]
            else:
                last = country[i]
            m_count += 1
        i += 1
    arr.append([first/5, second/5, start, last])
    return arr

def run_regression(country, dates, name):

    #region Removing ND
    i = 0
    nds = []
    while i < country.size:
        if country[i] == 'ND':
            nds.append(i)
        i += 1
    new_country = np.delete(country, nds)
    new_dates = np.delete(dates, nds)
    #endregion

    #region Creating and Splitting Datasets
    arr1 = make_monthly(new_country,new_dates)
    arr2 = make_bimonthly(new_country,new_dates)
    arr3 = hundred_split(new_country)
    print(len(arr3))
    train1, test1 = train_test_split(arr1, test_size = 120, shuffle = True)
    train2, test2 = train_test_split(arr2, test_size = 119, shuffle = True)
    train3, test3 = train_test_split(arr3, test_size = 2000, shuffle = True)
    #endregion

    #region Monthly Linear Regression
    lin_reg1 = LinearRegression()
    t = np.array(train1)
    x1, y1 = np.hsplit(t,2)
    s1, y1 = np.hsplit(y1,2)

    lin_reg1.fit(x1,y1)
    m_acc = lin_reg1.score(x1,y1)

    t = np.array(test1)
    x2, y2 = np.hsplit(t,2)
    s2, y2 = np.hsplit(y2,2)

    predictions = lin_reg1.predict(x2)
    acc = lin_reg1.score(x2,y2)
    print(name)
    print("Monthly Linear: {:.2f}".format(acc))

    plt.scatter(s2,y2)
    plt.title(name + "Monthly Value Comparison")
    plt.xlabel('10th Day Value')
    plt.ylabel('Final Day Value')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    plt.scatter(y2, predictions)
    plt.title(name + "Test 1 (Monthly)")
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100])
    plt.show()
    #endregion

    #region Bimonthly Linear Regression
    lin_reg2 = LinearRegression()
    t = np.array(train2)

    a, b, c, s1, y1 = np.hsplit(t,5)
    x1 = np.hstack((a, b, c))
    lin_reg2.fit(x1,y1)
    blin_acc = lin_reg2.score(x1,y1)

    t = np.array(test2)
    a, b, c, s2, y2 = np.hsplit(t,5)
    x2 = np.hstack((a, b, c))

    predictions = lin_reg2.predict(x2)
    acc = lin_reg2.score(x2,y2)
    print("Bimonthly Linear:  {:.2f}".format(acc))

    plt.scatter(y2, predictions)
    plt.title(name + "Test 2 (Bimonthly)")
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-1000, 1000], [-1000, 1000])
    plt.show()

    plt.scatter(s2,y2)
    plt.title(name + "Bimonthly Value Comparison")
    plt.xlabel('15th Day Value')
    plt.ylabel('Next Month Final Day Value')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-1000, 1000], [-1000, 1000])
    plt.show()
    #endregion

    #region Bimonthly Logistic Regression
    blog_acc = logistic_regression(x1,y1,s1,x2,y2,s2)
    #endregion

    #region 200 Day Linear Regression
    lin_reg3 = LinearRegression()
    t = np.array(train3)

    x1, y1 = np.hsplit(t,2)
    s1, y1 = np.hsplit(y1,2)
    x1 = np.hstack((x1,s1))
    lin_reg3.fit(x1,y1)
    dlin_acc = lin_reg3.score(x1,y1)

    t = np.array(test3)
    x2, y2 = np.hsplit(t,2)
    s2, y2 = np.hsplit(y2,2)
    x2 = np.hstack((x2,s2))

    predictions = lin_reg3.predict(x2)
    acc = lin_reg3.score(x2,y2)
    predictions = lin_reg3.predict(x2)
    acc = lin_reg3.score(x2,y2)
    print("200 Days Linear:  {:.2f}".format(acc))

    plt.scatter(y2, predictions)
    plt.title(name + "Test 3 (200 Days)")
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-1000, 1000], [-1000, 1000])
    plt.show()

    plt.scatter(s2,y2)
    plt.title(name + "200 Day Value Comparison")
    plt.xlabel('100th Day Value')
    plt.ylabel('200th Day Value')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-1000, 1000], [-1000, 1000])
    plt.show()
    #endregion

    #region 200 Days Logistic Regression
    dlog_acc = logistic_regression(x1,y1,s1,x2,y2,s2)
    #endregion

    return m_acc, blin_acc, blog_acc, dlin_acc, dlog_acc


#region Loading Dataset
data = pd.read_excel('Foreign_Exchange_Rates.xlsx')
euro = data['EURO AREA - EURO/US$']
euro = euro.to_numpy()
yen = data["JAPAN - YEN/US$"]
yen = yen.to_numpy()
yuan = data["CHINA - YUAN/US$"]
yuan = yuan.to_numpy()
rup = data["INDIA - INDIAN RUPEE/US$"]
rup = rup.to_numpy()
aus = data["AUSTRALIA - AUSTRALIAN DOLLAR/US$"]
aus = aus.to_numpy()
dates = data['Time Serie']
dates = dates.to_numpy()
#endregion

#region Running Regression
em, eb_lin, eb_log, ed_lin, ed_log = run_regression(euro,dates,"EURO/US$")
jm, jb_lin, jb_log, jd_lin, jd_log = run_regression(yen,dates,"YEN/US$")
cm, cb_lin, cb_log, cd_lin, cd_log = run_regression(yuan,dates,"YUAN/US$")
im, ib_lin, ib_log, id_lin, id_log = run_regression(rup,dates,"RUPEE/US$")
am, ab_lin, ab_log, ad_lin, ad_log = run_regression(aus,dates,"AUS$/US$")
#endregion

#region Monthly Model Bar Graph
labels = ['EURO', 'YEN', 'YUAN', 'RUPEE', 'AUS']
acc = [em ,jm ,cm , im, am]

x = np.arange(len(labels))
width = .60

fig, ax = plt.subplots()
ax.bar(x, acc, width, color=['red', 'green', 'orange', 'purple', 'yellow'])

ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of Monthly Model by Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
#endregion

#region Bimonthly Linear Regression Bar Graph
labels = ['EURO', 'YEN', 'YUAN', 'RUPEE', 'AUS']
acc = [eb_lin ,jb_lin ,cb_lin , ib_lin, ab_lin]

x = np.arange(len(labels))
width = .60

fig, ax = plt.subplots()
ax.bar(x, acc, width, color=['red', 'green', 'orange', 'purple', 'yellow'])

ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of Bimonthly Linear Model by Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
#endregion

#region Bimonthly Logistic Model Bar Graph
labels = ['EURO', 'YEN', 'YUAN', 'RUPEE', 'AUS']
acc = [eb_log ,jb_log ,cb_log , ib_log, ab_log]

x = np.arange(len(labels))
width = .60

fig, ax = plt.subplots()
ax.bar(x, acc, width, color=['red', 'green', 'orange', 'purple', 'yellow'])

ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of Bimonthly Logistic Model by Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
#endregion

#region 200 Day Period Linear Model Bar Graph
labels = ['EURO', 'YEN', 'YUAN', 'RUPEE', 'AUS']
acc = [ed_lin ,jd_lin ,cd_lin , id_lin, ad_lin]

x = np.arange(len(labels))
width = .60

fig, ax = plt.subplots()
ax.bar(x, acc, width, color=['red', 'green', 'orange', 'purple', 'yellow'])

ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of 200 Day Period Linear Model by Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
#endregion

#region 200 Day Period Logistic Model Bar Graph
labels = ['EURO', 'YEN', 'YUAN', 'RUPEE', 'AUS']
acc = [ed_log ,jd_log ,cd_log , id_log, ad_log]

x = np.arange(len(labels))
width = .60

fig, ax = plt.subplots()
ax.bar(x, acc, width, color=['red', 'green', 'orange', 'purple', 'yellow'])

ax.set_ylabel('Accuracy')
ax.set_title('Test Accuracy of 200 Day Period Logistic Model by Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
#endregion