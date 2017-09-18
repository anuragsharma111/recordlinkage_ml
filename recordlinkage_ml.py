
# coding: utf-8

# In[ ]:


#      Python Libraries 


import pandas as pd
import numpy as np
import recordlinkage as rl
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()

# Data Preprocessing and loading

pd.set_option('display.max_rows', 1200)
pd.set_option('display.max_columns', 200)

# Data Loading

parse_dates = ['DataSides.0.Settle Date', 'DataSides.1.Settle Date','DataSides.0.Trade Date','DataSides.1.Trade Date']
data = pd.read_excel("/home/anshul/record_linkage_api/Reconcilliation Data Mastering Using AI.xlsx",parse_dates=parse_dates, header=1)


m = pd.DataFrame({'pa': range(1, len(data) + 1, 1)})
data_source_0 = data.iloc[:,:46]
data_source_0 = data_source_0.join(m)
data_source_1 = data.iloc[:, 46:92]
data_source_1 = data_source_1.join(m)


# making new feature(Column) by adding cancel amount and net amount column

data_source_0['DataSides.0.Cancel Amount'] = data_source_0['DataSides.0.Cancel Amount'].replace(np.nan, 0)
data_source_0['Cancel_and_net_amount_addition'] = data_source_0['DataSides.0.Cancel Amount'] + data_source_0['DataSides.0.Net Amount']
data_source_1['DataSides.1.Cancel Amount'] = data_source_1['DataSides.1.Cancel Amount'].replace(np.nan, 0)
data_source_1['Cancel_and_net_amount_addition'] = data_source_1['DataSides.1.Cancel Amount'] + data_source_1['DataSides.1.Net Amount']

# making new feature(Column) by difference of settle date and trade date

data_source_0['diff_date'] = data_source_0['DataSides.0.Settle Date'].sub(data_source_0['DataSides.0.Trade Date'], axis=0)
data_source_0['diff_date'] = data_source_0['diff_date'] / np.timedelta64(1, 'D')
data_source_1['diff_date'] = data_source_1['DataSides.1.Settle Date'].sub(data_source_1['DataSides.1.Trade Date'], axis=0)
data_source_1['diff_date'] = data_source_1['diff_date'] / np.timedelta64(1, 'D')

# Preparing Data for training and testing 

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\     for true label          \\\\\\\\\\\\\\\\\\\\\\    
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\                              \\\\\\\\\\\\\\\\\\\\\\\   

source_0 = data_source_0
source_1 = data_source_1
pcl=rl.BlockIndex(on='pa')                     # Block Indexing 
pairs = pcl.index(data_source_0,data_source_1)
true_index = pairs                              # true matching record pairs

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\     for false label         \\\\\\\\\\\\\\\\\\\\\\    
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\                              \\\\\\\\\\\\\\\\\\\\\\\  

data_A = data_source_0[:900]        #Selecting the top 300 records
data_B = data_source_1[:900]

# data_A = data_source_0[758:760]       #Selecting the top 300 records
# data_B = data_source_1[758:760]


# data_A['DataSides.0.Transaction Type'] = data_A['DataSides.0.Transaction Type'].str.encode('utf-8')
# data_B['DataSides.1.Transaction Type'] = data_B['DataSides.1.Transaction Type'].str.encode('utf-8')

pcl=rl.FullIndex()                 # Full Indexing for the false matching records
pairs = pcl.index(data_A,data_B)   # multi index pairs 

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\     Record linkage implementation for matching score  \\\\\\\\\\\\\\\\\\\\\\    
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\                                                      \\\\\\\\\\\\\\\\\\\\\\\ 

compare_cl = rl.Compare(pairs, data_A, data_B) # compare vector  

# compairing score on exact matching

compare_cl.exact('DataSides.0.Currency', 'DataSides.1.Currency', name='Currency',missing_value=np.nan)   
compare_cl.exact('DataSides.0.Mapped Custodian Account', 'DataSides.1.Mapped Custodian Account', name='Mapped Custodian',missing_value=np.nan)
compare_cl.exact('DataSides.0.SEDOL', 'DataSides.1.SEDOL', name='SEDOL',missing_value=1)
compare_cl.exact('DataSides.0.CUSIP', 'DataSides.1.CUSIP', name='CUSIP',missing_value=1)
compare_cl.exact('DataSides.0.ISIN', 'DataSides.1.ISIN', name='ISIN',missing_value=1)
compare_cl.exact('DataSides.0.Underlying ISIN', 'DataSides.1.Underlying ISIN', name='ISIN_u',missing_value=1)
compare_cl.exact('DataSides.0.Underlying Sedol', 'DataSides.1.Underlying Sedol', name='SEDOL_u',missing_value=1)
compare_cl.exact('DataSides.0.Underlying Cusip', 'DataSides.1.Underlying Cusip', name='CUSIP_u',missing_value=1)
compare_cl.exact('DataSides.0.Cancel Amount', 'DataSides.1.Cancel Amount', name='Cancel Amount',missing_value=np.nan)
compare_cl.exact('Cancel_and_net_amount_addition', 'Cancel_and_net_amount_addition', name='add_amount',missing_value=np.nan)
compare_cl.exact('diff_date', 'diff_date', name='diff_date',missing_value=np.nan)

# compairing score on date matching

compare_cl.date('DataSides.0.Trade Date', 'DataSides.1.Trade Date', name='Trade Date',missing_value=np.nan)
compare_cl.date('DataSides.0.Settle Date', 'DataSides.1.Settle Date', name='Settle Date',missing_value=np.nan)
compare_cl.date('DataSides.0.Business Date', 'DataSides.1.Business Date', name='Business Date',missing_value=np.nan)

# compairing score on string matching

compare_cl.string('DataSides.0.Transaction Type', 'DataSides.1.Transaction Type',method='jarowinkler', name='Transaction Type',missing_value=np.nan)
compare_cl.string('DataSides.0.Investment Type', 'DataSides.1.Investment Type', method='jarowinkler', name='Investment type',missing_value=np.nan)



compare_score_data = compare_cl.vectors     # dataframe of matching score
compare_score_data = compare_score_data.fillna(method='ffill')   # null treatment

# making of true and false status record

true_label_data = compare_score_data.loc[(compare_score_data.index.get_level_values(1) ) == (compare_score_data.index.get_level_values(0) )]  #filtering match records         
false_label_data_cross = compare_score_data.loc[(compare_score_data.index.get_level_values(1) ) != (compare_score_data.index.get_level_values(0) )]  #filtering not match records
false_label_data = false_label_data_cross.sample(frac=0.0011)  # selecting records from cross joined records
true_label_data['Actual_match_status'] = True     # labeling for the true records
false_label_data['Actual_match_status'] = False    # labeling for the false records

final_labeled_data = true_label_data.append(false_label_data) # the final datafrme with both true and false labeled records
final_labeled_data_index = final_labeled_data.index    # index for the final dataset 


# splitting data in 70 and 30 percent ratio

msk = np.random.rand(len(final_labeled_data)) < 0.70   # randomly splitting data

train = final_labeled_data[msk]  # trainig data
test = final_labeled_data[~msk]  # testing data

train_X = train.iloc[:,0:16]     # trainig data input columns
train_Y = train.iloc[:,16:17]    # trainig data output column

test_X = test.iloc[:,0:16]       # testing data input columns
test_Y = test.iloc[:,16:17]      # testing data output column

# Model training

clf = MLPClassifier(hidden_layer_sizes=(10, ))   # multilayer perceptron classifier with the 10 hidden layer
clf.fit(train_X,train_Y)                         # Training model for prediction

# Prediction and result

predicted = clf.predict(test_X)  # prediction on the testing data
print("Accuracy score :", accuracy_score(test_Y, predicted))   # Accuracy of the prediction

tn, fp, fn, tp = confusion_matrix(test_Y, predicted).ravel()
print( "True negative:= ",tn)
print ("false positive:= ",fp)
print ("false negative:= ",fn)
print ("True positive:= ",tp)

# Plots

# Record In testing dataset  
import matplotlib.pyplot as plt; plt.rcdefaults()

objects = ('True', 'False')
y_pos = np.arange(len(objects))
performance = [(fn+tp),(tn+fp)]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
for a,b in zip(y_pos,performance):
    plt.text(a, b, str(b))
plt.xticks(y_pos, objects)
plt.ylabel('Number of Record')
plt.title('True and False record in testing dataset')
 
plt.show()

# Predicted True and False record by model

import matplotlib.pyplot as plt; plt.rcdefaults()

objects = ('True', 'False')
y_pos = np.arange(len(objects))
performance = [(tp+fp),(tn+fn)]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
for a,b in zip(y_pos,performance):
    plt.text(a, b, str(b))
plt.xticks(y_pos, objects)
plt.ylabel('Number of Record')
plt.title('Predicted True and False record by model')
 
plt.show()

# Summary of prediction


objects = ('True record and predicted True', 'False record and predicted true ','False record and predicted false ','True record and predicted False')
y_pos = np.arange(len(objects))
performance = [tp,fp,tn,fn]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
for a,b in zip(y_pos,performance):
    plt.text(a, b, str(b))
plt.xticks(y_pos, objects,rotation=70)
plt.ylabel('Number of Record')
plt.title('Summary of prediction')
 
plt.show()

# Saving Prediction with match status

test_index = test_X.index.values.tolist()
match_status = pd.DataFrame(predicted,columns=['Pridicted match status'])

list_source_0 = [i[0] for i in test_index]
list_source_1 = [i[1] for i in test_index]

data_0 = data_source_0.loc[list_source_0]
data_1 = data_source_1.loc[list_source_1]

data_0.reset_index(inplace=True)
data_1.reset_index(inplace=True)
test_Y.reset_index(inplace=True)

result = pd.concat([data_0, data_1,match_status,test_Y[['Actual_match_status']]],axis=1)

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
result.to_csv('final.csv')
train.to_csv('train.csv')
test.to_csv('test.csv')


