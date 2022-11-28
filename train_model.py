from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#Importing Libraries to create GUI
from tkinter import *

#Importing Libraries to perform calculations
import numpy as np
import pandas as pd
import os
import io

targetUser1 = ['Babies', 'Children', 'Adults']
productType1 = ['Hair', 'Body', 'Face', 'Scalp', 'Mouth']
condition1 = ['Psoriasis', 'Dry_Skin', 'Normal_Skin', 'Itchy_Skin', 'Sensitive_Skin', 'Inflammed_Skin', 'Infected_Skin', 'Dry_Scalp', 'Itchy_Scalp', 'Sensitive_Scalp', 'Oily_Scalp', 'Red_Scalp', 'Flaky_Scalp', 'Wounds', 'Burns', 'Blisters', 'Cuts', 'Red_Skin', 'Delicate_Skin', 'Sun_Rays_Protection']
allCondition = ['Babies', 'Children', 'Adults', 'Hair', 'Body', 'Face', 'Scalp', 'Mouth', 'Psoriasis', 'Dry_Skin', 'Normal_Skin', 'Itchy_Skin', 'Sensitive_Skin', 'Inflammed_Skin', 'Infected_Skin', 'Dry_Scalp', 'Itchy_Scalp', 'Sensitive_Scalp', 'Oily_Scalp', 'Red_Scalp', 'Flaky_Scalp', 'Wounds', 'Burns', 'Blisters', 'Cuts', 'Red_Skin', 'Delicate_Skin', 'Sun_Rays_Protection']
product = ['Pso-Rest Cream', 'Daily Intensive Moisturising Cream', 'Daily Advance Intensive Barrier Repair Cream', 'Gentle Hair Shampoo', 'Calming Body Wash', 'Scalp Repair Spray', 'Daily Resurging Face Serum', 'Hydrating Anti-Photoaging Sunscreen', 'Calming Baby Balm', 'Cooling Snow Cream', 'Moinsturising Cleanser', 'Hydrating Anti-Bacterial Body Wash', 'Facial Cleanser', 'Revitalising Life Essence Mist', 'Lightweight Moisture Booster', 'Aurora Silver Skin Spray', "R'Cares E Squalance Hydrating Oil", 'EZ Anti-Dandruff Shampoo', 'EZ Clean Body Wash']
root = Tk()
Condition1 = StringVar()
Condition2 = StringVar()
Condition3 = StringVar()
Condition4 = StringVar()
Condition5 = StringVar()
Condition6 = StringVar()
Condition7 = StringVar()
Condition1.set("Select Here")
Condition2.set("Select Here")
Condition3.set("Select Here")
Condition4.set("Select Here")
Condition5.set("Select Here")
Condition6.set("Select Here")
Condition7.set("Select Here")

pred1 = StringVar()

condition2 = []
for i in range(0, len(allCondition)):
      condition2.append(0)

df = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv')
DF = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv', index_col='Product')

df.replace({'Product':{'Pso-Rest Cream':0, 'Daily Intensive Moisturising Cream': 1, 'Daily Advance Intensive Barrier Repair Cream':2, 'Gentle Hair Shampoo':3, 
                                   'Calming Body Wash':4, 'Scalp Repair Spray':5, 'Daily Resurging Face Serum':6, 'Hydrating Anti-Photoaging Sunscreen':7, 
                                   'Calming Baby Balm':8, 'Cooling Snow Cream':9, 'Moinsturising Cleanser':10, 'Hydrating Anti-Bacterial Body Wash':11, 
                                   'Facial Cleanser':12, 'Revitalising Life Essence Mist':13, 'Lightweight Moisture Booster':14, 'Aurora Silver Skin Spray':15, 
                                   "R'Cares E Squalance Hydrating Oil":16, 'EZ Anti-Dandruff Shampoo':17, 'EZ Clean Body Wash':18}}, inplace=True)

#printing the top 5 rows of the training dataset

X= df[allCondition]
y = df[["Product"]]
np.ravel(y)
# print(X)
# print(y)

tr = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv')

tr.replace({'Product':{'Pso-Rest Cream':0, 'Daily Intensive Moisturising Cream': 1, 'Daily Advance Intensive Barrier Repair Cream':2, 'Gentle Hair Shampoo':3, 
                                   'Calming Body Wash':4, 'Scalp Repair Spray':5, 'Daily Resurging Face Serum':6, 'Hydrating Anti-Photoaging Sunscreen':7, 
                                   'Calming Baby Balm':8, 'Cooling Snow Cream':9, 'Moinsturising Cleanser':10, 'Hydrating Anti-Bacterial Body Wash':11, 
                                   'Facial Cleanser':12, 'Revitalising Life Essence Mist':13, 'Lightweight Moisture Booster':14, 'Aurora Silver Skin Spray':15, 
                                   "R'Cares E Squalance Hydrating Oil":16, 'EZ Anti-Dandruff Shampoo':17, 'EZ Clean Body Wash':18}}, inplace=True)
tr.head()

X_test= tr[allCondition]
y_test = tr[["Product"]]
np.ravel(y_test)
# print(X_test)
# print(y_test)

def DecisionTree():
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X,y)

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    y_pred=clf3.predict(X_test)
    print("Decision Tree")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    print("Confusion matrix")
    conf_matrix=confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    pconditions = [Condition1.get(),Condition2.get(),Condition3.get(),Condition4.get(),Condition5.get(),Condition6.get(),Condition7.get()]

    for k in range(0,len(allCondition)):
        for z in pconditions:
            if(z==allCondition[k]):
                condition2[k]=1

    inputtest = [condition2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(product)):
        if(predicted == a):
            h='yes'
            break
    
    if (h=='yes'):
        pred1.set(" ")
        pred1.set(product[a])
        result = product[a]
        print(result)
    else:
        pred1.set(" ")
        pred1.set("Not Found")

DecisionTree()



# #Importing required libraries
# from sklearn.preprocessing import StandardScaler

# #Importing Libraries to create GUI
# from tkinter import *

# #Importing Libraries to perform calculations
# import numpy as np
# import pandas as pd
# import pickle

# allCondition = ['Psoriasis', 'Dry_Skin', 'Normal_Skin', 'Itchy_Skin', 'Sensitive_Skin', 'Inflammed_Skin', 'Infected_Skin', 'Dry_Scalp', 'Itchy_Scalp', 'Sensitive_Scalp', 'Oily_Scalp', 'Red_Scalp', 'Flaky_Scalp', 'Wounds', 'Burns', 'Blisters', 'Cuts', 'Red_Skin', 'Delicate_Skin', 'Sun_Rays_Protection', 'Hair', 'Body', 'Face', 'Scalp', 'Mouth', 'Babies', 'Children', 'Adults']
# product = ['Pso-Rest Cream', 'Daily Intensive Moisturising Cream', 'Daily Advance Intensive Barrier Repair Cream', 'Gentle Hair Shampoo', 'Calming Body Wash', 'Scalp Repair Spray', 'Daily Resurging Face Serum', 'Hydrating Anti-Photoaging Sunscreen', 'Calming Baby Balm', 'Cooling Snow Cream', 'Moinsturising Cleanser', 'Hydrating Anti-Bacterial Body Wash', 'Facial Cleanser', 'Revitalising Life Essence Mist', 'Lightweight Moisture Booster', 'Aurora Silver Skin Spray', "R'Cares E Squalance Hydrating Oil", 'EZ Anti-Dandruff Shampoo', 'EZ Clean Body Wash']
# result = " "
# root = Tk()
# Condition1 = StringVar()
# Condition2 = StringVar()
# Condition3 = StringVar()
# Condition4 = StringVar()
# Condition5 = StringVar()
# Condition6 = StringVar()
# Condition7 = StringVar()
# pred1 = StringVar()

# condition2 = []
# for i in range(0, len(allCondition)):
#       condition2.append(0);

# #Loading data
# df = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv')
# DF = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv', index_col='Product')

# df.replace({'Product':{'Pso-Rest Cream':0, 'Daily Intensive Moisturising Cream': 1, 'Daily Advance Intensive Barrier Repair Cream':2, 'Gentle Hair Shampoo':3, 
#                                    'Calming Body Wash':4, 'Scalp Repair Spray':5, 'Daily Resurging Face Serum':6, 'Hydrating Anti-Photoaging Sunscreen':7, 
#                                    'Calming Baby Balm':8, 'Cooling Snow Cream':9, 'Moinsturising Cleanser':10, 'Hydrating Anti-Bacterial Body Wash':11, 
#                                    'Facial Cleanser':12, 'Revitalising Life Essence Mist':13, 'Lightweight Moisture Booster':14, 'Aurora Silver Skin Spray':15, 
#                                    "R'Cares E Squalance Hydrating Oil":16, 'EZ Anti-Dandruff Shampoo':17, 'EZ Clean Body Wash':18}}, inplace=True)

# #Feature selection
# X= df[allCondition]
# y = df[["Product"]]
# np.ravel(y)

# tr = pd.read_csv('C:/Users/USER/Downloads/TestData (3).csv')

# tr.replace({'Product':{'Pso-Rest Cream':0, 'Daily Intensive Moisturising Cream': 1, 'Daily Advance Intensive Barrier Repair Cream':2, 'Gentle Hair Shampoo':3, 
#                                    'Calming Body Wash':4, 'Scalp Repair Spray':5, 'Daily Resurging Face Serum':6, 'Hydrating Anti-Photoaging Sunscreen':7, 
#                                    'Calming Baby Balm':8, 'Cooling Snow Cream':9, 'Moinsturising Cleanser':10, 'Hydrating Anti-Bacterial Body Wash':11, 
#                                    'Facial Cleanser':12, 'Revitalising Life Essence Mist':13, 'Lightweight Moisture Booster':14, 'Aurora Silver Skin Spray':15, 
#                                    "R'Cares E Squalance Hydrating Oil":16, 'EZ Anti-Dandruff Shampoo':17, 'EZ Clean Body Wash':18}}, inplace=True)
# tr.head()

# X_test= tr[allCondition]
# y_test = tr[["Product"]]
# np.ravel(y_test)
    
# #Build decision tree model
# from sklearn import tree
# clf3 = tree.DecisionTreeClassifier()
# clf3 = clf3.fit(X,y)

# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# y_pred=clf3.predict(X_test)
    
# #Evaluate model
# print("Decision Tree")
# print("Accuracy")
# print(accuracy_score(y_test, y_pred))
# print(accuracy_score(y_test, y_pred,normalize=False))
# print("Confusion matrix")
# conf_matrix=confusion_matrix(y_test,y_pred)
# print(conf_matrix)

# pconditions = [Condition1.get(),Condition2.get(),Condition3.get(),Condition4.get(),Condition5.get(),Condition6.get(),Condition7.get()]

# for k in range(0,len(allCondition)):
#     for z in pconditions:
#         if(z==allCondition[k]):
#             condition2[k]=1

# inputtest = [condition2]
# predict = clf3.predict(inputtest)
# predicted=predict[0]

# h='no'
# for a in range(0,len(product)):
#     if(predicted == a):
#         h='yes'
#         break
    
# if (h=='yes'):
#     pred1.set(" ")
#     pred1.set(product[a])
#     result = product[a]
#     print(result)
#     #pickle.dump(result, open("model.pkl", "wb"))
# else:
#     pred1.set(" ")
#     pred1.set("Not Found")

# # # check our current directory to make sure it saved
# # !ls

# #pickle.dump(clf3, open("model.pkl", "wb"))
# pickle.dump(clf3, open("model.pkl", "wb"))
# #pickle_result = pickle.load(open("model.pkl", "rb"))
