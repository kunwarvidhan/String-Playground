import pandas as pd
from keras.utils import to_categorical
import numpy
c=pd.read_csv("../input/emnist-balanced-train.csv")
d=pd.read_csv("../input/emnist-balanced-test.csv")
X = c.iloc[0:100000,1:].values
Y = c.iloc[0:100000,0].values
X1 = d.iloc[0:3000,1:].values
Y1 = d.iloc[0:3000,0].values



x_train =X.reshape((X.shape[0], int(numpy.sqrt(X.shape[1]) + 0.5), int(numpy.sqrt(X.shape[1]) + 0.5)))
x_test=X1.reshape((X1.shape[0], int(numpy.sqrt(X1.shape[1]) + 0.5), int(numpy.sqrt(X1.shape[1]) + 0.5)))
print(x_train.shape,x_test.shape)


import numpy as np # linear algebra
import pandas as pd 

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

xshow = X1.reshape(-1,28,28)/255
yshow = Y1
import matplotlib.pyplot as plt

import seaborn
r = 4
c = 6
fig=plt.figure(figsize=(r, c),dpi=100)
for i in range(1, r*c+1):
    img = xshow[i]
    ax = fig.add_subplot(r, c, i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_text(yshow[i])
    
    plt.imshow(img,cmap='gray')
plt.show()

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.utils import to_categorical
padding_size = 2
window_size = 5
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
print(x_train.shape,x_test.shape)
onehot_y_train = to_categorical(Y)
onehot_y_test = to_categorical(Y1)
print(onehot_y_train.shape,onehot_y_test.shape)
nn_model = Sequential()  # create model
nn_model.add(Conv2D(32, kernel_size=(window_size, window_size), strides=(1, 1), activation='relu', input_shape=x_train[0].shape))
nn_model.add(Conv2D(32, (window_size, window_size), activation='relu'))
nn_model.add(MaxPooling2D(pool_size=(padding_size, padding_size), strides=(2, 2)))
nn_model.add(Conv2D(64, (window_size, window_size), activation='relu'))
nn_model.add(Conv2D(64, (window_size, window_size), activation='relu'))
nn_model.add(MaxPooling2D(pool_size=(padding_size, padding_size)))
nn_model.add(Flatten())
nn_model.add(Dense(5000, activation='relu'))
nn_model.add(Dense(5000, activation='relu'))
nn_model.add(Dense(2500, activation='relu'))
nn_model.add(Dense(1000, activation='relu'))
nn_model.add(Dense(500, activation='relu'))
nn_model.add(Dense(onehot_y_train.shape[1], activation='softmax'))
nn_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.005), metrics=['accuracy'])
print("model formed")
s1=np.arange(25)
sy=np.arange(25)

for i in range(0,25):
    nn_model.fit(x_train, onehot_y_train, epochs=2, verbose=1, batch_size=256, initial_epoch=0)
    score = nn_model.evaluate(x_test, onehot_y_test, verbose=1)
    s1[i]=score[1]*100
    
print("model trained")

plt.plot(s1,sy)



string=[]
string1=[]
Dict={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",33:"X",34:"Y",35:"Z",36:"a",37:"b",38:"d",39:"e",40:"f",41:"g",42:"h",43:"n",44:"q",45:"r",46:"t"}
#nn_model.load_weights("mnistmodel.h5")
for i in range (50):
    img = x_test[i]
    test_img = img.reshape((1,28,28,1))
    img_class = nn_model.predict_classes(test_img)
    prediction = img_class[0]
    classname = img_class[0]
    #print("Class: ",classname)
    img = img.reshape((28,28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()
    string.append(Dict[classname])
print(string)
for i in range (51,101):
    img = x_test[i]
    test_img = img.reshape((1,28,28,1))
    img_class = nn_model.predict_classes(test_img)
    prediction = img_class[0]
    classname = img_class[0]
    #print("Class: ",classname)
    img = img.reshape((28,28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()
    string1.append(Dict[classname])
print(string1)
stra=""
stra=stra.join(string)
strb=""
strb=strb.join(string1)
print(stra)
print(strb)
lena=len(stra)
lenb=len(strb)



#longest common subsequebce
def lcs(u, v):
   
    c = [[-1]*(len(v) + 1) for _ in range(len(u) + 1)]
    lcs_helper(u, v, c, 0, 0)
    return c
 
 
def lcs_helper(u, v, c, i, j):
    
    if c[i][j] >= 0:
        return c[i][j]
 
    if i == len(u) or j == len(v):
        q = 0
    else:
        if u[i] == v[j]:
            q = 1 + lcs_helper(u, v, c, i + 1, j + 1)
        else:
            q = max(lcs_helper(u, v, c, i + 1, j),
                    lcs_helper(u, v, c, i, j + 1))
    c[i][j] = q
    return q
 
 
def print_lcs(u, v, c):
    
    i = j = 0
    while not (i == len(u) or j == len(v)):
        if u[i] == v[j]:
            print(u[i], end='')
            i += 1
            j += 1
        elif c[i][j + 1] > c[i + 1][j]:
            j += 1
        else:
            i += 1




# longest common substring
from difflib import SequenceMatcher 
  
def longestSubstring(str1,str2): 
  
   
    seqMatch = SequenceMatcher(None,str1,str2) 
  
    
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2)) 
  
  
    if (match.size!=0): 
          print (str1[match.a: match.a + match.size])  
    else: 
          print ('No longest common sub-string found') 



#Knuth Morris Prat (KMP) String Matching Algorithm
def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)
    lps = [0]*M
    j = 0 
    computeLPSArray(pat, M, lps)
 
    i = 0 
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
 
        if j == M:
            print("Found pattern at index " + str(i-j))
            j = lps[j-1]
        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
 
def computeLPSArray(pat, M, lps):
    len = 0 
 
    lps[0] 
    i = 1
 
    
    while i < M:
        if pat[i]==pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len != 0:
                len = lps[len-1]
            else:
                lps[i] = 0
                i += 1
choice=12
print("String A formed:",stra)
print("String B formed:",strb)
while(choice!=0):
    print("\n\t\t\tSTRING PLAYGROUND\nThe following program recognises the images from the EMNIST dataset and catagorises them,\nThen from the testing dataset it recognises the characters and forms a string of length 50 then applies the following functions.\nEnter:\n1 for Morris Prat (KMP) String Matching Algorithm\n2 for Longest Common Subsequence \n3 for Longest Common Substring\n0 to exit")
    print("-------------------------------------------------")
    choice = int(input(""))
    if (choice==1):
        txt = input("Enter text you want to search: ")
        pat = int(input("Enter the string NO. in which you want to search "))
        if(pat==1):
            s=stra
        else:
            s=strb
        KMPSearch(txt,s)
    if (choice==2):
        m=lcs(stra,strb)
        print(stra)
        print(strb)
        print_lcs(stra,strb,m)
    if (choice==3):
        longestSubstring(stra,strb)


