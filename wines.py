# Formed Using ML as well as DL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
red = pd.read_csv("red wines.csv",sep=';')
print(red.head())
white = pd.read_csv("white wines.csv",sep=';')
print(white.head())
red['type'] =1
white['type'] =0
wines = pd.concat([red,white],ignore_index=True)
wines.dropna(inplace=True)
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].hist(wines[wines['type']==1].alcohol,bins=10,facecolor = 'red',alpha = 0.5,label = 'Red wine')
ax[1].hist(wines[wines['type']==0].alcohol,bins=10,facecolor = 'white',edgecolor = 'black',lw=0.5,alpha = 0.5,label='white wine')

for a in ax:
    a.set_ylim([0,1000])
    a.set_xlabel('Alcohol in % vol')
    a.set_ylabel('Frequency')
ax[0].set_title('Alcohol content in Red Wine')
ax[1].set_title('Alcohol Content in White Wine')
fig.suptitle('Distribution Of Alcohol by Winw Type')
plt.tight_layout()
plt.show()

x = wines.iloc[:,:-1]
y = wines['type']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.34,random_state=45)
print(x_train)
print(y_train)


model =Sequential()
model.add(Dense(12,activation='relu',input_dim =12))
model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3,batch_size=1,verbose=1)

y_pred = model.predict(x_test)
y_pred_labels = (y_pred >= 0.5).astype(int)
for prediction in y_pred_labels[:12]:
    wine_type = "Red wine" if prediction==1 else "White Wine"
    print(f"Prediction:{wine_type}")
