import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# reading csv file 
df = pd.read_csv("imdb_top_1000.csv")
df.info()
df = df.dropna(subset=['Certificate'])
df = df.dropna(subset=['Series_Title'])
df = df.dropna(subset=['Runtime'])
df = df.dropna(subset=['Genre'])
df = df.dropna(subset=['IMDB_Rating'])
df = df.dropna(subset=['Meta_score'])
df.loc[df["Certificate"] == "S", "Certificate"] = "R"
df.loc[df["Certificate"] == "Unrated", "Certificate"] = "R"
df.loc[df["Certificate"] == "A", "Certificate"] = "R"
df.loc[df["Certificate"] == "TV-14", "Certificate"] = "PG-13"
df.loc[df["Certificate"] == "TV-PG", "Certificate"] = "PG"
df.loc[df["Certificate"] == "TV-MA", "Certificate"] = "R"
df.loc[df["Certificate"] == "1", "Certificate"] = "R"
df.loc[df["Certificate"] == "UA", "Certificate"] = "PG"
df.loc[df["Certificate"] == "U/A", "Certificate"] = "PG-13"
df.loc[df["Certificate"] == "16", "Certificate"] = "R"
df.loc[df["Certificate"] == "U", "Certificate"] = "G"
df.loc[df["Certificate"] == "GP", "Certificate"] = "PG"
df.loc[df["Certificate"] == "Passed", "Certificate"] = "R"
df.loc[df["Certificate"] == "Approved", "Certificate"] = "R"
     

df = df.drop(columns=["Poster_Link", "Released_Year", "Overview", "Director", "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross"])
df[['Genre1','Genre2', 'Genre3']] = df.Genre.str.split(expand=True)
df['Genre1'] = df['Genre1'].str.replace(',','')
df['Genre2'] = df['Genre2'].str.replace(',','')
df['Genre3'] = df['Genre3'].str.replace(',','')

one_df = pd.get_dummies(df, columns = ["Certificate"])

one_df[['Runtime','fluff']] = one_df.Runtime.str.split(expand=True) 
one_df = one_df.drop(columns = ["Series_Title", "fluff", "Genre"])

one_df["Genre1"][one_df["Genre1"].notnull()] = 1
gen1 = one_df["Genre1"]
gen1.fillna(value = "0", inplace=True)


one_df["Genre2"][one_df["Genre2"].notnull()] = 1
gen2 = one_df["Genre2"]
gen2.fillna(value = "0", inplace=True)


one_df["Genre3"][one_df["Genre3"].notnull()] = 1
gen3 = one_df["Genre3"]
gen3.fillna(value = "0", inplace=True)


ovrScore = []

for idx in one_df.index:
        
    score = 0
    if one_df["Genre1"][idx] == 1 and one_df["Genre2"][idx] == 1 and one_df["Genre3"][idx] == 1:
        score += 10
        
    if one_df["Genre1"][idx] == 1 and one_df["Genre2"][idx] == 1 and one_df["Genre3"][idx] == 0:
        score += 7
        
    if one_df["Genre1"][idx] == 1 and one_df["Genre2"][idx] == 0 and one_df["Genre3"][idx] == 0:
        score += 5
        
    if one_df['Certificate_G'][idx] == 1:
        score += 10
        
    if one_df['Certificate_PG'][idx] == 1:
        score += 20
                
    if one_df['Certificate_PG-13'][idx] == 1:
        score += 30
                
    if one_df['Certificate_R'][idx] == 1:
        score += 40
    ovrScore.append(score)    
            
print(len(ovrScore))
one_df["Overall_score"] = ovrScore
df["Overall_score"] = ovrScore
print(df)

X_train, X_test, y_train, y_test = train_test_split(one_df.iloc[:,:10],one_df.iloc[:,10], random_state = 42,test_size=0.20, shuffle= True)

n = np.arange(1, 16)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
  
for i, k in enumerate(n):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    
print("\nTraining Accuracy for 15-NN:",train_accuracy[14])
print("\nTest Accuracy for 15-NN:",test_accuracy[14])


model = MLPClassifier(hidden_layer_sizes= 50, activation='relu', max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#test = []                        #input from the API to recommend movies
#pred = model.predict(test)       # predicted score
mse = mean_squared_error(y_test,y_pred)
print("\nMetrics for Neural Network model:")
print("(a) \nMean square error:", mse)
print("(b) Root Mean Square Error(RMSE):", math.sqrt(mse))
print("(c) Mean Square Error(MSE):", mse)
print("(d) R^2 score:", r2_score(y_test,y_pred))
print("(e) Mean Absolute Error(MAE):", mean_absolute_error(y_test,y_pred))



plt.plot(n, train_accuracy, label = 'Training Accuracy')
plt.plot(n, test_accuracy, label = 'Testing Accuracy', linestyle= 'dashed')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

           
plt.scatter(y_test, y_pred, color='blue')
plt.title(f'Predicted vs. Actual (MSE: {mse:.2f})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
x = np.linspace(0, 50, 100)
y = x
plt.plot(x, y, linewidth=2.0)
plt.show()

#df["Series_Title", "Runtime"].where(df["Overall_score"] == pred)   #List of recommended movies