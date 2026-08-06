

import joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import os 

import kagglehub
#load latest version of the dataset

path=kagglehub.dataset_download('dgomonov/new-york-city-airbnb-open-data')
print('path of dataset:',path)

df=pd.read_csv(os.path.join(path,'AB_NYC_2019.csv'))
# print(df.head())


# eplaore data 
# print(df.info())
# print(df.describe())

# finding missing values
missing_values=df.isnull().sum()
print(missing_values[missing_values>0])

print(df['room_type'].unique())

sns.countplot(x='room_type',data=df)

numerical_cols=['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']

df[numerical_cols].hist(bins=30,figsize=(15,10))
# plt.show()

#univariate analysis : it is 
sns.countplot(x='neighbourhood_group',data=df)



#bivariate analysis
sns.boxplot(x='room_type',y='price',data=df)


#correlation analysis: it is used to find the relationship between the numerical variables
corr= df[numerical_cols + ['latitude','longitude']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
# plt.show()

#Geographic distribution : it is used to find the distribution of the data on the map
sns.scatterplot(x='longitude',y='latitude',hue='room_type',data=df,alpha=0.4,s=10)
# plt.show()



#data cleaning: it is used to remove the outliers and missing values from the data
# making a copy of the original dataframe
df_cleaned=df.copy()
df_cleaned=df.drop(columns=['id','name','host_name','last_review','host_id'])

# no reviews yet -> 0 reviews per month , not missingvalues
df_cleaned['reviews_per_month']=df_cleaned['reviews_per_month'].fillna(0)


#cap extreme outlier instead of removing them
price_cap=df_cleaned['price'].quantile(0.99)
nights_cap=df_cleaned['minimum_nights'].quantile(0.99)
df_cleaned['price']=df_cleaned['price'].clip(upper=price_cap)
df_cleaned['minimum_nights']=df_cleaned['minimum_nights'].clip(upper=nights_cap)

#df_cleaned=df_cleaned[df_cleaned['price']<price_cap]
#df_cleaned=df_cleaned[df_cleaned['minimum_nights']<nights_cap]

x=df_cleaned.drop(columns=['room_type'],axis=1)
y=df_cleaned['room_type']


#train test split
# stratified sampling: it is used to split the data into train and test set in such a way that the distribution
#  of the target variable is same in both the sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42,stratify=y)


#preprocessing: it is used to scale the numerical variables and encode the categorical variables
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

numerical_cols=['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']
categorical_cols=['neighbourhood_group','neighbourhood']


#1. pipline for numerical columns
numeric_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    # ('power_transformer',PowerTransformer(method='yeo-johnson',standardize=True)),
    ('scaler',StandardScaler()),
])


#2. pipeline for categorical columns
categorical_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore')),
])

#3. combine both pipelines using ColumnTransformer
preprocessor=ColumnTransformer(transformers=[
    ('numerical', numeric_pipeline, numerical_cols),
    ('Categorical', categorical_pipeline, categorical_cols)
    ])




#SMOTE : IT IS used to balance the dataset by oversampling the minority class
#SOMTE: STANDS FOR SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE
#1. HIGH chance of overfitting
#2.Data leakeage 


# instead of SMOTE we use class_weight='balanced' in the model to balance the dataset
# class_weight='balanced' : it is used to balance the dataset by giving more weight to the minority class

###########################################################3
#trying different alogorithms to find the best model for the data

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_validate


#class weight='balanced' : it is used to balance the dataset by giving more weight to the minority class
#but gradient boosting does not have class_weight parameter so we will use SMOTE for gradient boosting

models={
    'Logistic Regression':LogisticRegression(class_weight='balanced',random_state=42),
    'Decision Tree':DecisionTreeClassifier(class_weight='balanced',random_state=42),        
    'Random Forest':RandomForestClassifier(class_weight='balanced',random_state=42),
    'Gradient Boosting':GradientBoostingClassifier(random_state=42)
    }


for name,model in models.items():
    print('Training model:',name)

    #creating a pipeline for each model
    pipeline=Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('classifier',model)
    ])

    cross_val=cross_validate(pipeline,x_train,y_train,cv=5,scoring=['accuracy','precision_macro','recall_macro'
                                                                    ,'f1_macro'],return_train_score=True)

    print(f'{name}: Train Accuracy: {cross_val["train_accuracy"].mean():.4f}, Test Accuracy: {cross_val["test_accuracy"].mean():.4f}')





#hyperparameter tuning: it is used to find the best parameters for the model
from sklearn.model_selection import RandomizedSearchCV
best_pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',RandomForestClassifier(class_weight='balanced',random_state=42))
])

parameter_distribution={
    'classifier__n_estimators':[100,200,300,400,500],
    'classifier__max_depth':[None,5,10,15,20],
    'classifier__min_samples_split':[2,5,10],
}

search=RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=parameter_distribution,
    n_iter=10,
    cv=5,
    scoring='f1_macro',
    random_state=42)


search.fit(x_train,y_train)

print('Best parameters:',search.best_params_)
print('Best score:',search.best_score_)



##################################
#FINAL EVALUATION ON THE TEST SET 

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
best_pipeline=search.best_estimator_
y_pred=best_pipeline.predict(x_test)

print(f"accuracy score{accuracy_score(y_test,y_pred)}" )
print(f"F1_Score {f1_score(y_test,y_pred, average='macro')}")

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues',xticklabels=best_pipeline.classes_,
            yticklabels=best_pipeline.classes_)

plt.show()



###########################################################
#saving the final model artifact 
# joblib: it is used to save the model artifact in a file so that we can use it later for prediction.


import joblib

joblib.dump(best_pipeline,'house_type_classifier.pkl',compress=9)

print('Model artifact saved as house_type_classifier.pkl')

