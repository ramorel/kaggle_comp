# Load libraries
import os
os.getcwd()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


from scipy import stats
from scipy.stats import norm, skew #for some statistics

## Import the training and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ids = test['Id'].values
ids.head()
train.head()
test.head()

# Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Transform target
train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# Feature engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['total_baths'] = all_data['FullBath'] + (all_data['HalfBath'] * 0.5) + all_data['BsmtFullBath'] + (all_data['BsmtHalfBath'] * 0.50)
all_data['age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['new'] = np.where(all_data['YearBuilt']==all_data['YrSold'], 1, 0)
all_data['outside_area'] = all_data.loc[:,('WoodDeckSF', 'OpenPorchSF', '3SsnPorch')].sum(axis = 1)

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#from sklearn.preprocessing import LabelEncoder
#cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
#        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
#        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
#        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
#for c in cols:
#    lbl = LabelEncoder()
#    lbl.fit(list(all_data[c].values))
#    all_data[c] = lbl.transform(list(all_data[c].values))

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)

ntrain = train.shape[0]
ntest = test.shape[0]
train = all_data[:ntrain]



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb



#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso.fit(train.values, y_train)
preds = lasso.predict(test)
pd.DataFrame({'Id':ids, 'SalePrice':np.exp(preds)}).to_csv("submission.csv", index = False)

# Visualize the outcome/label variable
prices = pd.DataFrame({'price':all_dat['SalePrice'], 'log_price':np.log(all_dat['SalePrice'])})
prices.hist()

# Normalize the outcome variable
train['SalePrice'] = np.log(train['SalePrice'])

## Create full dataset for data cleaning and wrangling
all_dat = pd.concat((train.iloc[:, 1:], test.iloc[:, 1:]), sort = False)
all_dat = all_dat.drop(['SalePrice'], axis = 1)

all_dat.columns

# Deal with missing data
missing = all_dat.apply(lambda x: x.isnull().sum()) > 0
missing = missing[missing].index
missing_num = all_dat[missing].dtypes[all_dat[missing].dtypes != "object"].index
missing_obj = all_dat[missing].dtypes[all_dat[missing].dtypes == "object"].index

# for categorical variables, NAs typically mean the feature isn't there
all_dat.loc[:, missing_obj] = all_dat.loc[:, missing_obj].fillna("None")

# for numeric variables, first impute lot frontage
plt.scatter(all_dat['LotArea'], all_dat['LotFrontage'])
plt.scatter(all_dat['GrLivArea'], all_dat['LotFrontage'])
X = all_dat.loc[:, ('LotFrontage', 'LotArea', 'GrLivArea')]
# Create linear regression object
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
y = X[:,0]
X = X[:,1:3]
regr = linear_model.LinearRegression()
regr.fit(X, y)
preds = regr.predict(X).flatten()
preds = preds[all_dat['LotFrontage'][all_dat['LotFrontage'].isnull()].index].shape

# Impute predicted values
all_dat.loc[all_dat['LotFrontage'].isnull(), 'LotFrontage'] = preds

# Impute year built for missing garage built
all_dat.loc[all_dat['GarageYrBlt'].isnull(), 'GarageYrBlt'] = all_dat.loc[all_dat['GarageYrBlt'].isnull(), 'YearBuilt']

# Normalize skewed features
numeric_feats = all_dat.dtypes[all_dat.dtypes != "object"].index
train['MSSubClass'].skew()
skewed_cols = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_cols = skewed_cols[skewed_cols > 0.75].index
all_dat[skewed_cols] = np.log1p(all_dat[skewed_cols])

# Impute mean for missing numeric values
all_dat.loc[:, all_dat.columns != 'SalePrice']
all_dat = all_dat.fillna(all_dat.mean())

# One-hot encoding! OW
all_dat = pd.get_dummies(all_dat)

# Training and testing data sets
X_train = all_dat[:train.shape[0]]
X_test = all_dat[train.shape[0]:]
y_train = train['SalePrice']

# Regularized regression
alphas = 10**np.linspace(10,-2,100)*0.5
ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
ridge_cv = [cross_val_score(Ridge(alpha = alpha), X_train, y_train, scoring="neg_mean_squared_error", cv = 5) for alpha in alphas]
ridge_cv = [np.sqrt(-x) for x in ridge_cv]
ridge_cv =   [x.mean() for x in ridge_cv]

ridge_cv = pd.Series(ridge_cv, index = alphas)
ridge_cv.plot(title = "Validation - Just Do It")

ridge_cv.min()


# First, test the training data, since I don't have labels for the test dataset
X_train2,X_test2,y_train2,y_test2=train_test_split(X_train, y_train, test_size=0.3, random_state=31)



# Lasso regression
lasso = Lasso()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
lasso_cv = cross_val_score(model_lasso, X_train, y_train, scoring="neg_mean_squared_error", cv = 5)
lasso_cv = [np.sqrt(-x) for x in lasso_cv]
np.mean(lasso_cv)
sum(model_lasso.coef_ != 0)


from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(X_train2)
pca.n_components_
X_train2 = pca.transform(X_train2)
X_test2 = pca.transform(X_test2)
