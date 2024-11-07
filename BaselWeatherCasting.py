#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
weather = pd.read_csv("dataexport_20241106T135737.csv")


# In[5]:


weather


# In[7]:


weather.apply(pd.isnull).sum()


# In[29]:


core_weather = weather[["location","Basel","Basel.1","Basel.2","Basel.3","Basel.4","Basel.5","Basel.6","Basel.7"]].copy()


# In[31]:


core_weather.columns = ["date_loc","temp_max","temp_min","temp_mean","precip","wind_max","wind_min","wind_mean","sun_duration"]


# In[33]:


core_weather


# In[35]:


core_weather = core_weather.iloc[9:]


# In[37]:


core_weather


# In[39]:


core_weather = core_weather.reset_index(drop = True)


# In[43]:


core_weather = core_weather.set_index("date_loc")


# In[45]:


core_weather


# In[47]:


core_weather = core_weather.rename_axis("DATE")


# In[49]:


core_weather


# In[51]:


core_weather = core_weather.iloc[:-8]


# In[53]:


core_weather


# In[55]:


core_weather.apply(pd.isnull).sum()


# In[57]:


core_weather.dtypes


# In[59]:


core_weather = core_weather.astype('float64')


# In[61]:


core_weather


# In[63]:


core_weather.dtypes


# In[65]:


core_weather.index


# In[67]:


core_weather.index = pd.to_datetime(core_weather.index, format='%Y%m%dT%H%M')


# In[69]:


core_weather.index


# In[71]:


core_weather


# In[73]:


core_weather.apply(lambda x: (x==9999).sum())


# In[75]:


core_weather[["temp_max","temp_min"]].plot()


# In[77]:


core_weather.index.year.value_counts()


# In[81]:


core_weather["precip"].plot()


# In[83]:


core_weather["wind_mean"].plot()


# In[85]:


core_weather["sun_duration"].plot()


# In[87]:


core_weather.groupby(core_weather.index.year).sum()


# In[89]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[91]:


core_weather


# In[93]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[95]:


core_weather


# In[99]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alpha_values = [0.01, 0.1, 1, 10, 100, 1000]

ridge = Ridge()

param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error') 
X = core_weather[['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_max', 'wind_min', 'wind_mean', 'sun_duration']]
y = core_weather['target']
grid_search.fit(X, y)  

best_alpha = grid_search.best_params_['alpha']
best_score = grid_search.best_score_

print(f"Best alpha: {best_alpha}")
print(f"Best cross-validated score: {best_score}")


# In[101]:


reg = Ridge(alpha=10)


# In[103]:


predictors = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_max', 'wind_min', 'wind_mean', 'sun_duration']


# In[105]:


train = core_weather.loc[:"2015-12-31"]
test = core_weather.loc["2016-01-01":]


# In[107]:


reg.fit(train[predictors], train["target"])


# In[109]:


predictions = reg.predict(test[predictors])


# In[111]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(test["target"], predictions)


# In[113]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)


# In[115]:


combined


# In[117]:


combined.columns = ["actual", "predictions"]


# In[119]:


combined


# In[121]:


combined.plot()


# In[123]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()


# In[125]:


core_weather


# In[133]:


core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]


# In[135]:


core_weather


# In[141]:


core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[143]:


core_weather


# In[145]:


predictors = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_max', 'wind_min', 'wind_mean', 'sun_duration',"month_max","month_day_max","max_min"]


# In[147]:


core_weather = core_weather.iloc[30:,:].copy()


# In[153]:


core_weather


# In[155]:


core_weather.apply(pd.isnull).sum()


# In[157]:


import numpy as np
core_weather[predictors].applymap(np.isinf).sum()


# In[159]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[161]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2015-12-31"]
    test = core_weather.loc["2016-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return  error, combined


# In[163]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[165]:


error


# In[167]:


combined.plot()


# In[169]:


core_weather["monthly_avg"] = core_weather.groupby(core_weather.index.month)["temp_max"].transform(lambda x: x.expanding().mean())


# In[171]:


core_weather


# In[173]:


error


# In[175]:


predictors = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_max', 'wind_min', 'wind_mean', 'sun_duration',"month_max","month_day_max","max_min","monthly_avg"]


# In[177]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[179]:


error


# In[181]:


combined.plot()


# In[183]:


combined["actual"].plot()


# In[185]:


combined["predictions"].plot()


# In[ ]:




