#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:/Users/HP/Downloads/Analytics_loan_collection_dataset_cs.csv")


# In[3]:


df.head(1)


# In[4]:


print(df.shape)
df["Target"].value_counts()


# # Data exploration

# In[5]:


# Distribution of data along different categorical variables
print(df["Location"].value_counts())
print(df["EmploymentStatus"].value_counts())
print(df["LoanType"].value_counts())


# In[6]:


# distribution of customers who missed the payment along different categorical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
df.groupby("Location")["Target"].sum().plot(
    kind="bar", ax=axes[0], color="blue", title="By Location")
df.groupby("EmploymentStatus")["Target"].sum().plot(
    kind="bar", ax=axes[1], color="blue", title="By Employment Status")
df.groupby("LoanType")["Target"].sum().plot(
    kind="bar", ax=axes[2], color="blue", title="By Loan Type")
plt.tight_layout()
plt.show()


# In[7]:


df.columns


# In[8]:


bins = [20, 40, 60, 100]
labels = ["g1", "g2", "g3"]
grouped = df.groupby(pd.cut(df["Age"], bins=bins, labels=labels))["Target"].agg(["count", "sum"])
grouped.plot(kind="bar", figsize=(8,5), color=["skyblue", "orange"])
plt.title("Age Group: Count vs Sum(Target)")
plt.xlabel("Age Group")
plt.ylabel("Value")
plt.legend(["Total customer", "Missed payment"])
plt.show()


# In[9]:


q1 = df["Income"].quantile(0.25)
q2 = df["Income"].quantile(0.50)
q3 = df["Income"].quantile(0.75)
bins = [df["Income"].min()-1, q1, q2, q3, df["Income"].max()]
labels = ["Q1", "Q2", "Q3", "Q4"]
income_groups = pd.cut(df["Income"], bins=bins, labels=labels)
grouped = df.groupby(income_groups)["Target"].agg(["count", "sum"])

grouped.plot(kind="bar", figsize=(8,5), color=["skyblue", "orange"])
plt.title("Target Count and Sum by Income Quartiles")
plt.xlabel("Income Quartile")
plt.ylabel("Value")
plt.legend(["Total Customer", "Missed payment"])
plt.show()


# In[10]:


df["LoanAmountRatio"]= df["LoanAmount"]/df["Income"]
bins = [0, 0.5, 1, 2, 10]
labels = ["G1", "G2", "G3", "G4"]
grouped = df.groupby(pd.cut(df["LoanAmountRatio"], bins=bins, labels=labels))["Target"].agg(["count", "sum"])
grouped.plot(kind="bar", figsize=(8,5), color=["skyblue", "orange"])
plt.title("Loan amount ratio Group: Count vs Sum(Target)")
plt.xlabel("Ratio Group")
plt.ylabel("Value")
plt.legend(["Total customer", "Missed payment"])
plt.show()


# In[11]:


df.groupby("MissedPayments")["Target"].agg(["count", "sum"]).plot(kind="bar", figsize=(6,4), color=["blue","orange"])


# In[12]:


df["Location"] = df["Location"].replace({"Urban":1, "Rural":0, "Suburban":2})
df["EmploymentStatus"] = df["EmploymentStatus"].replace({"Salaried":1, "Unemployed":0, "Student":2,"Self-Employed":3})
df["LoanType"] = df["LoanType"].replace({"Auto":1, "Education":0, "Business":2,"Personal":3, "Home":4})


# In[13]:


X = df.drop(columns=["CustomerID","Target"])
y = df["Target"]

tree_model = DecisionTreeClassifier(
    max_depth=3,       
    random_state=42
)
tree_model.fit(X, y)

plt.figure(figsize=(14, 8))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True
)
plt.show()


# In[2]:


print((578+98)/(578+98+67+222))
print((578+98)/984)


# In[ ]:





# In[14]:


## Model Building
df.dtypes


# In[15]:


X = df.drop(columns=["Target", "CustomerID"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,  random_state=42,
                          use_label_encoder=False,eval_metric="logloss"  
)


# In[16]:


xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)


# In[17]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[18]:


import matplotlib.pyplot as plt

importances = xgb_model.feature_importances_

feat_importance = pd.Series(importances, index=X.columns)

feat_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color="skyblue")
plt.title("XGBoost Feature Importance")
plt.show()


# In[ ]:





# In[19]:


y_pred_prob = xgb_model.predict_proba(X_test)


# In[20]:


prediction = y_pred.tolist()

X_test["prediction"] = prediction
X_test["Target"] = y_test


# In[21]:


lst = y_pred_prob.tolist()
prob_list = []
for i in lst:
    prob_list.append(i[1])
print(len(prob_list))


# In[22]:


X_test["prediction_prob"] = prob_list


# ## Chatbot to resolve issues

# # please refer to below link for customized chatbot
# https://chatbot-xs5tuqsvab63mnpaq39qmh.streamlit.app/

# In[23]:


df["InteractionAttempts"].value_counts()


# In[24]:


df["SentimentScore_ch"] = df["SentimentScore"].apply(lambda x: 0 if x >= 0.3 else (1 if -0.3 <= x < 0.3 else 2))
df["SentimentScore_ch"].value_counts()


# In[25]:


# InteractionAttempts 0 to 9
# Complaints 0 to 4
# PartialPayments 0 to 6
df["PartialPayments"].value_counts()


# In[26]:


def assign_persona(row):
    if row["SentimentScore"] > 0.25 and row["PartialPayments"] >= 2:
        return "Cooperative"
    elif row["SentimentScore"] > 0.25 and row["PartialPayments"] == 1:
        return "Confused"
    elif row["SentimentScore"] > 0.25 and row["PartialPayments"] < 1:
        return "Evasive"
    elif row["SentimentScore"] < 0.25 and row["PartialPayments"] >= 2:
        return "Confused"
    elif row["SentimentScore"] < 0.25 and row["PartialPayments"] == 1:
        return "Evasive"
    else:
        return "Aggressive" 

df["Persona"] = df.apply(assign_persona, axis=1)

print(df["Persona"].value_counts())


# In[29]:


print(df.columns.tolist())


# In[ ]:


empathetic, assertive, informative


# In[ ]:





# In[ ]:





# In[ ]:




