# Data Science Salary Prediction

Uta Nishii and Kyllan Wunder


## [dss.kyllan.dev](https://dss.kyllan.dev)

Graduating soon and applying to jobs? This app will help you negotiate your salary and find the best offer.


## Data

Sourced from [https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

[Data README](./data/README.md)

## Process Diagram

![Process Diagram](./process-diagram/processdiagram.drawio.svg)

## Modeling

This uses an XGBoost model to predict the salary of a data science candidate based on their experience, location, and other features. It has a MAE of $34,731.91. It's hyperparameters were tuned using a grid search with cross-validation. The final model has these hyperparameters:

```
{'colsample_bytree': 0.8, 
'learning_rate': 0.1, 
'max_depth': 5, 
'min_child_weight': 1, 
'n_estimators': 75, 
'subsample': 0.8}
```

## Deployment

All models are tracked in [mlflow](https://mlflow.org/docs/latest/index.html), when a new best model is found it is saved and containerized into a docker image.

This docker image is then deployed locally to an Ubuntu 22.04 virtual machine running on a proxmox cluster. 

## App

The app is built with Streamlit and available at [dss.kyllan.dev](https://dss.kyllan.dev) through a cloudflare tunnel. This allows for quick and secure deployment of the app locally so no cloud resources are needed.