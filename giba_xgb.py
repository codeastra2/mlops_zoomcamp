# %%
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.notebook import tqdm
import mlflow
from prefect import flow, task

# %%
mlflow.xgboost.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("gibaexp")


# %%
@task
def load_dset():
    BASE = '../datasets/godaddy-microbusiness-density-forecasting/'
    train = pd.read_csv(BASE + 'train.csv')
    test = pd.read_csv(BASE + 'test.csv')
    sub = pd.read_csv(BASE + 'sample_submission.csv')
    print(train.shape, test.shape, sub.shape)
    return train, test, sub

# %%
@task
def proc_dset(train, test):
    train['istest'] = 0
    test['istest'] = 1
    raw = pd.concat((train, test)).sort_values('row_id').reset_index(drop=True)

    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year
    raw["month"] = raw["first_day_of_month"].dt.month
    raw["dcount"] = raw.groupby(['cfips', 'istest'])['row_id'].cumcount()
    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    
    return raw


# %%
@task
def build_features(raw):

    for lag in range(1, 36):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')['active'].shift(lag)
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].bfill()
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[f'act_lag_{lag}'].bfill()

        raw.tail(20)
        
    return raw
    
#gc.collect()


# %%
@task
def vis1(raw):
    raw.iloc[-20:,:20]

# %%
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

@task
def met_vis():
    print( smape( np.array([0, 0]),  np.array([0, 0]) ) )
    print( smape( np.array([0, 0]),  np.array([0, 1]) ) )
    print( smape( np.array([0, 0]),  np.array([1, 0]) ) )
    print( smape( np.array([0, 0]),  np.array([1, 1]) ) )


# %%
@task
def print_lag_vs_smape(raw):
    for i in range(1, 36):
        print(f'smape lag{i}:', smape(raw.loc[raw.istest==0, 'microbusiness_density'], raw.loc[raw.istest==0, f'mbd_lag_{i}']))

# %%
@task
def train_model(raw):

    LAG_FEATURES = list(raw.columns[13:]) 
    print(LAG_FEATURES)

    for lead in range(1):

        with mlflow.start_run(run_name=f"Runconfig{lead}"):

            mlflow.set_tag("model", "xgboost")

            mlflow.log_params({f"lead": lead})


            print(lead)
            
            model = xgb.XGBRegressor(
                tree_method="hist",
                n_estimators=2000,
                learning_rate=0.025,
                #max_depth=8,
                max_leaves = 255,
                subsample=0.60,
                colsample_bytree=0.90,
                max_bin=256,
                n_jobs=2,
                eval_metric=smape, 
                disable_default_eval_metric=True,
                early_stopping_rounds=50,
            )

            features = []
            for f in LAG_FEATURES:
                if (int(f.split('_')[-1]) >= (lead+1)) and (int(f.split('_')[-1]) <= (lead+32)):
                    features.append(f)
            fw = np.ones(len(features))
            fw[0] = len(fw)
            fw /= np.sum(fw)
                    
            train_indices = (raw.istest==0) & (raw.dcount != 38)
            valid_indices = (raw.istest==0) & (raw.dcount == 38)
            model.fit(
                raw.loc[train_indices, features],
                raw.loc[train_indices, 'microbusiness_density'],
                eval_set=[(raw.loc[valid_indices, features], raw.loc[valid_indices, 'microbusiness_density'])],
                verbose=50,
                feature_weights=fw,
            )
            
            '''model = xgb.XGBRegressor(
                tree_method="hist",
                n_estimators=model.best_iteration+1,
                learning_rate=0.025,
                #max_depth=8,
                max_leaves = 255,
                subsample=0.60,
                colsample_bytree=0.90,
                max_bin=256,
                n_jobs=2,
                #eval_metric=smape, 
                #disable_default_eval_metric=True,
                #early_stopping_rounds=50,
            )    
            train_indices = (raw.istest==0)
            model.fit(
                raw.loc[train_indices, features],
                raw.loc[train_indices, 'microbusiness_density'],
                feature_weights=fw,
            )'''    

            mlflow.log_metric("rmse", model.evals_result()["validation_0"]["smape"][-1])
            mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost-model",
            registered_model_name="xgboost-model"
            )


            test_indices = (raw.istest==1) & (raw.dcount == lead)
            raw.loc[test_indices, 'microbusiness_density'] = model.predict(raw.loc[test_indices, features])
            print()

'''# %%
ax = raw.loc[(raw.cfips==10001)&(raw.istest==0)].plot(x='first_day_of_month', y='microbusiness_density')
raw.loc[(raw.cfips==10001)&(raw.istest==1)].plot(ax=ax, x='first_day_of_month', y='microbusiness_density')

# %%
ax = raw.loc[(raw.cfips==10003)&(raw.istest==0)].plot(x='first_day_of_month', y='microbusiness_density')
raw.loc[(raw.cfips==10003)&(raw.istest==1)].plot(ax=ax, x='first_day_of_month', y='microbusiness_density')

# %%
ax = raw.loc[(raw.cfips==9011)&(raw.istest==0)].plot(x='first_day_of_month', y='microbusiness_density')
raw.loc[(raw.cfips==9011)&(raw.istest==1)].plot(ax=ax, x='first_day_of_month', y='microbusiness_density')

# %%
ax = raw.loc[(raw.cfips==9013)&(raw.istest==0)].plot(x='first_day_of_month', y='microbusiness_density')
raw.loc[(raw.cfips==9013)&(raw.istest==1)].plot(ax=ax, x='first_day_of_month', y='microbusiness_density')

# %%
ax = raw.loc[(raw.cfips==9015)&(raw.istest==0)].plot(x='first_day_of_month', y='microbusiness_density')
raw.loc[(raw.cfips==9015)&(raw.istest==1)].plot(ax=ax, x='first_day_of_month', y='microbusiness_density')'''

# %%
@task
def make_preds(raw):
    test = raw.loc[raw.istest==1, ['row_id', 'microbusiness_density']].copy()
    test.to_csv('submission.csv', index=False)
    test.head(40)

# %%
@flow
def main():
    train, test, sub = load_dset()
    raw = proc_dset(train, test)
    raw = build_features(raw)

    #vis1(raw)
    met_vis()
    print_lag_vs_smape(raw)
    train_model(raw)
    make_preds(raw)
    

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    work_queue_name="ml"
)

deployment.apply()

    

# %%
#main()


