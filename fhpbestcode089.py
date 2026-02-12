import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings

SEED = 42
warnings.filterwarnings('ignore')
np.random.seed(SEED)

def load_data():
    train = pd.read_csv('/kaggle/input/fhp-challenge/Train (1).csv')
    test = pd.read_csv('/kaggle/input/fhp-challenge/Test (1).csv')
    print(f"✓ Loaded: Train {train.shape}, Test {test.shape}")
    return train, test

def target_encode_cv(train_df, col, n_folds=5):
    encoded = np.zeros(len(train_df))
    target_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y = train_df['Target'].map(target_map)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for trn_idx, val_idx in kf.split(train_df, y):
        means = train_df.iloc[trn_idx].groupby(col)['Target'].apply(
            lambda x: x.map(target_map).mean()
        )
        encoded[val_idx] = train_df.iloc[val_idx][col].map(means).fillna(y.iloc[trn_idx].mean())
    return encoded

def advanced_features(train, test):
    print("\n[Feature Engineering]")
    
    # Missing values
    has_cols = [c for c in train.columns if 'has_' in c.lower()]
    for col in has_cols:
        train[col] = train[col].fillna('Never had')
        test[col] = test[col].fillna('Never had')
    
    fin_cols = ['business_turnover', 'business_expenses', 'personal_income']
    for col in fin_cols:
        if col in train.columns:
            med = train[train[col] > 0][col].median() if (train[col] > 0).any() else 0
            train[col] = train[col].fillna(med)
            test[col] = test[col].fillna(med)
    
    # Financial ratios
    for df in [train, test]:
        df['turnover_expense_ratio'] = df['business_turnover'] / (df['business_expenses'] + 1)
        df['income_turnover_ratio'] = df['personal_income'] / (df['business_turnover'] + 1)
        df['profit_margin'] = (df['business_turnover'] - df['business_expenses']) / (df['business_turnover'] + 1)
        df['debt_service'] = df['business_expenses'] / (df['personal_income'] + 1)
        df['capital_efficiency'] = df['business_turnover'] / (df['business_age_months'].fillna(1) + 1)
        df['is_profitable'] = (df['business_turnover'] > df['business_expenses']).astype(int)
        
        # Time features
        df['total_months'] = df['business_age_years'].fillna(0) * 12 + df['business_age_months'].fillna(0)
        df['is_startup'] = (df['total_months'] < 24).astype(int)
        df['is_mature'] = (df['total_months'] > 60).astype(int)
        
        # Financial tools
        df['fin_tool_count'] = sum([df[col].apply(lambda x: 1 if pd.notna(x) and 'have now' in str(x).lower() else 0) for col in has_cols])
        df['has_fin_access'] = (df['fin_tool_count'] > 0).astype(int)
    
    # Group aggregations
    for g in ['country', 'owner_sex']:
        if g in train.columns:
            for col in ['business_turnover', 'personal_income']:
                stats = train.groupby(g)[col].agg(['mean', 'median']).reset_index()
                stats.columns = [g, f'{g}_{col}_mean', f'{g}_{col}_med']
                train = train.merge(stats, on=g, how='left')
                test = test.merge(stats, on=g, how='left')
                test[f'{g}_{col}_mean'].fillna(train[col].median(), inplace=True)
                test[f'{g}_{col}_med'].fillna(train[col].median(), inplace=True)
    
    # Target encoding
    for col in ['country', 'owner_sex']:
        if col in train.columns:
            train[f'{col}_tenc'] = target_encode_cv(train, col)
            target_map = {'Low': 0, 'Medium': 1, 'High': 2}
            cat_means = train.groupby(col)['Target'].apply(lambda x: x.map(target_map).mean())
            test[f'{col}_tenc'] = test[col].map(cat_means).fillna(train['Target'].map(target_map).mean())
    
    # Clustering
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[fin_cols].fillna(0))
    test_scaled = scaler.transform(test[fin_cols].fillna(0))
    km = KMeans(n_clusters=8, random_state=SEED, n_init=10)
    train['cluster'] = km.fit_predict(train_scaled)
    test['cluster'] = km.predict(test_scaled)
    
    # Encoding
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c not in ['ID', 'Target']]
    combined = pd.concat([train[cat_cols], test[cat_cols]])
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    tmap = {'Low': 0, 'Medium': 1, 'High': 2}
    train['Target_Code'] = train['Target'].map(tmap)
    
    print(f"✓ Features: {train.shape[1] - 3}")
    return train, test, tmap

def train_lgb(X, y, X_test, n_folds=7):
    print("\n[LightGBM]")
    params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 7,
        'min_child_samples': 40, 'feature_fraction': 0.75, 'bagging_fraction': 0.75,
        'lambda_l1': 0.4, 'lambda_l2': 0.4, 'random_state': SEED, 'verbose': -1
    }
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    
    for fold, (trn, val) in enumerate(kf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[trn], y.iloc[trn])
        dval = lgb.Dataset(X.iloc[val], y.iloc[val])
        model = lgb.train(params, dtrain, 3000, valid_sets=[dval],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[val] = model.predict(X.iloc[val], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        print(f"  Fold {fold+1}: {f1_score(y.iloc[val], np.argmax(oof[val], axis=1), average='weighted'):.5f}")
    
    score = f1_score(y, np.argmax(oof, axis=1), average='weighted')
    print(f"  OOF: {score:.5f}")
    return oof, test_pred, score

def train_xgb(X, y, X_test, n_folds=7):
    print("\n[XGBoost]")
    params = {
        'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
        'max_depth': 7, 'learning_rate': 0.015, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.4, 'reg_lambda': 0.4,
        'seed': SEED, 'tree_method': 'hist', 'verbosity': 0
    }
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    
    for fold, (trn, val) in enumerate(kf.split(X, y)):
        dtrain = xgb.DMatrix(X.iloc[trn], y.iloc[trn])
        dval = xgb.DMatrix(X.iloc[val], y.iloc[val])
        model = xgb.train(params, dtrain, 3000, evals=[(dval, 'v')],
                         early_stopping_rounds=150, verbose_eval=0)
        oof[val] = model.predict(dval)
        test_pred += model.predict(xgb.DMatrix(X_test)) / n_folds
        print(f"  Fold {fold+1}: {f1_score(y.iloc[val], np.argmax(oof[val], axis=1), average='weighted'):.5f}")
    
    score = f1_score(y, np.argmax(oof, axis=1), average='weighted')
    print(f"  OOF: {score:.5f}")
    return oof, test_pred, score

def train_cat(X, y, X_test, n_folds=7):
    print("\n[CatBoost]")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(X_test), 3))
    
    for fold, (trn, val) in enumerate(kf.split(X, y)):
        model = CatBoostClassifier(
            iterations=3000, learning_rate=0.015, depth=7, l2_leaf_reg=3,
            loss_function='MultiClass', random_seed=SEED, verbose=0, early_stopping_rounds=150
        )
        model.fit(X.iloc[trn], y.iloc[trn], eval_set=(X.iloc[val], y.iloc[val]), verbose=0)
        oof[val] = model.predict_proba(X.iloc[val])
        test_pred += model.predict_proba(X_test) / n_folds
        print(f"  Fold {fold+1}: {f1_score(y.iloc[val], np.argmax(oof[val], axis=1), average='weighted'):.5f}")
    
    score = f1_score(y, np.argmax(oof, axis=1), average='weighted')
    print(f"  OOF: {score:.5f}")
    return oof, test_pred, score

def ensemble(train, test):
    print("\n" + "="*70)
    print("ENSEMBLE")
    print("="*70)
    
    X = train.drop(['ID', 'Target', 'Target_Code'], axis=1)
    y = train['Target_Code']
    X_test = test.drop(['ID', 'Target', 'Target_Code'], axis=1, errors='ignore')
    
    lgb_oof, lgb_test, lgb_sc = train_lgb(X, y, X_test)
    xgb_oof, xgb_test, xgb_sc = train_xgb(X, y, X_test)
    cat_oof, cat_test, cat_sc = train_cat(X, y, X_test)
    
    scores = np.array([lgb_sc, xgb_sc, cat_sc])
    weights = scores ** 2
    weights /= weights.sum()
    
    print(f"\nWeights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
    
    oof = lgb_oof * weights[0] + xgb_oof * weights[1] + cat_oof * weights[2]
    test_pred = lgb_test * weights[0] + xgb_test * weights[1] + cat_test * weights[2]
    
    ens_f1 = f1_score(y, np.argmax(oof, axis=1), average='weighted')
    print(f"\n✓ ENSEMBLE F1: {ens_f1:.5f}")
    print(classification_report(y, np.argmax(oof, axis=1), target_names=['Low', 'Med', 'High'], digits=4))
    
    return oof, test_pred, ens_f1

def optimize_thresholds(oof, y):
    print("\n[Threshold Optimization]")
    
    def loss(t):
        p = np.zeros(len(oof))
        for i in range(len(oof)):
            p[i] = 2 if oof[i, 2] >= t[2] else (1 if oof[i, 1] >= t[1] else 0)
        return -f1_score(y, p, average='weighted')
    
    result = minimize(loss, [0.33, 0.33, 0.33], bounds=[(0.1, 0.9)] * 3, method='Nelder-Mead')
    print(f"Optimal: {result.x}")
    print(f"F1 improvement: {-result.fun - f1_score(y, np.argmax(oof, axis=1), average='weighted'):+.5f}")
    return result.x

def create_submission(test_df, probs, thresholds, tmap):
    preds = np.zeros(len(probs), dtype=int)
    for i in range(len(probs)):
        preds[i] = 2 if probs[i, 2] >= thresholds[2] else (1 if probs[i, 1] >= thresholds[1] else 0)
    
    inv = {v: k for k, v in tmap.items()}
    labels = [inv[i] for i in preds]
    
    sub = pd.DataFrame({'ID': test_df['ID'], 'Target': labels})
    sub.to_csv('top10_submission.csv', index=False)
    
    print(f"\n✓ Saved: top10_submission.csv")
    dist = pd.Series(labels).value_counts()
    for lbl in ['Low', 'Medium', 'High']:
        print(f"  {lbl}: {dist.get(lbl, 0)} ({dist.get(lbl, 0)/len(sub)*100:.1f}%)")
    return sub

if __name__ == "__main__":
    print("="*70)
    print("FINANCIAL HEALTH PREDICTION - TOP 10 SOLUTION")
    print("="*70)
    
    train_df, test_df = load_data()
    train_eng, test_eng, target_map = advanced_features(train_df, test_df)
    oof_ensemble, test_ensemble, ens_f1 = ensemble(train_eng, test_eng)
    opt_thresh = optimize_thresholds(oof_ensemble, train_eng['Target_Code'])
    submission = create_submission(test_eng, test_ensemble, opt_thresh, target_map)
    
    print("\n" + "="*70)
    print(f"COMPLETE - Final F1: {ens_f1:.5f}")
    print("="*70)