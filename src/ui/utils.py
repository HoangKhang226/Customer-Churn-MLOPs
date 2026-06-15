import pandas as pd
import numpy as np
import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0: return 0.0
    return np.sqrt(phi2corr / denom)

def compute_risk_features(df):
    df_feat = df.copy()
    
    # 1. security_score
    protective_cols = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
    available_prot = [c for c in protective_cols if c in df_feat.columns]
    if available_prot:
        df_feat['security_score'] = (df_feat[available_prot] == 'Yes').sum(axis=1)
        if 'InternetService' in df_feat.columns:
            df_feat.loc[df_feat['InternetService'] == 'No', 'security_score'] = -1
    else:
        df_feat['security_score'] = 0

    # 2. streaming_score
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    available_stream = [c for c in streaming_cols if c in df_feat.columns]
    if available_stream:
        df_feat['streaming_score'] = (df_feat[available_stream] == 'Yes').sum(axis=1)
        if 'InternetService' in df_feat.columns:
            df_feat.loc[df_feat['InternetService'] == 'No', 'streaming_score'] = -1
    else:
        df_feat['streaming_score'] = 0
        
    # 3. zero_supportive_service
    if 'security_score' in df_feat.columns:
        df_feat['zero_supportive_service'] = (df_feat['security_score'] == 0).astype(int)
        
    # 4. manual_payment
    if 'PaymentMethod' in df_feat.columns:
        manual_methods = ['Electronic check', 'Mailed check']
        df_feat['manual_payment'] = df_feat['PaymentMethod'].isin(manual_methods).astype(int)
        
    # 5. composite_risk_profile
    if 'Contract' in df_feat.columns and 'InternetService' in df_feat.columns:
        df_feat['composite_risk_profile'] = (
            (df_feat['Contract'] == 'Month-to-month') & 
            (df_feat['InternetService'] == 'Fiber optic')
        ).astype(int)
        
    return df_feat
