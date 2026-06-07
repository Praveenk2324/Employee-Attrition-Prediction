import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def build_pipeline(config: dict) -> Pipeline:
    """
    Builds a single sklearn Pipeline:
      step 1 — ColumnTransformer (preprocessor)
      step 2 — XGBClassifier
 
    Accepts raw DataFrame. No manual .transform() needed anywhere.
    """
    
    feat = config['features']
    m_cfg = config['model']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehotencoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feat['onehot']),
            ('ordinalencoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), feat['ordinal']),
            ("binary", OrdinalEncoder(), feat["binary"]),
            ("num", "passthrough", feat["numeric"]),
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    model_params = {
        "n_estimators": m_cfg["n_estimators"],
        "max_depth": m_cfg["max_depth"],
        "learning_rate": m_cfg["learning_rate"],
        "scale_pos_weight": m_cfg["scale_pos_weight"],
        "subsample": m_cfg["subsample"],
        "colsample_bytree": m_cfg["colsample_bytree"],
        "eval_metric": m_cfg["eval_metric"],
        "random_state": m_cfg["random_state"],
        "tree_method": "hist",
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('XGboost', XGBClassifier(**model_params))
    ])
    logger.info("Pipeline built: ColumnTransformer → XGBClassifier")
    return pipeline