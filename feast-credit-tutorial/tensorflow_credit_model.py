"""TensorFlow/Keras credit scoring model backed by Feast feature store."""

from pathlib import Path

import feast
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class TensorFlowCreditScoringModel:
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "city",
        "state",
        "location_type",
    ]

    feast_features = [
        "zipcode_features:city",
        "zipcode_features:state",
        "zipcode_features:location_type",
        "zipcode_features:tax_returns_filed",
        "zipcode_features:population",
        "zipcode_features:total_wages",
        "credit_history:credit_card_due",
        "credit_history:mortgage_due",
        "credit_history:student_loan_due",
        "credit_history:vehicle_loan_due",
        "credit_history:hard_pulls",
        "credit_history:missed_payments_2y",
        "credit_history:missed_payments_1y",
        "credit_history:missed_payments_6m",
        "credit_history:bankruptcies",
        "total_debt_calc:total_debt_due",
    ]

    target = "loan_status"
    model_dir = "tf_credit_model"
    encoder_filename = "tf_encoder.bin"
    scaler_filename = "tf_scaler.bin"

    def __init__(self):
        import tensorflow as tf

        self.tf = tf
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.model = None
        self.fs = feast.FeatureStore(repo_path="feature_repo")

        if Path(self.model_dir).exists():
            self.model = tf.keras.models.load_model(self.model_dir)
            self.encoder = joblib.load(self.encoder_filename)
            self.scaler = joblib.load(self.scaler_filename)

    def _build_model(self, input_dim: int):
        tf = self.tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def train(self, loans: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        X, y = self._get_training_features(loans)
        self.model = self._build_model(X.shape[1])

        callbacks = [
            self.tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]
        history = self.model.fit(
            X, y.values,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )

        self.model.save(self.model_dir)
        joblib.dump(self.encoder, self.encoder_filename)
        joblib.dump(self.scaler, self.scaler_filename)
        return history

    def _get_training_features(self, loans: pd.DataFrame):
        df = self.fs.get_historical_features(
            entity_df=loans, features=self.feast_features
        ).to_df()
        self.encoder.fit(df[self.categorical_features])
        df[self.categorical_features] = self.encoder.transform(df[self.categorical_features])
        drop_cols = [self.target, "event_timestamp", "created_timestamp", "loan_id", "zipcode", "dob_ssn"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.reindex(sorted(X.columns), axis=1).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, df[self.target]

    def predict(self, request: dict) -> dict:
        feature_vector = self.fs.get_online_features(
            entity_rows=[{
                "zipcode": request["zipcode"][0],
                "dob_ssn": request["dob_ssn"][0],
                "loan_amnt": request["loan_amnt"][0],
            }],
            features=self.feast_features,
        ).to_dict()

        features = request.copy()
        features.update(feature_vector)
        df = pd.DataFrame.from_dict(features)
        df[self.categorical_features] = self.encoder.transform(df[self.categorical_features])
        df = df.drop(columns=["zipcode", "dob_ssn"], errors="ignore")
        df = df.reindex(sorted(df.columns), axis=1).fillna(0)
        X = self.scaler.transform(df)

        prob = float(self.model.predict(X, verbose=0)[0][0])
        return {"approved": int(prob >= 0.5), "probability": round(prob, 4)}
