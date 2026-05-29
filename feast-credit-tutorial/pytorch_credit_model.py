"""PyTorch credit scoring model backed by Feast feature store."""

from pathlib import Path

import feast
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class CreditNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class PyTorchCreditScoringModel:
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
    model_filename = "pytorch_credit_model.pt"
    encoder_filename = "pytorch_encoder.bin"
    scaler_filename = "pytorch_scaler.bin"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.model: CreditNet | None = None
        self.input_dim: int | None = None
        self.fs = feast.FeatureStore(repo_path="feature_repo")

        if Path(self.model_filename).exists():
            checkpoint = torch.load(self.model_filename, map_location=self.device)
            self.input_dim = checkpoint["input_dim"]
            self.model = CreditNet(self.input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.encoder = joblib.load(self.encoder_filename)
            self.scaler = joblib.load(self.scaler_filename)

    def train(self, loans: pd.DataFrame, epochs: int = 50, lr: float = 1e-3):
        X, y = self._get_training_features(loans)

        self.input_dim = X.shape[1]
        self.model = CreditNet(self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y.values, dtype=torch.float32).to(self.device)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

        torch.save(
            {"state_dict": self.model.state_dict(), "input_dim": self.input_dim},
            self.model_filename,
        )
        joblib.dump(self.encoder, self.encoder_filename)
        joblib.dump(self.scaler, self.scaler_filename)
        self.model.eval()

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

    def predict(self, request: dict) -> int:
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

        self.model.eval()
        with torch.no_grad():
            prob = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).item()
        return 1 if prob >= 0.5 else 0
