"""LLM-based credit explanation model using Feast feature store + RAG."""

from pathlib import Path
from typing import Optional

import feast
import numpy as np
import pandas as pd
from feast import FeatureStore


SYSTEM_PROMPT = """You are a credit risk analyst assistant.
Given a loan applicant's financial profile, explain the credit decision clearly.
Be concise, factual, and reference specific feature values in your explanation."""


def _format_credit_profile(features: dict) -> str:
    """Build a structured text profile from Feast feature values."""
    return f"""
Credit Application Profile:
- Total Debt Due:       ${features.get('total_debt_due', 0):,.0f}
- Credit Card Due:      ${features.get('credit_card_due', 0):,.0f}
- Mortgage Due:         ${features.get('mortgage_due', 0):,.0f}
- Student Loan Due:     ${features.get('student_loan_due', 0):,.0f}
- Vehicle Loan Due:     ${features.get('vehicle_loan_due', 0):,.0f}
- Hard Pulls (recent):  {features.get('hard_pulls', 0)}
- Missed Payments 6m:   {features.get('missed_payments_6m', 0)}
- Missed Payments 1y:   {features.get('missed_payments_1y', 0)}
- Missed Payments 2y:   {features.get('missed_payments_2y', 0)}
- Bankruptcies:         {features.get('bankruptcies', 0)}
- Location:             {features.get('city', 'N/A')}, {features.get('state', 'N/A')}
- Population (zip):     {features.get('population', 0):,}
- Total Wages (zip):    ${features.get('total_wages', 0):,}
""".strip()


class LLMCreditAdvisor:
    """
    Retrieves credit features from Feast and generates a natural-language
    credit decision explanation using an LLM.

    Supports:
      - OpenAI-compatible API (GPT-4, local Ollama, vLLM)
      - HuggingFace transformers (local inference)
    """

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

    def __init__(
        self,
        repo_path: str = "feature_repo",
        llm_backend: str = "openai",  # "openai" | "huggingface" | "ollama"
        model_name: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.fs = FeatureStore(repo_path=repo_path)
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client:
            return self._llm_client

        if self.llm_backend in ("openai", "ollama"):
            from openai import OpenAI
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._llm_client = OpenAI(**kwargs)

        elif self.llm_backend == "huggingface":
            from transformers import pipeline
            self._llm_client = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
            )

        return self._llm_client

    def get_features(self, zipcode: int, dob_ssn: str, loan_amnt: int) -> dict:
        """Retrieve credit features from Feast online store."""
        result = self.fs.get_online_features(
            entity_rows=[{
                "zipcode": zipcode,
                "dob_ssn": dob_ssn,
                "loan_amnt": loan_amnt,
            }],
            features=self.feast_features,
        ).to_dict()
        # Flatten single-element lists
        return {k: v[0] if isinstance(v, list) else v for k, v in result.items()}

    def explain_decision(
        self,
        zipcode: int,
        dob_ssn: str,
        loan_amnt: int,
        loan_intent: str,
        decision: int,  # 0 = rejected, 1 = approved
    ) -> str:
        """
        Retrieve features from Feast and generate a natural-language explanation.
        """
        features = self.get_features(zipcode, dob_ssn, loan_amnt)
        profile_text = _format_credit_profile(features)

        decision_label = "APPROVED" if decision == 1 else "REJECTED"

        user_prompt = f"""
The following loan application has been {decision_label}:

Loan Amount Requested: ${loan_amnt:,}
Loan Intent: {loan_intent}

{profile_text}

Please provide a concise explanation (3-5 sentences) of why this application
was {decision_label}, referencing the specific financial data above.
""".strip()

        return self._generate(user_prompt)

    def _generate(self, user_prompt: str) -> str:
        client = self._get_llm_client()

        if self.llm_backend in ("openai", "ollama"):
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        elif self.llm_backend == "huggingface":
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            output = client(prompt)
            return output[0]["generated_text"][len(prompt):].strip()

        return "LLM backend not configured."

    def risk_summary(self, features: dict) -> dict:
        """Rule-based risk scoring from Feast features — used alongside LLM."""
        score = 0
        flags = []

        if features.get("missed_payments_6m", 0) > 1:
            score += 30
            flags.append("Multiple missed payments in last 6 months")
        if features.get("bankruptcies", 0) > 0:
            score += 40
            flags.append("Bankruptcy on record")
        if features.get("hard_pulls", 0) > 3:
            score += 10
            flags.append("High number of recent credit inquiries")
        if features.get("total_debt_due", 0) > 50000:
            score += 20
            flags.append("High total debt load")

        return {
            "risk_score": min(score, 100),
            "risk_level": "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW",
            "flags": flags,
        }
