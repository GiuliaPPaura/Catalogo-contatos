"""Treina o pipeline de previsao de inadimplencia e salva em pipeline.pkl.

Uso: python treinar_pipeline.py caminho/credit_risk_student_dataset.csv [--holdout]

Estrutura descoberta nos dados:
- As 32 primeiras variaveis formam 8 blocos de 4 variaveis fortemente
  correlacionadas (cada bloco mede um fator latente). As 5 restantes sao ruido.
- Nenhuma variavel isolada tem poder preditivo (AUC ~0.50), nem arvores de
  gradient boosting conseguem aprender (AUC ~0.50): o default e a PARIDADE
  (XOR) dos sinais dos 8 fatores, i.e. default = 1 quando o produto dos sinais
  dos fatores e negativo.

Pipeline (apenas objetos padrao do sklearn/numpy, para que o pickle.load
funcione sem precisar de nenhum codigo customizado):
  1. selecao das 32 colunas pelo nome (ignora indice/colunas extras)
  2. QuantileTransformer -> normal padrao
  3. combinacao linear por bloco (pesos otimizados por maxima verossimilhanca)
  4. tanh (versao suave do sinal) e produto entre os 8 fatores
  5. regressao logistica sobre o produto (calibra as probabilidades)
"""
import pickle
import sys
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer

GRUPOS = [
    ["annual_income", "savings_balance", "total_assets", "monthly_savings_rate"],
    ["debt_to_income_ratio", "credit_utilization", "num_open_credit_lines", "existing_loan_balance"],
    ["credit_score", "years_of_credit_history", "num_late_payments", "credit_mix_score"],
    ["years_at_current_job", "employment_stability_index", "income_volatility_score", "num_jobs_last_5y"],
    ["home_equity_ratio", "appraised_collateral_value", "loan_to_value_ratio", "num_collateral_assets"],
    ["years_with_primary_bank", "num_bank_products_held", "average_account_balance", "direct_deposit_tenure_months"],
    ["discretionary_spending_ratio", "spending_volatility_score", "essential_expense_ratio", "avg_transaction_size"],
    ["emergency_fund_months", "liquid_assets_ratio", "checking_savings_balance", "num_overdrafts_last_year"],
]
COLUNAS = [c for g in GRUPOS for c in g]
NG = len(GRUPOS)


def otimizar_pesos(Z, y):
    """Pesos de cada variavel dentro do seu bloco, maximizando a verossimilhanca de
    p = sigmoid(a * prod_g tanh(sum_j w_gj z_gj) + c)."""
    corr = np.corrcoef(Z.T)
    w0 = np.concatenate([np.sign(corr[4 * i, 4 * i:4 * i + 4]) for i in range(NG)]) * 0.5

    def nll(p):
        fatores = (Z * p[:-2]).reshape(len(Z), NG, 4).sum(2)
        s = p[-2] * np.tanh(fatores).prod(1) + p[-1]
        return np.mean(np.logaddexp(0, s) - y * s)

    r = minimize(nll, np.concatenate([w0, [-3.0, 0.0]]), method="L-BFGS-B", options={"maxiter": 3000})
    W = np.zeros((len(COLUNAS), NG))
    for i in range(NG):
        W[4 * i:4 * i + 4, i] = r.x[4 * i:4 * i + 4]
    return W


def treinar(df):
    quantis = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    Z = quantis.fit_transform(df[COLUNAS].to_numpy())
    W = otimizar_pesos(Z, df["default"].to_numpy())
    pipe = Pipeline([
        ("selecionar", FunctionTransformer(itemgetter(COLUNAS))),
        ("quantis", quantis),
        ("fatores", FunctionTransformer(np.dot, kw_args={"b": W})),
        ("tanh", FunctionTransformer(np.tanh)),
        ("paridade", FunctionTransformer(np.prod, kw_args={"axis": 1, "keepdims": True})),
        ("logistica", LogisticRegression()),
    ])
    # quantis e pesos ja ajustados; reajustar o pipeline inteiro so recalibra a logistica
    pipe.fit(df.drop(columns="default"), df["default"])
    return pipe


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    if "--holdout" in sys.argv:
        tr, te = train_test_split(df, test_size=0.3, random_state=1, stratify=df["default"])
        p = treinar(tr)
        print("AUC holdout:", roc_auc_score(te["default"], p.predict_proba(te.drop(columns="default"))[:, 1]))
    pipe = treinar(df)
    print("AUC treino (base completa):", roc_auc_score(df["default"], pipe.predict_proba(df.drop(columns="default"))[:, 1]))
    with open("pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f, protocol=4)
