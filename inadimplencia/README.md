# Pipeline de previsão de inadimplência

## Integrantes do grupo
- Giulia Paura
- Maria Vitória Soares
- Valentina Ribeiro
- Roberta Correa

- `pipeline.pkl`: o objeto `pipe` (`sklearn.pipeline.Pipeline`) com `.predict` e `.predict_proba`.
- `treinar_pipeline.py`: script que gera o `pipeline.pkl` a partir do CSV.

## Resultado
- AUC em holdout (30% dos dados, nunca vistos no ajuste): **0.936**
- Testado carregando com `pickle.load` em scikit-learn 1.3 → 1.9 e numpy 1.26 / 2.x
  (o `.pkl` foi gerado com scikit-learn 1.3.2 + numpy 1.26.4 para máxima compatibilidade).
- Aceita `X` como DataFrame ou array numpy, com ou sem a coluna de índice (`Unnamed: 0`):
  as colunas são selecionadas pela posição contada a partir do final.

## Ideia
As variáveis isoladas têm AUC ≈ 0.50 e até gradient boosting fica em ≈ 0.50. Analisando
correlações, as 32 primeiras variáveis formam 8 blocos de 4 que medem 8 fatores latentes
(as 5 últimas são ruído). O default é a **paridade (XOR)** dos sinais desses 8 fatores.
O pipeline estima cada fator (quantis → normal, combinação linear com pesos otimizados),
aplica `tanh` (sinal suave), multiplica os 8 fatores e calibra com regressão logística.
Usa apenas objetos padrão de sklearn/numpy, então nenhum código extra é necessário para carregar.

Para regerar: `python treinar_pipeline.py credit_risk_student_dataset.csv --holdout`
