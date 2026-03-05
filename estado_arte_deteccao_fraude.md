# Estado da Arte em Detecção de Fraude em Transações Financeiras
### Pesquisa Nacional e Internacional — 2025

---

## Sumário Executivo

O sistema financeiro mundial enfrenta uma escalada sem precedentes de fraudes digitais. Globalmente, as perdas com fraude em cartão de crédito devem ultrapassar **US$ 403 bilhões nos próximos dez anos** (Nilson Report, 2024), enquanto perdas gerais com crimes financeiros representam entre **2% e 5% do PIB global** (FMI). No Brasil, o cenário é particularmente crítico: as perdas com fraude ultrapassaram **R$ 10 bilhões em 2024** — equivalente a 0,35%–0,40% do PIB nacional — e os golpes via PIX geraram prejuízo estimado de **R$ 4,9 bilhões** só em 2024, uma alta de 70% em relação ao ano anterior.

Em resposta, o estado da arte evoluiu de sistemas baseados em regras rígidas para arquiteturas híbridas com múltiplos modelos de machine learning, deep learning, e agora redes neurais em grafos (GNN) e modelos de fundação (Foundation Models). O foco atual é combinar **alta acurácia** com **latência mínima**, **explicabilidade regulatória** e **privacidade de dados**.

---

## 1. Contexto e Dimensão do Problema

### 1.1 Cenário Internacional

| Métrica | Valor | Fonte |
|---|---|---|
| Perdas globais com fraude de cartão | US$ 403,88 bi (10 anos) | Nilson Report, 2024 |
| Perdas globais com crimes financeiros | 2–5% do PIB global | FMI, 2024 |
| Fraude online projetada até 2028 | US$ 362 bi acumulados | US Treasury, 2024 |
| Prevenção via ML (EUA/Treasury, FY2024) | US$ 4 bi | US Dept. of Treasury, 2024 |
| Deepfake attacks em 2024 | 1 a cada 5 minutos | Identity Fraud Report, 2025 |
| Falsificações digitais de documentos | +244% ano a ano | Identity Fraud Report, 2025 |

Instituições com sistemas baseados em regras puras apresentaram **declínio de 37,4% na efetividade** nos últimos três anos, com **taxa de falsos positivos chegando a 92,3%** (Kroll Financial Crime Report, 2025), o que motivou uma migração acelerada para soluções baseadas em ML e IA.

### 1.2 Cenário Nacional — Brasil

O Brasil possui características únicas que amplificam os desafios de detecção de fraude:

**PIX como vetor principal:** O PIX ultrapassou R$ 26 trilhões transacionados e mais de 62 bilhões de transações em 2024 — crescimento de 48% — tornando-se responsável por mais de 80% das transferências do país. Com operação 24/7 e liquidação em segundos, o sistema não oferece janela de reversão manual.

**Nubank — Caso de Referência Nacional:** A Defense Platform do Nubank processa aproximadamente **450 milhões de eventos por dia**, gerando cerca de **5 milhões de requisições internas por minuto**. Um único evento como uma transação PIX pode acionar dezenas de processos subsequentes. A plataforma opera com **mais de 99,98% de disponibilidade**, em arquitetura distribuída com 20 shards no Brasil, usando stack baseada em Clojure, Datomic (sobre DynamoDB) e Kafka, com modelos de ML desenvolvidos em Python.

**Perfil das fraudes PIX (2024):** 38% das contas marcadas como fraudulentas estavam no nome do próprio fraudador; 27% eram contas-laranja; 1% envolveu falsidade ideológica na abertura. Isso indica que o problema não é apenas técnico — é sistêmico e envolve engenharia social.

**Regulação:** O Banco Central determina que fintechs e bancos utilizem IA e ML para identificar padrões fraudulentos em tempo real. O Mecanismo de Devolução Especial (MED) e o novo mecanismo de contestação de Pix (ativo no BB, Caixa, Itaú, Nubank, Inter e PicPay) são infraestruturas regulatórias que dependem diretamente de sistemas de detecção.

---

## 2. Definição de Requisitos do Sistema

Um sistema de detecção de fraude de nível bancário opera sob uma matriz de requisitos conflitantes. A engenharia desses sistemas consiste essencialmente em navegar esses trade-offs.

### 2.1 Requisitos Funcionais

| Requisito | Descrição | Métrica-Alvo |
|---|---|---|
| **Detecção (Recall/Sensitividade)** | Identificar a maior proporção possível de fraudes reais | ≥ 90% |
| **Precisão** | Minimizar transações legítimas incorretamente bloqueadas | ≥ 85% |
| **F1-Score** | Média harmônica de precision e recall | ≥ 0,90 |
| **AUC-ROC** | Capacidade discriminativa global do modelo | ≥ 0,95 |
| **AUPRC** | Área sob curva precisão-recall (mais relevante com classe desbalanceada) | ≥ 0,80 |

### 2.2 Requisitos Não-Funcionais

| Requisito | Descrição | Referência |
|---|---|---|
| **Latência (Real-Time)** | Tempo de inferência por transação | < 100ms (ideal < 50ms) |
| **Throughput** | Transações processadas por segundo | Milhões/hora |
| **Disponibilidade** | Uptime do sistema de detecção | ≥ 99,99% |
| **Escalabilidade** | Suporte a crescimento de volume sem degradação | Horizontal |
| **Explicabilidade (XAI)** | Justificativa interpretável de cada decisão | Obrigatório regulatório |
| **Privacidade** | Conformidade com LGPD/GDPR | Mandatório |
| **Adaptabilidade** | Capacidade de aprender novos padrões de fraude | Retraining contínuo |
| **Robustez Adversarial** | Resistência a ataques dos fraudadores ao próprio modelo | Alta |

### 2.3 O Problema Fundamental: Desequilíbrio de Classes

O principal desafio técnico de qualquer sistema antifraude é que transações fraudulentas representam tipicamente **menos de 1% do total** — às vezes 0,1% ou menos. Isso significa que um modelo ingênuo que classifica tudo como "legítimo" atingiria 99%+ de acurácia sem detectar nenhuma fraude. As métricas relevantes são, portanto, Precision, Recall, F1-Score e AUPRC — nunca apenas Accuracy.

Estratégias para lidar com o desequilíbrio:
- **Oversampling:** SMOTE (Synthetic Minority Over-sampling Technique), SMOTE-ENN
- **Undersampling:** Remoção de exemplos majoritários
- **Pesos de classe:** Penalização maior para erros na classe minoritária
- **Threshold tuning:** Ajuste do limiar de decisão conforme custo de negócio
- **Ensemble com reamostragem estratificada**

### 2.4 Concept Drift

Fraudadores evoluem continuamente suas táticas, tornando modelos treinados em dados históricos obsoletos. Isso exige:
- Monitoramento contínuo de distribuição de features
- Retreinamento periódico ou online (incremental learning)
- Sistemas de alerta para degradação de performance

---

## 3. Taxonomia dos Algoritmos

Os algoritmos são classificados em três grandes paradigmas:

```
Detecção de Fraude
├── Aprendizado Supervisionado (requer labels históricos)
│   ├── Algoritmos Clássicos (Logistic Regression, SVM, Decision Tree)
│   ├── Ensemble Methods (Random Forest, XGBoost, LightGBM, CatBoost)
│   └── Deep Learning (MLP, LSTM, CNN, Transformer, GNN)
├── Aprendizado Não-Supervisionado (sem labels)
│   ├── Detecção de Anomalias (Isolation Forest, Autoencoder)
│   └── Clustering (K-Means, DBSCAN)
└── Abordagens Híbridas
    ├── Semi-Supervisionado (poucos labels + muito dado não rotulado)
    ├── Ensemble Stacking (combinação de modelos)
    └── Federated Learning (treinamento distribuído com privacidade)
```

---

## 4. Algoritmos Mais Utilizados em Produção

### 4.1 XGBoost / LightGBM / CatBoost — Gradient Boosted Trees

**Posição:** O padrão-ouro atual para sistemas em produção na maioria das instituições.

**Como funciona:** Ensembles de árvores de decisão treinadas iterativamente, onde cada árvore corrige os erros das anteriores via gradient boosting.

**Performance empírica:**
- XGBoost: AUC-ROC de **0,964**, Precision de **0,928** em datasets de referência com 284.807 transações
- Em comparativo multi-modelo: XGBoost obteve F1-score de **0,94** e AUC-ROC de **0,97**, superando Random Forest (0,92/0,96) e Isolation Forest (0,85/0,81)
- LightGBM: similar ao XGBoost, com vantagem de velocidade em datasets muito grandes
- CatBoost: superior com features categóricas (ex: tipo de estabelecimento, cidade)

**Vantagens:**
- Excelente performance com dados tabulares estruturados
- Alta velocidade de inferência (< 10ms por transação)
- Interpretabilidade via SHAP values
- Robusto ao desequilíbrio de classes
- Amplamente suportado por frameworks de produção

**Limitações:**
- Não captura relacionamentos entre transações (cada transação é analisada isoladamente)
- Não modela sequências temporais nativamente
- Feature engineering manual intenso

**Onde é usado:** Praticamente todos os grandes bancos e fintechs possuem XGBoost ou LightGBM como camada base de scoring.

---

### 4.2 Random Forest

**Posição:** Segunda opção mais comum em produção, frequentemente usado em conjunto com boosting.

**Performance:** Accuracy de **92,4%** vs 83,7% de árvores isoladas; AUC-ROC de **0,96**. Em dataset de 565.000 transações reais (2024), atingiu **95,79% de acurácia** para detecção de fraude.

**Vantagens:** Menos suscetível a overfitting, naturalmente paralelizável, boa interpretabilidade.

**Limitação:** Geralmente levemente inferior ao XGBoost em performance pura.

---

### 4.3 Redes LSTM (Long Short-Term Memory)

**Posição:** Padrão para análise temporal de comportamento de clientes.

**Como funciona:** Redes recorrentes com células de memória que capturam dependências de longo prazo em sequências. Uma sequência de transações de um cliente é processada como uma série temporal.

**Caso de uso típico:** "Este cliente normalmente faz compras de R$ 50–200 em supermercados em São Paulo. Uma compra de R$ 8.000 em eletrônicos em Dubai às 3h da manhã é anômala em relação ao histórico."

**Performance:** Em stacking com Random Forest e meta-learner MLP + SMOTE-ENN: sensitividade (recall) de **1,00** e especificidade de **0,997**.

**Vantagens:** Captura padrões temporais e comportamento longitudinal do usuário.

**Limitações:** Mais lento que modelos baseados em árvores, requer mais dados, maior custo computacional de inferência.

---

### 4.4 Isolation Forest

**Posição:** Principal algoritmo não-supervisionado para detecção de anomalias.

**Como funciona:** Isola anomalias em árvores aleatórias — pontos anômalos são isolados mais rapidamente (precisam de menos divisões).

**Performance:** AUC-ROC de **0,81**, F1-score de **0,85**, Precision de **0,88**. Reduziu falsos positivos de 19,2% para 7,8% mantendo sensitividade de 91,3% em 1,26 milhão de transações.

**Vantagem principal:** Não requer dados rotulados — essencial quando labels históricos são escassos ou não confiáveis.

**Uso típico:** Monitoramento de novas contas (sem histórico), detecção de comportamentos inéditos.

---

### 4.5 Autoencoders

**Posição:** Principal abordagem de deep learning não-supervisionada.

**Como funciona:** Redes neurais treinadas para reconstruir as próprias entradas. Transações normais são reconstruídas com baixo erro; transações anômalas geram alto erro de reconstrução (anomaly score).

**Vantagem:** Aprende representações densas de comportamento normal sem necessidade de labels de fraude.

**Limitação:** Requer tuning cuidadoso do threshold de anomalia.

---

### 4.6 Regressão Logística

**Posição:** Baseline obrigatória e ainda amplamente usada em sistemas regulados por sua **plena interpretabilidade**.

**Performance:** Inferior às opções modernas (AUC-ROC típico de 0,85–0,90), mas cada coeficiente é diretamente interpretável.

**Onde ainda domina:** Sistemas com requisitos regulatórios extremos, scorecard de crédito, modelos auxiliares onde explicabilidade é mandatória.

---

## 5. Algoritmos em Ascensão

### 5.1 Graph Neural Networks (GNN) — A Maior Revolução em Curso

**Por que são transformadores:** A grande maioria dos algoritmos clássicos analisa cada transação de forma isolada. Mas fraudes raramente ocorrem em isolamento. Fraudadores operam em redes complexas: usam múltiplas contas, dispositivos compartilhados, endereços IPs em comum, transferem dinheiro por "laranjas" em cadeias.

GNNs modelam o sistema financeiro como um grafo onde:
- **Nós:** clientes, contas, cartões, dispositivos, comerciantes, IPs
- **Arestas:** transações, conexões, interações

Ao propagar informação pelo grafo, a GNN consegue sinalizar uma conta aparentemente normal porque ela está conectada a contas conhecidamente fraudulentas — algo impossível com modelos tabulares clássicos.

**Performance empírica:**
- GNN com Neo4j: **91% de accuracy**, AUC de **0,961**, com detecção de anéis de fraude coordenados
- RL-GNN (Reinforcement Learning + GNN): F1-score de **0,839**, AUROC de **0,872**, latência de **~42ms por batch** para grafos com mais de 500K transações, redução de **33% nos falsos positivos** vs GNNs baseline
- GNN + XGBoost (blueprint NVIDIA): Combinação que oferece o melhor dos dois mundos — relacionamentos entre transações + interpretabilidade de árvores

**Arquiteturas GNN mais relevantes:**
- **GAT (Graph Attention Networks):** Foca seletivamente nos vizinhos mais relevantes do grafo
- **GraphSAGE:** Eficiente para grafos grandes e dinâmicos
- **Temporal GNN (Tem-GNN, DGANN):** Incorpora dinâmica temporal às relações do grafo
- **Heterogeneous GNN:** Lida com múltiplos tipos de nós e arestas simultaneamente
- **JA-GNN (Jump-Attentive GNN):** Mitiga o problema de over-smoothing em GNNs profundas

**Desafios:** Escalabilidade computacional, latência em tempo real, camouflage (fraudadores que imitam padrões de usuários legítimos no grafo), e custo de infraestrutura.

**Adoção:** A NVIDIA lançou um AI Blueprint oficial combinando GNN + XGBoost, integrado com AWS, Cloudera e Dell. É o sinal mais claro de que GNNs estão deixando o laboratório e entrando em produção.

---

### 5.2 Federated Learning — Privacidade como Habilitador de Colaboração

**Problema que resolve:** Bancos possuem dados valiosos sobre fraudes que não podem compartilhar entre si devido à LGPD/GDPR e regulação bancária. Um fraudador que atuou no Itaú pode aparecer no Nubank amanhã — mas os bancos não podem trocar dados de clientes.

**Como funciona:** Cada instituição treina o modelo localmente em seus próprios dados. Apenas os gradientes (parâmetros do modelo) — não os dados — são agregados em um servidor central para atualizar o modelo global.

**Performance:** Precision de **0,95**, Recall de **0,88**, AUPRC de **0,96** em estudos com FL + XAI (LIME). FedAvg como algoritmo de agregação dominante, atingindo **91% de accuracy**.

**Tendência:** Combinação com XAI (Explainable AI) para compliance regulatório — o AWS CleanRooms, por exemplo, já viabiliza essa colaboração em dados compartilhados sem exposição de dados sensíveis.

**Limitação:** Ainda é um campo em desenvolvimento — centralizado supera FL em latência, e ataques adversariais aos gradientes são uma área de pesquisa ativa.

---

### 5.3 Foundation Models para Transações Financeiras

**A nova fronteira:** Assim como LLMs são pré-treinados em bilhões de textos e depois ajustados para tarefas específicas, Foundation Models financeiros são pré-treinados em bilhões de transações reais e depois fine-tuned para detecção de fraude, previsão de inadimplência, churn, etc.

**NPPR (Featurespace, 2024):** O principal exemplo atual — modelo treinado em bilhões de transações reais de 180 bancos europeus. Demonstrou capacidade zero-shot (detectar fraudes em novos contextos sem retreinamento), análogo ao que GPT faz para texto. A arquitetura usa encoder GRU (mais eficiente que Transformers para sequências transacionais longas).

**Perspectiva:** A analogia com LLMs é precisa: uma sequência de transações de um cliente é como uma "frase", e cada transação é um "token". O modelo aprende o "idioma" do comportamento financeiro normal.

**Status:** Ainda em fase de prototype e pesquisa acadêmica para a maioria das aplicações. A questão é se vão sair dos demos e entrar em produção em escala — e a tendência de 2025 é que sim.

---

### 5.4 Reinforcement Learning para Detecção Adaptativa

**Ideia:** Em vez de classificar estaticamente, o agente de RL aprende a melhor política de detecção maximizando uma função de recompensa que balanceia recall, precision e custo computacional.

**RL-GNN:** A combinação mais promissora — usa um controlador RL para otimizar dinamicamente o comportamento da GNN, resultando em ganho de **19,7% no recall** e redução de **33% nos falsos positivos** vs GNNs convencionais.

**Limitação:** Maior complexidade de implementação e instabilidade de treinamento.

---

### 5.5 Ensemble Stacking de Última Geração

**Tendência atual:** Em vez de usar um único modelo, empilham-se múltiplos modelos especializados cujas saídas alimentam um meta-learner.

**Exemplos de alto desempenho (2023–2025):**
- SMOTE-ENN + LSTM + GRU com MLP como meta-learner: recall **1,00**, especificidade **0,997**
- XGBoost + CatBoost + LightGBM com Bayesian hyperparameter tuning: F1-score de **0,92**, AUC de **1,00**
- CCAD: quatro detectores de anomalias + XGBoost como meta-learner com discordance learning

---

## 6. Comparativo de Algoritmos: Confiabilidade e Trade-offs

### 6.1 Mapa de Trade-offs

| Algoritmo | Acurácia (AUC) | Latência | Explicabilidade | Dados Necessários | Custo Infra | Concept Drift |
|---|---|---|---|---|---|---|
| Logistic Regression | ★★☆ (0,85–0,90) | ★★★ (<1ms) | ★★★ (total) | ★★★ (pouco) | ★★★ (mínimo) | ★★☆ |
| Random Forest | ★★★ (0,94–0,96) | ★★★ (<5ms) | ★★☆ (SHAP) | ★★☆ | ★★★ | ★★☆ |
| XGBoost/LightGBM | ★★★ (0,95–0,97) | ★★★ (<10ms) | ★★☆ (SHAP) | ★★☆ | ★★★ | ★★☆ |
| LSTM/GRU | ★★★ (0,93–0,97) | ★★☆ (20–50ms) | ★☆☆ | ★★★ (muito) | ★★☆ | ★★★ |
| Isolation Forest | ★★☆ (0,79–0,85) | ★★★ (<10ms) | ★★☆ | ★★★ (sem labels) | ★★★ | ★★☆ |
| Autoencoder | ★★☆ (0,82–0,90) | ★★☆ | ★☆☆ | ★★★ (sem labels) | ★★☆ | ★★★ |
| GNN | ★★★ (0,87–0,96) | ★★☆ (40–100ms) | ★★☆ | ★★★ (grafo) | ★☆☆ (alto) | ★★★ |
| GNN + XGBoost | ★★★ (melhor) | ★★☆ | ★★☆ (SHAP) | ★★★ | ★★☆ | ★★★ |
| Federated Learning | ★★☆ (0,88–0,93) | ★★☆ | ★★☆ (XAI) | ★★☆ | ★☆☆ | ★★☆ |
| Foundation Model | ★★★ (potencial) | ★☆☆ (alto) | ★☆☆ | ★☆☆ (bilhões) | ★☆☆ | ★★★ |

*★★★ = Excelente | ★★☆ = Bom | ★☆☆ = Limitado*

---

### 6.2 Algoritmos por Cenário de Aplicação

**Cenário 1 — Startup Fintech (recursos limitados, alta necessidade de velocidade de iteração):**
> Recomendação: **XGBoost + Isolation Forest**
> XGBoost para scoring supervisionado; Isolation Forest para novas contas sem histórico. Stack simples, interpretável, rápido de colocar em produção e atualizar.

**Cenário 2 — Banco médio com compliance regulatório estrito (LGPD, Banco Central):**
> Recomendação: **XGBoost/LightGBM + SHAP (XAI) + Logistic Regression como fallback**
> O modelo de boosting oferece alta performance; SHAP garante explicabilidade para auditoria; logística como fallback quando o modelo principal está em retreinamento. Federated Learning para enriquecer dados sem compartilhamento.

**Cenário 3 — Grande banco (escala > 50M transações/dia, operação PIX 24/7):**
> Recomendação: **Ensemble Stacking (XGBoost + LSTM + GNN) com meta-learner**
> GNN para detectar redes de fraude coordenada e contas-laranja; LSTM para análise comportamental temporal; XGBoost para scoring transacional rápido. Arquitetura em camadas com diferentes latências.

**Cenário 4 — Detecção de Lavagem de Dinheiro (AML) e Redes de Fraude Complexas:**
> Recomendação: **GNN (GAT ou GraphSAGE) + Temporal GNN**
> A natureza relacional da lavagem de dinheiro exige modelagem de grafo. Não há alternativa eficiente com modelos tabulares.

**Cenário 5 — Detecção sem Dados Rotulados (novas modalidades de fraude, novas contas):**
> Recomendação: **Isolation Forest + Autoencoder**
> Identificam anomalias sem conhecimento prévio do que é fraude. Alta taxa de falsos positivos, mas essencial para cobrir o desconhecido.

---

## 7. A Arquitetura Vencedora: Defesa em Camadas

O estado da arte não é um único algoritmo, mas uma **arquitetura multicamadas** onde diferentes modelos operam em paralelo ou em sequência, cada um especializado em um tipo de sinal:

```
Camada 1 — Regras de Negócio (Latência: < 1ms)
  • Limites de valor, países bloqueados, horários suspeitos
  • Rápido, mas estático — captura apenas fraudes conhecidas

Camada 2 — Scoring em Tempo Real (Latência: < 50ms)
  • XGBoost/LightGBM: score de risco por transação
  • Features: valor, hora, local, tipo, histórico recente

Camada 3 — Análise Comportamental/Temporal (Latência: 50–200ms)
  • LSTM/GRU: compara transação com histórico do cliente
  • "Isso é típico para este cliente?"

Camada 4 — Análise de Rede (Latência: 50–150ms, pode ser assíncrona)
  • GNN: analisa conexões entre contas, dispositivos, IPs
  • Detecta anéis de fraude e contas-laranja

Camada 5 — Fusão e Decisão
  • Meta-learner (MLP ou XGBoost) combina scores das camadas anteriores
  • Threshold adaptado ao custo de negócio

Camada 6 — Revisão Humana (assíncrona)
  • Casos de alta incerteza enviados para analistas
  • Feedback humano alimenta retreinamento
```

Esta é essencialmente a arquitetura da Defense Platform do Nubank e de sistemas equivalentes em grandes bancos internacionais.

---

## 8. Explicabilidade (XAI): De Desejo a Requisito Legal

Com a LGPD no Brasil e GDPR na Europa, explicar por que uma transação foi bloqueada passou de boa prática para **obrigação legal**. As principais técnicas:

**SHAP (SHapley Additive exPlanations):** Decompõe a predição de qualquer modelo nas contribuições de cada feature. "Esta transação foi bloqueada porque: valor 3x acima da média (+0,45), novo beneficiário (+0,30), horário incomum (+0,15)." Funciona perfeitamente com XGBoost/Random Forest.

**LIME (Local Interpretable Model-agnostic Explanations):** Aproxima localmente o modelo complexo com um modelo linear interpretável. Usado principalmente com deep learning.

**Atenção em Transformers/LSTMs:** Os pesos de atenção fornecem interpretabilidade parcial, indicando quais transações históricas mais influenciaram a decisão atual.

A combinação **Federated Learning + XAI** está emergindo como o padrão para sistemas que precisam ser simultaneamente colaborativos, privados e auditáveis.

---

## 9. Métricas de Avaliação: O que Realmente Importa

| Métrica | Fórmula | Por que importa em fraude |
|---|---|---|
| **Precision** | TP/(TP+FP) | Custo dos falsos positivos — clientes bloqueados indevidamente |
| **Recall/Sensitividade** | TP/(TP+FN) | Custo dos falsos negativos — fraudes não detectadas |
| **F1-Score** | 2×(P×R)/(P+R) | Balance entre os dois |
| **AUC-ROC** | Área sob curva ROC | Performance geral independente de threshold |
| **AUPRC** | Área sob curva P-R | Mais informativa que AUC-ROC com dados desbalanceados |
| **MCC** | (TP×TN-FP×FN)/√... | Leva em conta todos os quadrantes — robusto com desbalanceamento |

**Regra de ouro:** Nunca otimize apenas Accuracy em detecção de fraude. Um modelo que diz "tudo é legítimo" terá 99,9% de acurácia e zero utilidade.

---

## 10. Tendências para 2025–2028

1. **GNN como padrão de mercado:** A adoção de GNNs passará de pesquisa acadêmica para produção mainstream nos próximos 2–3 anos, especialmente para detecção de redes de contas-laranja (problema crítico no Brasil).

2. **Foundation Models transacionais:** Seguindo o caminho dos LLMs, modelos pré-treinados em bilhões de transações permitirão fine-tuning rápido para novas tarefas e transferência de conhecimento entre instituições.

3. **Federated Learning como infraestrutura regulatória:** Com a pressão regulatória do Banco Central e LGPD, FL pode tornar-se o mecanismo padrão de colaboração antifraude entre instituições no Brasil.

4. **Detecção de fraude com IA generativa:** Assim como fraudadores usam IA para criar deepfakes e documentos falsos, os sistemas de defesa precisarão de modelos adversariais capazes de detectar conteúdo sintético.

5. **RL para otimização de políticas de decisão:** Reinforcement Learning para otimizar dinamicamente thresholds e políticas de bloqueio em função do contexto e custo de negócio.

6. **Quantum ML:** Ainda distante da produção, mas pesquisas com Quantum GNNs para detecção de fraude já aparecem na literatura (2024).

---

## 11. Referências Principais

- MDPI Applied Sciences: "An Introduction to Machine Learning Methods for Fraud Detection" (Nov. 2025)
- Nature Sci. Reports: "RL-GNN fusion for real-time financial fraud detection" (Dez. 2025)
- Frontiers of Computer Science: "GNNs for Financial Fraud Detection: A Review" (Jan. 2025)
- NVIDIA Technical Blog: "Supercharging Fraud Detection with GNNs" (Jun. 2025)
- MDPI JRFM: "ML as a Tool for Fraud Risk in Banking Transactions" (Mar. 2025)
- arXiv 2502.00201: "Year-over-Year Developments in Financial Fraud Detection via Deep Learning" (Jan. 2025)
- Nubank Engineering Blog: "Escalando a Defesa Contra Fraudes" (Jul. 2025)
- DataRudder: "Data Report PIX 2025: Segurança em Pagamentos Instantâneos" (Out. 2025)
- ML Frontiers: "From XGBoost to Foundation Models" (Out. 2025)
- Alloy: "2025 State of Fraud Benchmark Report" (2025)
- Kroll Financial Crime Report 2025
- Nilson Report 2024

---

*Pesquisa compilada em março de 2026. Estado da arte baseado em literatura revisada por pares publicada entre 2024–2026, relatórios de mercado e documentação técnica de plataformas de produção.*
