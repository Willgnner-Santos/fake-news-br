# Datasets para Detecção / Verificação de Fake News (foco em PT-BR)

Este README lista conjuntos de dados relevantes para **fake news** e **fact-checking**, dando prioridade aos recursos em **português do Brasil (PT-BR)**. Inclui notas sobre escopo (fake/real vs. viés, rumor/stance, etc.) e links oficiais.

---

## 📊 Tabela resumida (prioridade PT-BR)

| Dataset | Idioma | Tarefa / Classes | Link principal | Observações |
|---|---|---|---|---|
| **Fake.Br Corpus** | PT-BR | **Fake vs Real** | https://github.com/roneysco/Fake.br-Corpus | Baseline clássico PT-BR, notícias rotuladas. |
| **FactChecks.br (Anonymous)** | PT-BR | Metadados de checagens | https://huggingface.co/datasets/fake-news-Anonymous/FactChecksbr | Links e registros de *fact-checks* brasileiros (não binário direto). |
| **MuMiN-PT** | PT-BR (subset) | Rumor / Stance / Social | https://huggingface.co/datasets/ju-resplande/MuMiN-PT | Subconjunto PT de corpus multilíngue focado em rumores/redes sociais. |
| **FakeRecogna** | PT-BR | Detecção (ver repo p/ rótulos) | https://github.com/Gabriel-Lino-Garcia/FakeRecogna | Recurso PT-BR; verificar esquema e rótulos no repositório. |
| **Central de Fatos (paper)** | PT-BR | Referência / Mapeamento | https://sol.sbc.org.br/index.php/dsw/article/view/17421 | Artigo; discute acervos e fluxos de checagem no Brasil. |
| **Survey CEUR-WS** | PT-BR | Revisão (survey) | https://ceur-ws.org/Vol-3199/paper1.pdf | Visão geral acadêmica sobre detecção/checagem; útil para referenciar. |
| **FactNews** | PT-BR/Mult | **Não** é fake/real: −1=Quotes, 0=Factual, 1=Biased | http://zenodo.org/records/10794023 | Útil p/ viés/factualidade; **não** ground truth fake/real. |
| **FactCheckTweet (tweets)** | PT-BR | Metadados de checagens em tweets | (Google Drive ID / gdown) | Tweets com links a checagens (*article_url*). Exige normalização; nem sempre há rótulo direto FAKE/REAL. |
| **FKTC** | PT-BR | Coleta / ferramentas | https://github.com/GoloMarcos/FKTC | Acervo/coleção; conferir documentação. |

### Outros (principalmente EN ou multilíngue)

| Dataset | Idioma | Tarefa / Classes | Link principal | Observações |
|---|---|---|---|---|
| **LIAR** | EN | Multiclasse (e.g., *pants-on-fire*, *false*, …) | https://huggingface.co/datasets/ucsbnlp/liar/blob/main/liar.py | Curto, político, não PT-BR. |
| **PolitiFact (Kaggle)** | EN | Fact-checking (metadados/texto) | https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data | Bom p/ benchmarks EN. |
| **FakeNewsNet** | EN | Multimodal (texto + engajamento) | https://github.com/KaiDMML/FakeNewsNet | Largamente em inglês (BuzzFeed/PolitiFact). |
| **FakeNewsSet** | — | — | https://huggingface.co/datasets/fake-news-Anonymous/FakeNewsSet/blob/main/FakeNewsSet.py | Definição disponível; dados completos não totalmente públicos. |
| **FCN** | Misto | Verificar escopo/tema | https://zenodo.org/records/5236636 | Pode ser temático (ex.: COVID); confirmar idioma. |

> **Nota sobre FactNews:** classes **−1 → “Quotes”**, **0 → “Factual”**, **1 → “Biased”**. **Não** usar como substituto direto de *fake vs real*.

---

## 🇧🇷 PT-BR (detalhado)

### Fake.Br Corpus
- **Link:** https://github.com/roneysco/Fake.br-Corpus  
- **Tarefa:** classificação **fake vs real** em PT-BR.  
- **Formato:** textos de notícias com rótulos binários.  
- **Uso típico:** baseline PT-BR; ótimo para few/zero-shot ou supervisionado.  
- **Dica:** padronize colunas (`idx`, `text`, `label`) e salve em Parquet/CSV p/ pipelines.

### FactChecks.br (acervo Anonymous)
- **Link:** https://huggingface.co/datasets/fake-news-Anonymous/FactChecksbr  
- **Tipo:** **metadados** de checagens brasileiras (títulos, URLs, etc.).  
- **Uso:** ótimo para coletar **evidências** e rastros de checagem; não é binário direto.

### MuMiN-PT (subset)
- **Links:** https://huggingface.co/datasets/ju-resplande/MuMiN-PT · Paper: https://dl.acm.org/doi/abs/10.1145/3477495.3531744  
- **Tarefa:** rumor verification / stance / interações sociais.  
- **Observação:** não é “notícia longa → fake/real” clássico; foco em rumor/redes sociais.

### FakeRecogna
- **Link:** https://github.com/Gabriel-Lino-Garcia/FakeRecogna  
- **Observação:** conjunto + ferramentas para PT-BR; conferir no repositório o esquema de rótulos/partições.

### Central de Fatos (paper)
- **Link:** https://sol.sbc.org.br/index.php/dsw/article/view/17421  
- **Uso:** referência/survey sobre acervos e o ecossistema brasileiro de checagem.

### **Survey CEUR-WS (Panorama PT-BR)**
- **Link:** https://ceur-ws.org/Vol-3199/paper1.pdf  
- **Conteúdo:** visão geral de técnicas e desafios em PT-BR; bom para contextualizar trabalhos e citar em revisões relacionadas.

### FactNews (atenção ao rótulo)
- **Links:** Zenodo: http://zenodo.org/records/10794023 · Paper: https://arxiv.org/pdf/2301.11850  
- **Classes:** −1 *Quotes*, 0 *Factual*, 1 *Biased*.  
- **Uso:** análise de viés/factualidade; **não** usar como fake/real.

### **FactCheckTweet (tweets com checagens)**
- **Aquisição:** disponível via **Google Drive ID** (ex.: `gdown --id <ID>`).  
- **Esquema típico:** `tweet_id`, `article_url`, `label` (quando presente).  
- **Observações práticas:**  
  - Muitos registros trazem **apenas o link da checagem**; nem sempre há rótulo FAKE/REAL diretamente.  
  - Alguns **links quebrados** exigem *crawling* ou *resolvers* de URL.  
  - Requer **normalização** para alinhar com tarefas binárias (ex.: inferir rótulos a partir do veredito do artigo de checagem).  

### FKTC
- **Link:** https://github.com/GoloMarcos/FKTC  
- **Observação:** acervo/coleta; ver docs para formato e disponibilidade.

---

## 🌐 Multilíngue / Inglês (para transferência ou comparação)

### LIAR
- **Links:** HF: https://huggingface.co/datasets/ucsbnlp/liar/blob/main/liar.py · Kaggle (var.): https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset?select=valid.tsv  
- **Idioma:** inglês.  
- **Tarefa:** *fact-checking* multiclasse (rótulos granulares).  
- **Uso:** útil para *transfer learning* e comparação com PT-BR.

### PolitiFact (Kaggle)
- **Link:** https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset/data  
- **Observação:** checagens em EN; bom para benchmarks.

### FakeNewsNet
- **Link:** https://github.com/KaiDMML/FakeNewsNet  
- **Observação:** multimodal (texto + engajamento), majoritariamente EN.

### FakeNewsSet
- **Links:** ACM: https://dl.acm.org/doi/abs/10.1145/3428658.3430965 · HF def.: https://huggingface.co/datasets/fake-news-Anonymous/FakeNewsSet/blob/main/FakeNewsSet.py  
- **Observação:** definição publicada; dataset completo não totalmente público.

### FCN
- **Link:** https://zenodo.org/records/5236636  
- **Observação:** verificar idioma e domínio (pode ser temático).

---

## 🧰 Exemplos rápidos de uso

### Carregar no Hugging Face
```python
from datasets import load_dataset

# FactChecks.br (metadados de fact-checks brasileiros)
ds_fc = load_dataset("fake-news-XXX/FactChecksbr")

# MuMiN-PT (subset em português)
ds_mumin = load_dataset("ju-resplande/MuMiN-PT")
```

### Converter corpus local para Parquet
```python
import pandas as pd

df = pd.DataFrame([
    {"idx":"fake_0001","text":"Exemplo de notícia...", "label":"FAKE"},
    {"idx":"true_0001","text":"Outra notícia...", "label":"REAL"},
])
df.to_parquet("meu_dataset.parquet", index=False)
```

---

## ✅ Escolha rápida por objetivo

- **Fake vs Real (PT-BR):** **Fake.Br** (principal), **FakeRecogna** (ver labels), e subsets PT do **MuMiN-PT** (se sua tarefa for rumor/stance).  
- **Evidências e histórico de checagens (PT-BR):** **FactChecks.br**, **Central de Fatos**, **FKTC**, **FactCheckTweet**.  
- **Viés/Factualidade (não fake/real):** **FactNews**.  
- **Transferência / comparação (EN):** **LIAR**, **PolitiFact**, **FakeNewsNet**.

> Dica: ao unificar corpora, normalize colunas para `idx`, `text`, `label` (com `label ∈ {FAKE, REAL}` quando aplicável) e centralize tudo em Parquet para pipelines reprodutíveis.


Novo dataset: 
FACTCK.BR https://github.com/jghm-f/FACTCK.BR/blob/master/FACTCKBR.tsv
Esses tem as classes: ['falso',
 'distorcido',
 'impreciso',
 'exagerado',
 'insustentável',
 'verdadeiro',
 'outros',
 'Falso',
 'Subestimado',
 'Verdadeiro',
 'Exagerado',
 'Impossível provar',
 'Discutível',
 'Sem contexto',
 nan,
 'Distorcido',
 'De olho',
 'Verdadeiro, mas',
 'Ainda é cedo para dizer'] - No artigo, sometne considere Falso e Verdadeiro
FAKETRUE.BR https://github.com/roneysco/Fake.br-Corpus