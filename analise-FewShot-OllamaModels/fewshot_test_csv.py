#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase de TESTE — Rotular CSV externo usando few-shot por recuperação no corpus Fake.Br
------------------------------------------------------------------------------------
- NÃO usa os dados do CSV para few-shot. Os exemplos few-shot vêm SOMENTE do corpus Fake.Br.
- O CSV externo é apenas o conjunto de ALVOS a rotular.
- Campos importantes no CSV: ID, text, Rótulo (o rótulo é opcional; se existir, é salvo junto para avaliação posterior).

Uso (exemplo):
    python fewshot_test_csv.py \
      --corpus-parquet "datasets/portuguese-fact-checking/Fake.br_raw.parquet" \
      --api-host "" \
      --models gemma3 gemma2:9b \
      --embed-model nomic-embed-text:latest \
      --k 6 \
      --test-csv "codes/GUAIA-textualDataset-compose/datasets/300-noticias-v2-filtradas.csv" \
      --test-id-col "ID" \
      --test-text-col "text" \
      --test-label-col "Rótulo" \
      --out-dir outputs_test_csv

Requisitos:
    pip install pandas numpy scikit-learn pyarrow ollama
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Cliente Python do Ollama
try:
    from ollama import Client
except Exception as _e:
    Client = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\\s+", " ", str(s)).strip()


def norm_id(x) -> str:
    return str(x).strip()


def find_json_block(text: str) -> str:
    s = str(text)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return ""


def parse_label(raw: str) -> str:
    block = find_json_block(raw)
    if block:
        try:
            js = json.loads(block)
            label = str(js.get("label", "")).upper().strip()
            if label in {"FAKE", "REAL"}:
                return label
        except Exception:
            pass
    low = str(raw).lower()
    if "fake" in low and "real" not in low:
        return "FAKE"
    if "real" in low and "fake" not in low:
        return "REAL"
    return "FAKE"


def clean_text_for_prompt(text: str, max_chars: int = 1200) -> str:
    t = normalize_whitespace(text)
    if len(t) > max_chars:
        t = t[:max_chars] + " [...]"
    return t


def detect_label_mapping(series: pd.Series) -> Dict:
    unique_vals = series.dropna().unique().tolist()
    if all(
        isinstance(v, (int, np.integer, np.int64, np.int32)) for v in unique_vals
    ) and set(unique_vals).issubset({0, 1}):
        return {0: "FAKE", 1: "REAL"}
    inv = {}
    for v in unique_vals:
        s = str(v).strip().lower()
        if "fake" in s:
            inv["FAKE"] = v
        elif "real" in s or "verdad" in s:
            inv["REAL"] = v
    if "FAKE" in inv and "REAL" in inv:
        return {inv["FAKE"]: "FAKE", inv["REAL"]: "REAL"}
    if len(unique_vals) == 2:
        vals_sorted = sorted(unique_vals, key=lambda x: str(x))
        return {vals_sorted[0]: "FAKE", vals_sorted[1]: "REAL"}
    raise ValueError(f"Rótulos inesperados no corpus: {unique_vals}")


PROMPT_TEMPLATE_HEADER = """Você é um verificador de fatos. Classifique notícias como "FAKE" ou "REAL".
Regras:
- Leia APENAS o texto fornecido.
- Não use conhecimento externo; não invente fatos.
- Responda em UMA linha no formato JSON exato: {"label": "FAKE"|"REAL", "justificativa": "<máx. 20 palavras>"}.
"""

PROMPT_EX_BLOCK = """[EX{idx}]
Texto: "{texto}"
Rótulo: {{"label": "{rotulo}", "justificativa": "{just}"}}"""

PROMPT_TEMPLATE_TAIL = """--- NOVO CASO ---
Texto: "{texto_alvo}"
Responda:
"""


def build_fewshot_prompt(examples: List[Tuple[str, str]], target_text: str) -> str:
    blocks = []
    for i, (tx, lab) in enumerate(examples, start=1):
        just = "..."
        txc = clean_text_for_prompt(tx).replace('"', "'")
        blocks.append(PROMPT_EX_BLOCK.format(idx=i, texto=txc, rotulo=lab, just=just))
    prompt = (
        PROMPT_TEMPLATE_HEADER
        + "\n"
        + "\n".join(blocks)
        + "\n"
        + PROMPT_TEMPLATE_TAIL.format(
            texto_alvo=clean_text_for_prompt(target_text).replace('"', "'")
        )
    )
    return prompt


class OllamaClient:
    def __init__(self, host: str, embed_model: str):
        if Client is None:
            raise RuntimeError(
                "Pacote 'ollama' não encontrado. Instale com 'pip install ollama'."
            )
        self.client = Client(host=host)
        self.embed_model = embed_model

    def get_embedding(self, text: str) -> np.ndarray:
        model_name = self.embed_model
        try:
            resp = self.client.embeddings(model=model_name, prompt=text)
        except Exception as e:
            if ":lates" in model_name:
                model_name = model_name.replace(":lates", ":latest")
                resp = self.client.embeddings(model=model_name, prompt=text)
            else:
                raise e
        emb = (
            resp.get("embedding", None)
            if isinstance(resp, dict)
            else getattr(resp, "embedding", None)
        )
        if emb is None:
            raise ValueError(
                f"Resposta de embedding inesperada: {type(resp)} -> {resp}"
            )
        return np.asarray(emb, dtype=np.float32)

    def generate(
        self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 256
    ) -> str:
        out = self.client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        if isinstance(out, dict):
            return out.get("response", "")
        return getattr(out, "response", "")


def select_neighbors_balance(
    df_corpus: pd.DataFrame,
    emb_matrix: np.ndarray,
    text_target: str,
    text_col_corpus: str,
    label_col_corpus: str,
    mapping: Dict,
    oc: OllamaClient,
    k: int = 6,
) -> List[int]:
    assert k % 2 == 0, "k deve ser par (k/2 FAKE + k/2 REAL)"
    q_emb = oc.get_embedding(text_target).astype(np.float32)
    sims = cosine_similarity(q_emb.reshape(1, -1), emb_matrix)[0]

    labels = df_corpus[label_col_corpus].values
    labels_sr = np.array([mapping.get(lab, str(lab)) for lab in labels], dtype=object)

    neigh_idx = []
    half = k // 2
    for wanted in ["FAKE", "REAL"]:
        mask = labels_sr == wanted
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        order = cand[np.argsort(-sims[cand])]
        take = order[:half]
        neigh_idx.extend(take.tolist())
    return neigh_idx


@dataclass
class ResultRow:
    target_id: str
    model: str
    prompt_embedding: List[float]
    prompt_token_len_est: int
    selected_neighbors_idx: List[int]
    selected_neighbors_id: List[str]
    selected_neighbors_labels: List[str]
    response_raw: str
    response_label: str
    csv_true_label: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus-parquet",
        type=str,
        required=True,
        help="Caminho para Fake.br_raw.parquet (corpus para few-shot)",
    )
    ap.add_argument(
        "--api-host",
        type=str,
        required=True,
        help="Host do Ollama, ex.:",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["gemma3", "gemma2:9b"],
        help="Modelos Ollama para inferência",
    )
    ap.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text:latest",
        help="Modelo de embedding do Ollama",
    )
    ap.add_argument(
        "--text-col-corpus",
        type=str,
        default="text_no_url",
        help="Coluna de texto no corpus",
    )
    ap.add_argument(
        "--label-col-corpus",
        type=str,
        default="label",
        help="Coluna de rótulo no corpus",
    )
    ap.add_argument(
        "--k", type=int, default=6, help="Número total de exemplos few-shot (par)"
    )
    ap.add_argument(
        "--out-dir", type=str, default="outputs_test_csv", help="Pasta de saída"
    )

    ap.add_argument(
        "--test-csv", type=str, required=True, help="Caminho para o CSV externo"
    )
    ap.add_argument(
        "--test-id-col", type=str, default="ID", help="Nome da coluna ID no CSV"
    )
    ap.add_argument(
        "--test-text-col", type=str, default="text", help="Nome da coluna texto no CSV"
    )
    ap.add_argument(
        "--test-label-col",
        type=str,
        default="Rótulo",
        help="Nome da coluna de rótulo no CSV (opcional)",
    )
    ap.add_argument(
        "--cache-embeddings",
        type=str,
        default=r"outputs_test_csv\cache_corpus_embeddings.parquet",
        help="Parquet com embeddings do CORPUS (colunas: idx, embedding). Não recalcula.",
    )

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # 0) Carrega corpus (Fake.Br)
    corpus_path = getattr(args, "corpus_parquet", None)
    if corpus_path is None:
        corpus_path = getattr(args, "corpus-parquet", None)
    print("[1] Lendo corpus:", corpus_path)
    df_corpus = pd.read_parquet(corpus_path)
    df_corpus = df_corpus.reset_index(names="idx")

    required_corpus_cols = ["idx", args.text_col_corpus, args.label_col_corpus]
    missing = [c for c in required_corpus_cols if c not in df_corpus.columns]
    if missing:
        raise ValueError(
            f"[corpus] Colunas faltantes: {missing}. Colunas: {df_corpus.columns.tolist()}"
        )

    df_corpus["idx"] = df_corpus["idx"].astype(str).map(norm_id)
    df_corpus[args.text_col_corpus] = df_corpus[args.text_col_corpus].astype(str)
    mapping = detect_label_mapping(df_corpus[args.label_col_corpus])
    print("[info] Mapeamento de rótulos no corpus:", mapping)

    # 1) Conecta Ollama (apenas para gerar embedding do PROMPT e do TEXTO-ALVO)
    oc = OllamaClient(host=args.api_host, embed_model=args.embed_model)

    # 2) Embeddings do CORPUS: carregar do parquet informado (sem recalcular)
    cache_path = args.cache_embeddings
    print("[2] Lendo cache de embeddings do corpus:", cache_path)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Arquivo de cache não encontrado: {cache_path}\n"
            "Gere-o previamente na fase de few-shot/treino e aponte aqui via --cache-embeddings."
        )

    cache_df = pd.read_parquet(cache_path)

    # Valida estrutura mínima
    if "idx" not in cache_df.columns or "embedding" not in cache_df.columns:
        raise ValueError(
            f"Cache inválido: precisa conter colunas 'idx' e 'embedding'. Presentes: {list(cache_df.columns)}"
        )

    # Normaliza chaves e ALINHA os embeddings ao df_corpus por 'idx'
    cache_df["idx"] = cache_df["idx"].astype(str).map(norm_id)
    df_corpus["idx"] = df_corpus["idx"].astype(str).map(norm_id)

    join = df_corpus[["idx"]].merge(
        cache_df, on="idx", how="left", validate="one_to_one"
    )
    faltantes = join["embedding"].isna().sum()
    if faltantes > 0:
        # Se o cache foi feito sobre subset diferente, avise claramente
        ids_faltando = df_corpus.loc[join["embedding"].isna(), "idx"].head(5).tolist()
        raise ValueError(
            f"[cache] {faltantes} embeddings ausentes para o corpus atual. "
            f"Exs. de idx faltantes: {ids_faltando}. "
            "Recrie o cache para este corpus, ou ajuste --cache-embeddings."
        )

    # Constrói matriz (n_doc x dim)
    emb_matrix = np.vstack(join["embedding"].map(np.array).values).astype(np.float32)
    print("[ok] Embeddings do corpus carregados e alinhados:", emb_matrix.shape)

    # 3) Carrega CSV externo (targets)
    print("[3] Lendo CSV de alvos:", args.test_csv)
    df_csv = pd.read_csv(args.test_csv)
    required_csv_cols = [args.test_id_col, args.test_text_col]
    missing = [c for c in required_csv_cols if c not in df_csv.columns]
    if missing:
        raise ValueError(
            f"[csv] Colunas faltantes: {missing}. Colunas: {df_csv.columns.tolist()}"
        )
    df_csv[args.test_id_col] = df_csv[args.test_id_col].astype(str).map(norm_id)
    df_csv[args.test_text_col] = df_csv[args.test_text_col].astype(str)
    has_true_label = args.test_label_col in df_csv.columns
    if has_true_label:
        df_csv[args.test_label_col] = df_csv[args.test_label_col].astype(str)

    # 4) Arquivos de saída + retomada
    res_path = os.path.join(args.out_dir, "results_test_csv.parquet")
    neigh_path = os.path.join(args.out_dir, "neighbors_test_csv.parquet")

    if os.path.exists(res_path):
        results_df = pd.read_parquet(res_path)
        results_df["target_id"] = results_df["target_id"].astype(str).map(norm_id)
        results_df["model"] = results_df["model"].astype(str)
        processed = set(zip(results_df["target_id"], results_df["model"]))
        print(
            f"[resume] {len(results_df)} linhas já salvas em results_test_csv.parquet"
        )
    else:
        results_df = pd.DataFrame()
        processed = set()

    if os.path.exists(neigh_path):
        neighbors_df = pd.read_parquet(neigh_path)
        neighbors_df["target_id"] = neighbors_df["target_id"].astype(str).map(norm_id)
        seen_neighbors_ids = set(neighbors_df["target_id"].tolist())
        print(
            f"[resume] {len(neighbors_df)} linhas já salvas em neighbors_test_csv.parquet"
        )
    else:
        neighbors_df = pd.DataFrame()
        seen_neighbors_ids = set()

    # 5) Loop de alvos do CSV
    for i, row in df_csv.iterrows():
        target_id = norm_id(row[args.test_id_col])
        text_target = str(row[args.test_text_col])
        if all((target_id, m) in processed for m in args.models):
            continue

        neighbor_pos = select_neighbors_balance(
            df_corpus=df_corpus,
            emb_matrix=emb_matrix,
            text_target=text_target,
            text_col_corpus=args.text_col_corpus,
            label_col_corpus=args.label_col_corpus,
            mapping=mapping,
            oc=oc,
            k=args.k,
        )
        neighbor_ids = df_corpus.iloc[neighbor_pos]["idx"].astype(str).tolist()
        neighbor_labels = [
            mapping.get(
                df_corpus.iloc[p][args.label_col_corpus],
                str(df_corpus.iloc[p][args.label_col_corpus]),
            )
            for p in neighbor_pos
        ]
        examples = [
            (str(df_corpus.iloc[p][args.text_col_corpus]), neighbor_labels[j])
            for j, p in enumerate(neighbor_pos)
        ]

        prompt = build_fewshot_prompt(examples, text_target)
        prompt_emb = oc.get_embedding(prompt).astype(np.float32).tolist()
        approx_token_len = int(len(prompt) / 3.5)
        csv_label = str(row[args.test_label_col]) if has_true_label else ""

        for model_name in args.models:
            key = (target_id, model_name)
            if key in processed:
                continue

            resp = oc.generate(
                model=model_name, prompt=prompt, temperature=0.0, max_tokens=160
            )
            label = parse_label(resp)

            row_dict = asdict(
                ResultRow(
                    target_id=target_id,
                    model=model_name,
                    prompt_embedding=prompt_emb,
                    prompt_token_len_est=approx_token_len,
                    selected_neighbors_idx=neighbor_pos,
                    selected_neighbors_id=neighbor_ids,
                    selected_neighbors_labels=neighbor_labels,
                    response_raw=resp,
                    response_label=label,
                    csv_true_label=csv_label,
                )
            )
            results_df = pd.concat(
                [results_df, pd.DataFrame([row_dict])], ignore_index=True
            )
            results_df.to_parquet(res_path, index=False)
            processed.add(key)

        if target_id not in seen_neighbors_ids:
            neigh_row = {
                "target_id": target_id,
                "neighbor_positions_full": neighbor_pos,
                "neighbor_ids": neighbor_ids,
                "neighbor_labels": neighbor_labels,
            }
            neighbors_df = pd.concat(
                [neighbors_df, pd.DataFrame([neigh_row])], ignore_index=True
            )
            neighbors_df.to_parquet(neigh_path, index=False)
            seen_neighbors_ids.add(target_id)

        if (i + 1) % 20 == 0:
            print(f"[loop] Targets processados (CSV): {i+1}/{len(df_csv)}")

    print("[ok] Finalizado. Arquivos atualizados:")
    print(" -", res_path)
    print(" -", neigh_path)


if __name__ == "__main__":
    main()
