#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot por recuperação (coseno) + Ollama (checkpointing) — versão ALL
------------------------------------------------------------------------
- Dataset com colunas: ['idx','text_no_url','label','old_split','new_split']
- Seleção automática de exemplos few-shot por similaridade de cosseno (balanceado FAKE/REAL)
- Geração de resposta com modelos do Ollama (loop em gemma3 e gemma2:9b por padrão)
- Embeddings:
    * do CORPUS (text_no_url) para seleção dos vizinhos (nomic-embed-text:latest)
    * do PROMPT few-shot (salvos junto ao resultado)
- Checkpointing:
    * results_fewshot_ollama.parquet atualizado a cada resposta (por (idx, modelo))
    * neighbors.parquet atualizado a cada idx (uma vez por alvo)
    * retomada automática se arquivos existirem (pula o que já foi processado — pode ser ignorado com --ignore-resume)

Uso (exemplo):
    python fewshot_retrieval_ollama_all.py \
      --dataset-path "datasets/portuguese-fact-checking/Fake.br_raw.parquet" \
      --api-host "" \
      --models gemma3 gemma2:9b \
      --embed-model nomic-embed-text:latest \
      --k 6 \
      --max-targets -1 \
      --sample-frac 1.0 \
      --ignore-resume \
      --target-split "" \
      --out-dir outputs_fewshot

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
except Exception:
    Client = None


# -------------------------------
# Utilidades
# -------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def norm_id(x) -> str:
    return str(x).strip()


def find_json_block(text: str) -> str:
    """Extrai primeiro bloco { ... } plausível da saída do LLM."""
    s = str(text)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return ""


def parse_label(raw: str) -> str:
    """Extrai 'FAKE' ou 'REAL' de resposta JSON (com fallback heurístico)."""
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
    return "FAKE"  # default prudente


def clean_text_for_prompt(text: str, max_chars: int = 1200) -> str:
    t = normalize_whitespace(text)
    if len(t) > max_chars:
        t = t[:max_chars] + " [...]"
    return t


def detect_label_mapping(series: pd.Series) -> Dict:
    """
    Mapeia rótulos para {'FAKE','REAL'}.
    Aceita {0,1}, {'FAKE','REAL'} etc. Mantém flexibilidade.
    """
    unique_vals = series.dropna().unique().tolist()
    # Caso binário numérico
    if all(
        isinstance(v, (int, np.integer, np.int64, np.int32)) for v in unique_vals
    ) and set(unique_vals).issubset({0, 1}):
        return {0: "FAKE", 1: "REAL"}
    # Caso string
    inv = {}
    for v in unique_vals:
        s = str(v).strip().lower()
        if "fake" in s:
            inv["FAKE"] = v
        elif "real" in s or "verdad" in s:
            inv["REAL"] = v
    if "FAKE" in inv and "REAL" in inv:
        return {inv["FAKE"]: "FAKE", inv["REAL"]: "REAL"}
    # Fallback: assume binário ordenado
    if len(unique_vals) == 2:
        vals_sorted = sorted(unique_vals, key=lambda x: str(x))
        return {vals_sorted[0]: "FAKE", vals_sorted[1]: "REAL"}
    raise ValueError(f"Rótulos inesperados: {unique_vals}")


# -------------------------------
# Prompt few-shot
# -------------------------------

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
        just = "..."  # justificativa curta placeholder
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


# -------------------------------
# Ollama wrapper
# -------------------------------


class OllamaClient:
    def __init__(self, host: str, embed_model: str):
        if Client is None:
            raise RuntimeError(
                "Pacote 'ollama' não encontrado. Instale com 'pip install ollama'."
            )
        self.client = Client(host=host)
        self.embed_model = embed_model

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Usa a API oficial: client.embeddings(model=..., prompt=text)
        Corrige automaticamente :lates -> :latest se necessário.
        """
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
                f"Resposta inesperada de embedding: {type(resp)} -> {resp}"
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


# -------------------------------
# Seleção por similaridade (balanceada)
# -------------------------------


def select_neighbors_balance(
    df: pd.DataFrame,
    emb_matrix: np.ndarray,
    idx_target: int,
    label_col: str,
    mapping: Dict,
    k: int = 6,
) -> List[int]:
    """
    Seleciona k vizinhos balanceados (k/2 por classe) por similaridade de cosseno.
    Retorna POSIÇÕES (iloc) do df completo (não o valor de df['idx']).
    """
    assert k % 2 == 0, "k deve ser par (k/2 FAKE + k/2 REAL)"
    target_vec = emb_matrix[idx_target : idx_target + 1]
    sims = cosine_similarity(target_vec, emb_matrix)[0]
    sims[idx_target] = -1.0  # evita auto-seleção

    labels = df[label_col].values
    labels_sr = np.array([mapping.get(lab, str(lab)) for lab in labels], dtype=object)

    neigh_idx = []
    for wanted in ["FAKE", "REAL"]:
        mask = labels_sr == wanted
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        order = cand[np.argsort(-sims[cand])]
        take = order[: (k // 2)]
        neigh_idx.extend(take.tolist())
    return neigh_idx


# -------------------------------
# Dataclass de resultado
# -------------------------------


@dataclass
class ResultRow:
    idx: str
    model: str
    prompt_embedding: List[float]
    prompt_token_len_est: int
    selected_neighbors_idx: List[int]
    selected_neighbors_id: List[str]
    selected_neighbors_labels: List[str]
    response_raw: str
    response_label: str


# -------------------------------
# Main
# -------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Caminho para Fake.br_raw.parquet",
    )
    ap.add_argument(
        "--api-host",
        type=str,
        required=True,
        help="Host do Ollama, ex.: """,
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
        "--text-col", type=str, default="text_no_url", help="Coluna de texto"
    )
    ap.add_argument("--label-col", type=str, default="label", help="Coluna de rótulo")
    ap.add_argument(
        "--k", type=int, default=6, help="Número total de exemplos few-shot (par)"
    )
    ap.add_argument(
        "--max-targets",
        type=int,
        default=-1,
        help="Número máximo de amostras alvo (use -1 para todas)",
    )
    ap.add_argument(
        "--target-split",
        type=str,
        default="",
        help="Filtrar por df['new_split'] (ex.: test/val/train). Vazio=sem filtro",
    )
    ap.add_argument(
        "--out-dir", type=str, default="outputs_fewshot", help="Pasta de saída"
    )
    ap.add_argument(
        "--cache-embeddings",
        type=str,
        default="cache_corpus_embeddings.parquet",
        help="Arquivo cache de embeddings do corpus",
    )
    ap.add_argument(
        "--ignore-resume",
        action="store_true",
        help="Ignora resultados anteriores e reprocessa todos os (idx,modelo)",
    )
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Usa fração das amostras-alvo (1.0 = 100%)",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 0) Carrega dataset
    print("[1] Lendo dataset:", args.dataset_path)
    df = pd.read_parquet(args.dataset_path)

    if "idx" not in df.columns:
        df = df.reset_index(names="idx")

    required_cols = ["idx", "text_no_url", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas faltantes no dataset: {missing}. Encontradas: {df.columns.tolist()}"
        )

    df["idx"] = df["idx"].astype(str).map(norm_id)
    df[args.text_col] = df[args.text_col].astype(str)

    # Filtro de split de alvo
    df_targets = df.reset_index(drop=True)

    # (opcional) subsample controlado
    if args.sample_frac is not None and 0 < args.sample_frac < 1.0:
        df_targets = df_targets.sample(
            frac=args.sample_frac, random_state=42
        ).reset_index(drop=True)

    print(f"[info] df total = {len(df)}, df_targets = {len(df_targets)}')")

    # Mapeamento de rótulos
    mapping = detect_label_mapping(df[args.label_col])
    print("[info] Mapeamento de rótulos:", mapping)

    # 1) Conecta Ollama
    oc = OllamaClient(host=args.api_host, embed_model=args.embed_model)

    # 2) Embeddings do corpus (com cache robusto e alinhamento por 'idx')
    res_path = os.path.join(args.out_dir, "results_fewshot_ollama.parquet")
    neigh_path = os.path.join(args.out_dir, "neighbors.parquet")
    cache_path = os.path.join(args.out_dir, args.cache_embeddings)

    # Carrega cache (se existir) e alinha pela chave 'idx'
    emb_matrix = None
    if os.path.isfile(cache_path):
        print("[2] Carregando cache de embeddings do corpus:", cache_path)
        cache_df = pd.read_parquet(cache_path)
        if "idx" not in cache_df.columns or "embedding" not in cache_df.columns:
            print(
                "[warn] Cache inválido. Ignorando e recalculando embeddings do corpus..."
            )
            cache_df = None
        else:
            cache_df["idx"] = cache_df["idx"].astype(str).map(norm_id)
            join = df[["idx"]].merge(
                cache_df, on="idx", how="left", validate="one_to_one"
            )
            if join["embedding"].isna().any():
                print(
                    "[warn] Cache incompleto (faltam embeddings). Recalculando corpus..."
                )
                cache_df = None
            else:
                emb_matrix = np.vstack(join["embedding"].map(np.array).values)

    if emb_matrix is None:
        print("[2] Calculando embeddings do corpus com", args.embed_model)
        emb_list = []
        for i, row in df.iterrows():
            txt = str(row[args.text_col])
            emb = oc.get_embedding(txt)
            emb_list.append(emb.astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  - {i+1}/{len(df)} embeddings prontos")
        cache_df = pd.DataFrame(
            {"idx": df["idx"], "embedding": [e.tolist() for e in emb_list]}
        )
        cache_df.to_parquet(cache_path, index=False)
        print("[ok] Cache salvo em", cache_path)
        emb_matrix = np.vstack(emb_list)

    # 3) Preparar retomada (checkpointing)
    if os.path.exists(res_path) and not args.ignore_resume:
        results_df = pd.read_parquet(res_path)
        results_df["idx"] = results_df["idx"].astype(str).map(norm_id)
        results_df["model"] = results_df["model"].astype(str)
        processed = set(zip(results_df["idx"], results_df["model"]))
        print(
            f"[resume] {len(results_df)} linhas já salvas; pulando pares (idx,modelo) existentes"
        )
    else:
        results_df = pd.DataFrame()
        processed = set()
        if os.path.exists(res_path) and args.ignore_resume:
            print(
                "[resume] ignorado por --ignore-resume: reprocessando todos os pares (idx,modelo)"
            )

    if os.path.exists(neigh_path) and not args.ignore_resume:
        neighbors_df = pd.read_parquet(neigh_path)
        neighbors_df["idx"] = neighbors_df["idx"].astype(str).map(norm_id)
        seen_neighbors_ids = set(neighbors_df["idx"].tolist())
    else:
        neighbors_df = pd.DataFrame()
        seen_neighbors_ids = set()

    # Mapear 'idx' -> posição no df completo
    idx_to_pos_full = {norm_id(df.iloc[i]["idx"]): i for i in range(len(df))}

    # 4) Loop de alvos
    if args.max_targets is None or args.max_targets < 0:
        max_targets = len(df_targets)
    else:
        max_targets = min(args.max_targets, len(df_targets))

    print(f"[setup] max_targets = {max_targets}")
    target_positions = list(range(max_targets))

    for tpos in target_positions:
        row_t = df_targets.iloc[tpos]
        target_id = norm_id(row_t["idx"])
        if not args.ignore_resume and all(
            (target_id, m) in processed for m in args.models
        ):
            continue

        full_pos = idx_to_pos_full[target_id]
        target_text = str(row_t[args.text_col])

        # Seleciona vizinhos (balanceado)
        neighbor_pos = select_neighbors_balance(
            df=df,
            emb_matrix=emb_matrix,
            idx_target=full_pos,
            label_col=args.label_col,
            mapping=mapping,
            k=args.k,
        )
        neighbor_ids = df.iloc[neighbor_pos]["idx"].astype(str).tolist()
        neighbor_labels = [
            mapping.get(df.iloc[p][args.label_col], str(df.iloc[p][args.label_col]))
            for p in neighbor_pos
        ]

        # Monta few-shot e embedding do prompt
        examples = [
            (str(df.iloc[p][args.text_col]), neighbor_labels[i])
            for i, p in enumerate(neighbor_pos)
        ]
        prompt = build_fewshot_prompt(examples, target_text)
        prompt_emb = oc.get_embedding(prompt).astype(np.float32).tolist()
        approx_token_len = int(len(prompt) / 3.5)

        # (A) Gera para cada modelo, salvando logo após cada resposta
        for model_name in args.models:
            key = (target_id, model_name)
            if not args.ignore_resume and key in processed:
                continue

            resp = oc.generate(
                model=model_name, prompt=prompt, temperature=0.0, max_tokens=160
            )
            label = parse_label(resp)

            row_dict = asdict(
                ResultRow(
                    idx=target_id,
                    model=model_name,
                    prompt_embedding=prompt_emb,
                    prompt_token_len_est=approx_token_len,
                    selected_neighbors_idx=neighbor_pos,
                    selected_neighbors_id=neighbor_ids,
                    selected_neighbors_labels=neighbor_labels,
                    response_raw=resp,
                    response_label=label,
                )
            )

            results_df = pd.concat(
                [results_df, pd.DataFrame([row_dict])], ignore_index=True
            )
            results_df.to_parquet(res_path, index=False)
            processed.add(key)

        # (B) Salva vizinhos para este idx (apenas 1 vez)
        if target_id not in seen_neighbors_ids:
            neigh_row = {
                "idx": target_id,
                "target_position_in_split": tpos,
                "neighbor_positions_full": neighbor_pos,
                "neighbor_ids": neighbor_ids,
                "neighbor_labels": neighbor_labels,
            }
            neighbors_df = pd.concat(
                [neighbors_df, pd.DataFrame([neigh_row])], ignore_index=True
            )
            neighbors_df.to_parquet(neigh_path, index=False)
            seen_neighbors_ids.add(target_id)

        if (tpos + 1) % 20 == 0 or (tpos + 1) == max_targets:
            print(f"[loop] Targets processados: {tpos+1}/{max_targets}")

    print("[ok] Finalizado. Arquivos atualizados:")
    print(" -", res_path)
    print(" -", neigh_path)
    print(" -", cache_path)


if __name__ == "__main__":
    main()
