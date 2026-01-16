#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import json
import gc
import torch

from small_text import (
    PoolBasedActiveLearner,
    PredictionEntropy,
    CategoryVectorInconsistencyAndRanking,
    random_initialization,
    TransformersDataset,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    TextDataset,
    list_to_csr
)

from transformers import AutoTokenizer

from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory

import mlflow

import sys
sys.path.append("./path/active-learning-synthetic-validation")

from helper.mlflow_logging import ActiveLearningMLflowLogger
from helper.helper import compute_metrics

# ML-FLOW Configuration

with open("./path/active-learning-synthetic-validation/mlflow.json", 'r') as file:
    data = json.load(file)
    os.environ['MLFLOW_TRACKING_USERNAME'] = data["MLFLOW_TRACKING_USERNAME"]
    os.environ['MLFLOW_TRACKING_PASSWORD'] = data["MLFLOW_TRACKING_PASSWORD"]

mlflow.set_tracking_uri("mlflow_url")

# -------------------------------------------------------------------
# EXPERIMENT SETTINGS
# -------------------------------------------------------------------

BASELINE_MODEL = "google-bert/bert-base-multilingual-cased"

SETFIT_MODELS = [
    "intfloat/multilingual-e5-small", # 0.1B
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 0.1B
    "intfloat/multilingual-e5-base", # 0.3B
]

QUERY_STRATEGIES = [
    PredictionEntropy(),
    CategoryVectorInconsistencyAndRanking()
]

SETFIT_HEAD_VARIANTS = [
    ("diff_e2e",     {"use_differentiable_head": True,  "end_to_end": True}),
    ("diff_no_e2e",  {"use_differentiable_head": True,  "end_to_end": False}),
    ("default",      {"use_differentiable_head": False, "end_to_end": False}),
]

INITIAL_LABELS = 20
BATCH_SIZE = 25
MAX_ITER = 5

TEXT_COL = "childPart"
LABEL_COL = "label"

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

train_df = pd.read_pickle("./data/train_df_new_interviews.pkl")
train_df = train_df[train_df['annotator'] != ''].copy()
train_df.reset_index(drop=True)
test_df  = pd.read_pickle("./data/test_df_new_interviews.pkl")
test_df  = test_df[test_df['peter'].notna()]
test_df['label']  = test_df['peter']

codes_health = ["physical_health", "mental_health", "daily_functioning", "health_unspecific", "health_none"]
codes_freq = ["freq_mentioned", "freq_not_mentioned"]
codes_neg = ["health_none", "freq_not_mentioned"]

codes_combined = codes_health + codes_freq
code_full = {label: idx for idx, label in enumerate(codes_combined)}
code_pos = {k: v for k, v in code_full.items() if k not in codes_neg}
code_neg = {k: v for k, v in code_full.items() if k in codes_neg}
code_pos_new = {i: k for k, i in enumerate(code_pos.keys())}

full_to_pos = {
    code_full[label]: new_idx
    for new_idx, label in enumerate(code_pos.keys())
}

NUM_CLASSES = len(code_pos)

train_setfit = TextDataset.from_arrays(
        list(train_df[TEXT_COL]),
        list_to_csr(list(train_df[LABEL_COL]), (len(train_df), len(code_pos))),
        target_labels=np.arange(len(code_pos))
    )

# -------------------------------------------------------------------
# ACTIVE LEARNING LOOP
# -------------------------------------------------------------------

def active_learning_run(model_name, model_factory, query_strategy, train, experiment_name, setfit=True):
    logger = ActiveLearningMLflowLogger("setfit_silicon_interviews")

    logger.start_experiment(
        run_name=experiment_name,
        model_name=model_name,
        model_args=model_factory,
        clf_factory=model_factory,
        query_strategy=query_strategy,
        dataset_info={"n_train": len(train_df),
                      "n_test": len(test_df),
                      'train_columns': train_df.columns.tolist(),
                      'test_columns': test_df.columns.tolist()
                      },
        train_labeled_by = "peter",
    )

    learner = PoolBasedActiveLearner(
        model_factory,
        query_strategy,
        train
    )

    np.random.seed(2025)
    initial_indices = random_initialization(train, INITIAL_LABELS)
    learner.initialize(initial_indices)

    for it in range(MAX_ITER):
        start = time.time()

        query_indices = learner.query(num_samples=BATCH_SIZE)
        y = train.y[query_indices]

        learner.update(y)

        train_time = time.time() - start

        gc.collect()
        torch.cuda.empty_cache()

        report, _, y_pred_proba, _ = compute_metrics(learner, test_df, code_pos_new)

        logger.log_iteration(
            iteration=it,
            classification_report_dict=report,
            num_labeled_samples=len(learner.indices_labeled),
            indices_labeled=query_indices,
            training_duration=train_time,
            train=train_df,
            y_prediction_proba=y_pred_proba,
            #model=learner
        )

    logger.end_experiment()

    del learner
    del logger

    gc.collect()
    torch.cuda.empty_cache()

# -------------------------------------------------------------------
# 1) BASELINE TRANSFORMER (Not SetFit)
# -------------------------------------------------------------------


tokenizer_kwargs = {'clean_up_tokenization_spaces': True}
tokenizer = AutoTokenizer.from_pretrained(
    BASELINE_MODEL,
    **tokenizer_kwargs
)

train_baseline = TransformersDataset.from_arrays(list(train_df[TEXT_COL]),
                                        list_to_csr(list(train_df[LABEL_COL]), (len(train_df), len(code_pos))),
                                        tokenizer,
                                        max_length=256,
                                        target_labels=np.arange(NUM_CLASSES))
test_baseline = TransformersDataset.from_arrays(list(test_df[TEXT_COL]),
                                        list_to_csr(list(test_df[LABEL_COL]), (len(test_df), len(code_pos))),
                                        tokenizer,
                                        max_length=256,
                                        target_labels=np.arange(NUM_CLASSES))

transformer_model = TransformerModelArguments(
    BASELINE_MODEL, 
    tokenizer_kwargs=tokenizer_kwargs)

baseline_factory = TransformerBasedClassificationFactory(
        transformer_model,
        NUM_CLASSES,
        classification_kwargs=dict({'device': 'cuda',
                                    'multi_label': True,
                                    'mini_batch_size': 8})    
)

EXP_NAME = f"baseline_{BASELINE_MODEL.split('/')[-1]}"

active_learning_run(
    model_name=BASELINE_MODEL,
    model_factory=baseline_factory,
    train=train_baseline,
    query_strategy=CategoryVectorInconsistencyAndRanking(),
    experiment_name=EXP_NAME
)


SKIP_EXPERIMENTS = {
#    "setfit_multilingual-e5-base__diff_e2e__PredictionEntropy",
#    "setfit_multilingual-e5-base__diff_e2e__CategoryVectorInconsistencyAndRanking",
#    "setfit_multilingual-e5-base__diff_no_e2e__PredictionEntropy",
}

# -------------------------------------------------------------------
# 2) SETFIT MODELS
# -------------------------------------------------------------------
for model_name in SETFIT_MODELS:
    for head_name, head_cfg in SETFIT_HEAD_VARIANTS:

        # sanity rule: end_to_end only with differentiable head
        if head_cfg["end_to_end"] and not head_cfg["use_differentiable_head"]:
            continue

        for qs in QUERY_STRATEGIES:
            
            EXP_NAME = (f"setfit_{model_name.split('/')[-1]}__{head_name}__{qs.__class__.__name__}")
        
            if EXP_NAME in SKIP_EXPERIMENTS:
                print(f"⏭️ Skipping completed experiment: {EXP_NAME}")
                continue

            args = SetFitModelArguments(
                model_name,
                end_to_end=head_cfg["end_to_end"],
                max_length=256,
                mini_batch_size=8,
            )

            classification_kwargs = {
                "device": "cuda",
                "multi_label": True,
                "use_differentiable_head": head_cfg["use_differentiable_head"],
            }
            
            factory = SetFitClassificationFactory(
                args,
                NUM_CLASSES,
                classification_kwargs=classification_kwargs,
            )

            active_learning_run(
                model_name=model_name,
                model_factory=factory,
                train=train_setfit,
                query_strategy=qs,
                experiment_name=EXP_NAME,
            )

            del factory
            gc.collect()
            torch.cuda.empty_cache()