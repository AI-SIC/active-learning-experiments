#!/usr/bin/env python
# coding: utf-8

"""
MLflow Logger for Active Learning Experiments
Separate logging module that doesn't interfere with the active learning loop
"""

import mlflow
import json
import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime
import numpy as np
import tempfile
import os
import pickle
import shutil


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary for MLflow parameter logging.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        elif isinstance(v, (str, int, float, bool, type(None))):
            items.append((new_key, v))
        else:
            items.append((new_key, str(v)))

    return dict(items)


class ActiveLearningMLflowLogger:
    """
    MLflow logger for active learning experiments.
    Supports:
      - SetFit (SetFitModelArguments, SetFitClassificationFactory)
      - small-text transformer integrations
    """

    def __init__(self, experiment_name="active_learning"):
        self.experiment_name = experiment_name
        self.parent_run_id = None
        self.parent_run = None
        self.artifact_dir = None

        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        run_name=None,
        model_name=None,
        model_args=None,
        clf_factory=None,
        query_strategy=None,
        dataset_info=None,
        train_labeled_by=None,
        tags=None,
    ):

        if run_name is None:
            run_name = f"al_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.parent_run = mlflow.start_run(run_name=run_name)
        self.parent_run_id = self.parent_run.info.run_id

        self.artifact_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")

        # -------------------------
        # Base parameters
        # -------------------------
        if model_name is not None:
            mlflow.log_param("model_name", model_name)

        if query_strategy is not None:
            mlflow.log_param("query_strategy", query_strategy.__class__.__name__)

        if train_labeled_by is not None:
            mlflow.log_param("train_labeled_by", train_labeled_by)

        # -------------------------
        # Model args & factory
        # -------------------------
        if model_args is not None:
            self._log_model_args(model_args)

        if clf_factory is not None:
            self._log_clf_factory(clf_factory)

        # -------------------------
        # Dataset info
        # -------------------------
        if dataset_info is not None:
            for key, value in dataset_info.items():
                if isinstance(value, (list, tuple)) and key.endswith("_columns"):
                    mlflow.log_param(f"dataset.{key}", ",".join(map(str, value)))
                else:
                    mlflow.log_param(f"dataset.{key}", value)

        # -------------------------
        # Tags
        # -------------------------
        mlflow.set_tag("experiment_type", "active_learning")
        mlflow.set_tag("framework", "small-text")

        if model_args is not None:
            mlflow.set_tag("model_args_class", model_args.__class__.__name__)

        if clf_factory is not None:
            factory_name = clf_factory.__class__.__name__
            mlflow.set_tag("clf_factory_class", factory_name)

            if "SetFit" in factory_name:
                mlflow.set_tag("model_type", "setfit")
            elif "Transformer" in factory_name:
                mlflow.set_tag("model_type", "transformer")
            else:
                mlflow.set_tag("model_type", "unknown")

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        print(f"Started MLflow experiment: {self.experiment_name}")
        print(f"Run name: {run_name}")
        print(f"Run ID: {self.parent_run_id}")

        return self.parent_run

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_model_args(self, model_args):
        """Log model arguments (SetFit or Transformer) safely"""
        try:
            model_args_dict = vars(model_args)
        except TypeError:
            print("Warning: model_args has no __dict__, skipping.")
            return

        flattened = flatten_dict(model_args_dict, parent_key="model_args")

        for key, value in flattened.items():
            try:
                if isinstance(value, (str, int, float, bool, type(None))):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            except Exception as e:
                print(f"Warning: Could not log param {key}: {e}")

    def _log_clf_factory(self, clf_factory):
        """Log classifier factory configuration generically"""
        clf_factory_dict = {}

        if hasattr(clf_factory, "num_classes"):
            clf_factory_dict["num_classes"] = clf_factory.num_classes

        if hasattr(clf_factory, "classification_kwargs"):
            clf_factory_dict["classification_kwargs"] = clf_factory.classification_kwargs

        if hasattr(clf_factory, "model_args"):
            clf_factory_dict["model_args_class"] = clf_factory.model_args.__class__.__name__

        flattened = flatten_dict(clf_factory_dict, parent_key="clf_factory")

        for key, value in flattened.items():
            try:
                if isinstance(value, (str, int, float, bool, type(None))):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            except Exception as e:
                print(f"Warning: Could not log param {key}: {e}")

    def _log_labeled_data(self, train_df, indices_labeled, prefix="labeled_data"):
        if self.artifact_dir is None:
            return

        labeled_df = train_df.iloc[indices_labeled]

        csv_path = os.path.join(self.artifact_dir, f"{prefix}.csv")
        labeled_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="labeled_data")

        json_path = os.path.join(self.artifact_dir, f"{prefix}.json")
        labeled_df.to_json(json_path, orient="records", indent=2)
        mlflow.log_artifact(json_path, artifact_path="labeled_data")

    # ------------------------------------------------------------------
    # Iteration logging
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        iteration,
        classification_report_dict,
        num_labeled_samples,
        indices_labeled,
        training_duration=None,
        train=None,
        additional_metrics=None,
        y_prediction_proba=None,
        model=None,
    ):

        if self.parent_run_id is None:
            raise RuntimeError("Experiment not started")

        with mlflow.start_run(run_name=f"iteration_{iteration}", nested=True):

            mlflow.log_param("iteration", iteration)
            mlflow.log_param("num_labeled_samples", num_labeled_samples)
            mlflow.log_param("indices_labeled", indices_labeled)

            if training_duration is not None:
                mlflow.log_param("training_duration", training_duration)

            # Per-class metrics
            for cls, metrics in classification_report_dict.items():
                if isinstance(metrics, dict):
                    for m, v in metrics.items():
                        mlflow.log_metric(f"{cls}_{m}", v, step=iteration)

            # Aggregate metrics
            if "accuracy" in classification_report_dict:
                mlflow.log_metric("accuracy", classification_report_dict["accuracy"], step=iteration)

            for avg in ["macro avg", "weighted avg"]:
                if avg in classification_report_dict:
                    mlflow.log_metric(f"{avg.replace(' ', '_')}_f1",
                                      classification_report_dict[avg]["f1-score"],
                                      step=iteration)

            if additional_metrics:
                for k, v in additional_metrics.items():
                    mlflow.log_metric(k, v, step=iteration)

            # Save model
            if model is not None:
                try:
                    path = os.path.join(self.artifact_dir, f"model_iter_{iteration}")
                    model.save(path)
                    mlflow.log_artifact(path, artifact_path=f"model_iter_{iteration}")
                except Exception as e:
                    print(f"Warning: could not save model: {e}")

            # Save report
            report_path = os.path.join(self.artifact_dir, f"classification_report_{iteration}.csv")
            pd.DataFrame(classification_report_dict).transpose().to_csv(report_path)
            mlflow.log_artifact(report_path)

            if train is not None:
                self._log_labeled_data(train, indices_labeled)

            if y_prediction_proba is not None:
                pred_path = os.path.join(self.artifact_dir, f"prediction_proba_{iteration}.pkl")
                with open(pred_path, "wb") as f:
                    pickle.dump(y_prediction_proba, f)
                mlflow.log_artifact(pred_path)

            print(f"Logged iteration {iteration}")

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def end_experiment(self):
        if self.parent_run_id is None:
            return

        mlflow.end_run()

        if self.artifact_dir and os.path.exists(self.artifact_dir):
            shutil.rmtree(self.artifact_dir)

        self.parent_run_id = None
        self.parent_run = None
        self.artifact_dir = None

        print("Ended MLflow experiment")
