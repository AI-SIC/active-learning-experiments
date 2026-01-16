from small_text import (
    BreakingTies,
    CategoryVectorInconsistencyAndRanking,
    PoolBasedActiveLearner,
    SubsamplingQueryStrategy,
    TextDataset,
    TransformersDataset,
    list_to_csr,
    random_initialization_balanced,
)
from transformers import AutoTokenizer
from IPython.display import HTML
from captum.attr import IntegratedGradients
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import Normalizer
from torch import nn
import ast
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import torch
import pandas as pd

# Used to Explain predictions with Integrated Gradients
class STForIG(nn.Module):
    """
    Wrapper that:
      - looks up token embeddings,
      - runs the HF encoder with inputs_embeds,
      - applies ST mean-pooling,
      - applies the differentiable SetFit head,
      - returns ONLY logits for Captum.
    """
    def __init__(self, st_body, torch_head):
        super().__init__()
        # st_body is a SentenceTransformer
        self.encoder = st_body[0].auto_model            # HF encoder (e.g., BERT)
        self.tokenizer = st_body[0].tokenizer           # HF tokenizer
        self.embedding = self.encoder.get_input_embeddings()
        self.head = torch_head                          # SetFitHead (torch), not sklearn

    def forward(self, input_embeds, attention_mask):
        # Run the transformer with inputs_embeds so IG works in embedding space
        outputs = self.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state

        # SentenceTransformers mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sentence_embeddings = sentence_embeddings / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        logits = self.head(sentence_embeddings)  # [batch, num_labels]
        return logits  # IMPORTANT: return ONLY logits for Captum

def explain(text: str, label_idx: int, wrapped, tokenizer, device, n_steps: int = 50):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        input_embeds = wrapped.embedding(input_ids)

    def forward_for_ig(input_embeds, attention_mask):
        out = wrapped(input_embeds, attention_mask)
        if isinstance(out, tuple):
            out = out[0]
        return out

    ig = IntegratedGradients(forward_for_ig)
    attributions, _ = ig.attribute(
        inputs=input_embeds,
        baselines=torch.zeros_like(input_embeds),
        additional_forward_args=(attention_mask,),
        target=label_idx,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist(), skip_special_tokens=False)
    return tokens, token_scores

def parse_interview_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

        interview_id = os.path.basename(filepath).replace('.json', '')
        conversation = data.get("conversation", [])

        rows = []
        for i in range(0, len(conversation), 2):
            rowInterviewer = conversation[i]
            rowChild = conversation[i+1] if i+1 < len(conversation) else None

            rows.append({
                "interview_id": interview_id,
                "speech_index": i,
                "childPart":rowChild.get("content", ""),
                "control_setting": re.search(r"\d{3}", filepath).group(),
                "label": [],  # to be labeled later
            })

    return pd.DataFrame(rows)

def parse_interview_with_turns(filepath, turns_df):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

        interview_id = os.path.basename(filepath)
        meta = data.get("metadata", [])
        conversation = data.get("conversation", [])

        all_dfs = []
        
        for i in range(0, len(conversation), 2):

            # extracts the turns for the relevant questions to filter these questions only
            question_ids = list(turns_df[turns_df['interview_id'] == interview_id].iloc[0][1:])

            if i / 2 in question_ids: 
            
                row = []
                
                rowInterviewer = conversation[i]
                rowChild = conversation[i+1] if i+1 < len(conversation) else None 
                
                interview_before = f"""<span style="color: lightgrey;">{''.join(
                    part.get("role", "") + ":<br>" + part.get("content", "") + "<br>"
                    for j, part in enumerate(conversation)
                    if j < i
                )}</span>
                """
                
                interview_at_point = f"""<b>{rowInterviewer.get("role", "")}:</b><br>
                    {rowInterviewer.get("content", "")}<br>
                    <b>{rowChild.get("role", "")}:</b><br>
                    {rowChild.get("content", "")}<br>
                    """
                
                interview_after = f"""<span style="color: lightgrey;">{''.join(
                    part.get("role", "") + ":<br>" + part.get("content", "") + "<br>"
                    for j, part in enumerate(conversation)
                    if j > i+1
                )}</span>"""
        
                interview = interview_before + interview_at_point + interview_after

                row.append({
                    "interview_id": interview_id,
                    "speech_index": i,
                    "childPart":rowChild.get("content", ""),
                    "interview": interview,
                    "label": [],  # to be labeled later
                })
    
                row_df = pd.DataFrame(row)
                all_dfs.append(pd.DataFrame([meta]).join(row_df))
            
        composed = pd.concat(all_dfs, ignore_index=True)

    return composed

def cluster_embeddings(df,
                       embed_col='embedding',
                       fixed_k=8,
                       pca_variance=0.95):
    emb = np.stack(df[embed_col].values)
    emb_norm = Normalizer('l2').fit_transform(emb)
    pca_full = PCA(n_components=pca_variance, random_state=42)
    emb_red = pca_full.fit_transform(emb_norm)
    mbkm = MiniBatchKMeans(n_clusters=fixed_k,
                           batch_size=256,
                           random_state=42,
                           n_init='auto')
    labels = mbkm.fit_predict(emb_red)
    pca_2d = PCA(n_components=2, random_state=42)
    pcs_2d = pca_2d.fit_transform(emb_norm)
    out = df.copy()
    out['cluster_id'] = labels
    out['pc1'] = pcs_2d[:, 0]
    out['pc2'] = pcs_2d[:, 1]
    return out


def compute_metrics(active_learner, test_df, codebook, tokenizer=None, max_length=256):
    """
    Works for:
      • SetFit classifiers
      • Transformer-based classifiers

    Returns:
      • classification report (dict)
      • predictions (CSR)
      • probabilities (np.ndarray)
      • enriched test_df
    """

    num_classes = len(codebook)

    y_true = list_to_csr(
        test_df["label"].tolist(),
        (len(test_df), num_classes),
    )

    clf = active_learner.classifier

    # 1) Build correct dataset type
    if hasattr(clf, "model") and hasattr(clf, "tokenizer"):
        # TransformerBasedClassifier
        test_dataset = TransformersDataset.from_arrays(
            test_df["childPart"].tolist(),
            y_true,
            clf.tokenizer if tokenizer is None else tokenizer,
            max_length=max_length,
            target_labels=np.arange(num_classes),
        )
    else:
        # SetFitClassifier
        test_dataset = TextDataset.from_arrays(
            test_df["childPart"].tolist(),
            y_true,
            target_labels=np.arange(num_classes),
        )

    # 2) Predictions
    y_pred_sparse = clf.predict(test_dataset)

    y_pred_list = [
        [int(idx) for idx in row.indices]
        for row in y_pred_sparse
    ]

    enriched_test_df = test_df.copy()
    enriched_test_df["y_pred"] = y_pred_list

    # 3) Probabilities
    probs = clf.predict_proba(test_dataset)

    class_names = list(codebook.keys())
    for i, cls in enumerate(class_names):
        enriched_test_df[f"prob_{cls}"] = probs[:, i]

    # 4) Metrics
    report = classification_report(
        y_true,
        y_pred_sparse,
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_sparse, zero_division=0))

    return report, y_pred_sparse, probs, enriched_test_df

def show_html(tokens, scores, class_id):
    # Normalize symmetrically around 0 to preserve positive/negative
    max_val = np.max(np.abs(scores)) + 1e-9
    norm_scores = scores / max_val  # now in [-1, 1]

    html_tokens = []
    for tok, val in zip(tokens, norm_scores):
        if val > 0:
            color = f"rgba(0,255,0,{abs(val)})"   # green
        elif val < 0:
            color = f"rgba(255,0,0,{abs(val)})"   # red
        else:
            color = "rgba(255,255,255,1)"         # white

        html_tokens.append(
            f"<span style='background-color:{color}; padding:1px 3px; margin:1px'>{tok}</span>"
        )

    display(HTML(f"Integrated Gradient contribution for <b>predicted</b> class {class_id}:<br>" + " ".join(html_tokens)))

def compute_classwise_confusion(active_learner, test_df):

    num_classes = active_learner.classifier.num_classes

    test = TextDataset.from_arrays(
        test_df.childPart.tolist(),
        list_to_csr(test_df.label.tolist(), (len(test_df), num_classes)), # y
        target_labels=(np.arange(num_classes))
    )

    y_pred_test = active_learner.classifier.predict(test)

    # Compute confusion matrix
    cm = multilabel_confusion_matrix(test.y.toarray(), y_pred_test)

    # Plot using seaborn
    i = 0  # class index to plot
    for i in range(num_classes):
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not class '+str(i), 'Class '+str(i)],
            yticklabels=['Not class '+str(i), 'Class '+str(i)])
        plt.title(f'Confusion matrix for class {i}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

def show_prediction_examples(active_learner, enriched_test_df, n_show):

    sample_rows = enriched_test_df.sample(n=n_show, random_state=2025)

    for i, row in sample_rows.iterrows():
        print(f"------id: {i}------------------------------------------------------")
        print(f"Speech part: {row['childPart']}")
        #print(f"\033[1mSetting:\033[0m \033[94m{row['setting']}\033[0m")  # bold + blue
        print(f"True label: {row['label']}")
        print(f"Pred label: {row['y_pred']}\n")

def show_ig_contributions(active_learner, enriched_test_df, i):
    classifier = active_learner.classifier
    st_body = classifier.model.model_body
    torch_head = classifier.model.model_head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped = STForIG(st_body, torch_head).to(device)
    tokenizer = wrapped.tokenizer

    # predicted labels for this sample
    pred_labels = enriched_test_df.iloc[i].y_pred

    for label_idx in pred_labels:
        tokens, scores = explain(
            enriched_test_df.iloc[i].childPart,
            int(label_idx),
            wrapped,
            tokenizer,
            device
        )
        show_html(tokens, scores, label_idx)

# def evaluate(active_learner, test_df, train_df, codebook,
#              show_confusion=False, show_clustering=False,
#              show_prediction_examples=False):

#     num_classes = len(codebook)
#     indices_labeled = active_learner.indices_labeled

#     test = TextDataset.from_arrays(
#         test_df.childPart.tolist(),
#         list_to_csr(test_df.label.tolist(), (len(test_df), num_classes)), # y
#         target_labels=(np.arange(num_classes))
#     )

#     y_pred_test = active_learner.classifier.predict(test)

#     # Generate classification report
#     report = classification_report(test.y, y_pred_test, output_dict=True)  # Dict format for easy plotting

#     print('Classification Report:')
#     print(classification_report(test.y, y_pred_test))  # Print readable version

#     probs =  active_learner.classifier.predict_proba(test)
#     codes = {v: k for k, v in codebook.items()}

#     y_pred_test_list = []
#     for i in range(len(test)):
#         curr = list(y_pred_test[i].toarray().nonzero()[1])
#         y_pred_test_list.append(curr)

#     test_df['y_pred'] = y_pred_test_list

#     if show_clustering:
#         #train_filter = train_df[train_df['label'].apply(lambda v: bool(len(v)))]
#         #joined_df = pd.concat([train_filter, test_df], ignore_index=True)
#         #joined_df = cluster_embeddings(joined_df, embed_col='embedding', fixed_k=6)
#         test_df = cluster_embeddings(test_df, embed_col='embedding', fixed_k = 5)
#         test_df['binary_error'] =  test_df['label'] == test_df['y_pred']

#         fig, ax = plt.subplots()
#         sns.scatterplot(data=test_df,
#                         x='pc1', y='pc2',
#                         hue='binary_error',
#                         #palette=palette_split,
#                         style='cluster_id',
#                         markers=True,
#                         s=80,
#                         edgecolor='black',
#                         linewidth=0.5,
#                         ax=ax)

#         #for _, row in joined_df.iterrows():
#         #    txt = str(row['label']).replace('[', '').replace(']', '')
#         #    ax.text(row['pc1'] + 0.02, row['pc2'] + 0.02,
#         #            txt,
#         #            fontsize=8,
#         #            ha='left',
#         #            va='bottom',
#         #            alpha=0.7)

#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles=handles, labels=labels,
#                   title='Legend',
#                   bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.set_title('PCA for test data – binary error (colour) / cluster (shape)')
#         ax.set_xlabel('PC 1')
#         ax.set_ylabel('PC 2')
#         plt.tight_layout()
#         plt.show()

#     return report  # Return as dict for later plotting

def plot_class_metrics_over_time(all_reports, classes_to_track=None):
    """
    Plots precision, recall, and f1-score over time for each class in its own subplot.
    
    Parameters:
    - all_reports: List of classification report dicts (output from classification_report(..., output_dict=True))
    - classes_to_track: List of class labels (as strings). If None, inferred from first report.
    """

    # Auto-detect class labels if not provided
    if classes_to_track is None:
        classes_to_track = [k for k in all_reports[0].keys() if k.isdigit()]

    codes = {v: k for k, v in code_pos_new.items()}
    metrics = ['precision', 'recall', 'f1-score']
    iterations = list(range(len(all_reports)))

    num_classes = len(classes_to_track)
    cols = 3  # number of subplots per row
    rows = (num_classes + cols - 1) // cols  # ceil division for rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Class-wise Metrics over Time", fontsize=16)

    for idx, cls in enumerate(classes_to_track):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        for metric in metrics:
            values = [report[cls][metric] for report in all_reports]
            ax.plot(iterations, values, marker='o', label=metric)

        ax.set_title(f"Class: {codes[idx]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for i in range(num_classes, rows * cols):
        fig.delaxes(axes[i // cols][i % cols])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def get_labels_with_annotators(records, code_neg, code_full):

    results = []

    for record in records:

        annotators = record["health.responses.users"]
        health_lists = record["health.responses"]
        freq_labels = record["frequency.responses"]

        record_output = []

        for user, h_list, f in zip(annotators, health_lists, freq_labels):

            # merge
            labels = list(h_list) + [f]

            # remove negatives
            labels = [lbl for lbl in labels if lbl not in code_neg]

            # name → full index
            full_idx = [code_full[lbl] for lbl in labels]

            # full → positive continuous
            pos_idx = [full_to_pos[i] for i in full_idx if i in full_to_pos]

            record_output.append({
                "annotator": user,
                "labels": pos_idx
            })

        results.append(record_output)

    return results

__all__ = [name for name in globals()
           if callable(globals()[name]) and not name.startswith("_")]
