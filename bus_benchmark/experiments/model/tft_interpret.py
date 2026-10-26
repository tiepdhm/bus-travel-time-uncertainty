import numpy as np
import torch
import matplotlib.pyplot as plt

def interpret_tft_attention_components(
    attention_components_list: list[dict[str, torch.Tensor]],
    enc_steps: int
) -> dict[str, torch.Tensor]:
    """
    Aggregates a list of attention_components dictionaries (e.g. from dataloader batches)
    into a single dictionary of tensors averaged over the dataset.

    Expected keys in each attention_components dict:
        'decoder_self_attn': (head, B, T, T)
        'static_flags': (B, num_static, 1) or None
        'historical_flags': (B, T_enc, 1, num_hist)
        'future_flags': (B, T_dec, 1, num_future)

    Returns:
        dict with:
            'attention': (T_enc + 1,) attention weights from the first decoder step to the history
            'static_variables': (num_static,) or empty
            'historical_variables': (num_hist,)
            'future_variables': (num_future,)
    """
    if not attention_components_list:
        return {}
    
    # 1. decoder_self_attn
    all_self_attn = torch.cat([comp['decoder_self_attn'] for comp in attention_components_list], dim=1) # dim 1 is B
    # Average over heads
    attn = all_self_attn.mean(dim=0)  # (B, T, T)
    # Average over batch
    attn = attn.mean(dim=0)           # (T, T)
    # Extract the row corresponding to the first decoder step (index = enc_steps)
    # It attends to steps 0 to enc_steps.
    attention_to_history = attn[enc_steps, :enc_steps + 1]

    # 2. static_flags
    if attention_components_list[0].get('static_flags') is not None:
        all_static = torch.cat([comp['static_flags'] for comp in attention_components_list], dim=0)
        static_variables = all_static.mean(dim=0).squeeze(-1) # (num_static,)
    else:
        static_variables = torch.tensor([])

    # 3. historical_flags
    all_hist = torch.cat([comp['historical_flags'] for comp in attention_components_list], dim=0) # (B, T_enc, 1, num_hist)
    # Average over batch and time
    historical_variables = all_hist.mean(dim=(0, 1)).squeeze(0) # (num_hist,)

    # 4. future_flags
    all_future = torch.cat([comp['future_flags'] for comp in attention_components_list], dim=0) # (B, T_dec, 1, num_future)
    # Average over batch and time
    future_variables = all_future.mean(dim=(0, 1)).squeeze(0) # (num_future,)

    return {
        "attention": attention_to_history.cpu(),
        "static_variables": static_variables.cpu(),
        "historical_variables": historical_variables.cpu(),
        "future_variables": future_variables.cpu(),
    }


def plot_tft_interpretation(
    interpretation: dict[str, torch.Tensor],
    enc_steps: int,
    static_labels: list[str],
    historical_labels: list[str],
    future_labels: list[str],
) -> dict[str, plt.Figure]:
    """
    Plots the interpreted weights into matplotlib Figures.

    Returns:
        dict of figures with keys 'attention', 'static_variables', 'historical_variables', 'future_variables'.
    """
    figs = {}

    # 1. Attention
    fig, ax = plt.subplots(figsize=(8, 4))
    attention = interpretation["attention"].numpy()
    # Normalize just in case
    attention = attention / attention.sum()
    time_indices = np.arange(-enc_steps, 1) # from -enc_steps to 0

    ax.plot(time_indices, attention, marker='o')
    ax.set_xlabel("Time index relative to prediction")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Attention (First Prediction Step vs History)")
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    figs["attention"] = fig

    def make_selection_plot(title: str, values: np.ndarray, labels: list[str]):
        if len(values) == 0:
            return None
        
        # Determine scaling for plot height based on number of labels
        # Adjust small height so big labels don't get squished
        height = max(4, len(labels) * 0.25)
        
        fig, ax = plt.subplots(figsize=(8, height))
        order = np.argsort(values)
        values = values / values.sum(-1, keepdims=True)
        ax.barh(
            np.arange(len(values)),
            values[order] * 100,
            tick_label=np.asarray(labels)[order],
        )
        ax.set_title(title)
        ax.set_xlabel("Importance in %")
        plt.tight_layout()
        return fig

    # 2. Static Variables
    if interpretation["static_variables"].numel() > 0:
        figs["static_variables"] = make_selection_plot(
            "Static Variables Importance",
            interpretation["static_variables"].numpy(),
            static_labels,
        )

    # 3. Historical Variables
    # Filter 30 highest, otherwise the plot will explode if there are 400 links
    historical_vals = interpretation["historical_variables"].numpy()
    if len(historical_vals) > 30:
        top_indices = np.argsort(historical_vals)[-30:]
        top_vals = historical_vals[top_indices]
        top_labels = [historical_labels[i] for i in top_indices]
        title_suffix = " (Top 30)"
    else:
        top_vals = historical_vals
        top_labels = historical_labels
        title_suffix = ""

    figs["historical_variables"] = make_selection_plot(
        f"Historical Variables Importance{title_suffix}",
        top_vals,
        top_labels,
    )

    # 4. Future Variables
    figs["future_variables"] = make_selection_plot(
        "Future Variables Importance",
        interpretation["future_variables"].numpy(),
        future_labels,
    )

    return figs
