import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _compute_target_p(target, loss_mask, pad_length, topk):
    target_head = target
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)

    target_p_padded = F.pad(
        target_p,
        pad=(0, 0, 0, pad_length),
        mode="constant",
        # For bitwise equality with previous code
        value=1 / target_p.shape[-1],
    )

    loss_mask_padded = F.pad(
        loss_mask,
        pad=(0, 0, 0, pad_length),
        mode="constant",
        value=0,
    )

    target_p_padded = target_p_padded.detach()
    _, target_p_padded_topk = target_p_padded.topk(topk, dim=-1)
    return target_p_padded, loss_mask_padded, target_p_padded_topk.detach()

class OfflineEagleModel(nn.Module):
    """
    In sgl-spec, we implement offline/online training.
    Offline training means we have the target hidden_states available before training.
    """

    def __init__(
        self,
        target_head, draft_model,
        length: int = 1,
        attention_backend="sdpa",
        topk=8,
    ):
        """
        Args:
            target_head: the target head to process the target hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.target_head = target_head
        # TODO to support TTT
        self.length = length
        self.attention_backend = attention_backend
        self.topk = topk

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # norm_hidden_states = self.norm(hidden_states)
        return self.target_head(hidden_states)

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        target = self.target_head(hidden_states.roll(-1, -2))
        # Step 0: handle vocab size
        with torch.no_grad():
            target_p_padded, loss_mask_padded, target_p_topk_padded = _compute_target_p(
                target=target,
                loss_mask=loss_mask,
                pad_length=self.length - 1,
                topk=self.topk,
            )
        del target

        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # make attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, hidden_states, batch_size, seq_length, 0
        )
        
        plosses = []
        acces = []
        topk_losses = []
        cache_hidden = ([], [])
        for i in range(self.length):
            target_p = target_p_padded[:, i : seq_length + i]
            target_p_topk = target_p_topk_padded[:, i : seq_length + i]
            loss_mask = loss_mask_padded[:, i : seq_length + i]
            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(input_ids.roll(-i, -1))

            # Step 5.2: run the draft model backbone
            hidden_states = self.draft_model(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=cache_hidden,
            )

            # Step 5.4: get logits
            logits = self.compute_logits(hidden_states)

            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            plogp_topk = plogp.gather(-1, target_p_topk)
            loss = -torch.sum(loss_mask * plogp, 2).mean()
            topk_loss = -torch.sum(loss_mask * plogp_topk, 2).mean()
            with torch.no_grad():
                acc = ((logits.argmax(-1) == target_p.argmax(-1)) * loss_mask.squeeze(-1)).sum() / loss_mask.sum().clamp_min(1e-6)

            plosses.append(loss)
            acces.append(acc)
            topk_losses.append(topk_loss)

        return plosses, acces, topk_losses
