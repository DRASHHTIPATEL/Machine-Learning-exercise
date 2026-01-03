from typing import Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig


class TransformerClassifier(torch.nn.Module):
    """Wrapper around Hugging Face AutoModelForSequenceClassification.

    Allows toggling output_attentions and freezing the base model for ablation.
    """

    def __init__(self, model_name: str, num_labels: int = 2, output_attentions: bool = True, freeze_base: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, output_attentions=output_attentions)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        self.output_attentions = output_attentions
        if freeze_base:
            # freeze all parameters except classification head
            for name, p in self.model.named_parameters():
                if "classifier" not in name and "pre_classifier" not in name and "lin" not in name:
                    p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True):
        # Forward through HF model. We return whatever the HF model returns.
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=return_dict)

    def predict(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            attentions = getattr(out, "attentions", None)
            return preds, logits, attentions
