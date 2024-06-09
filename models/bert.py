import torch
from torch import nn
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, 
                 bert_name='bert-base-uncased',
                 num_layers=1
                 ):
        super(BertEncoder, self).__init__()
        self.bert_name = bert_name

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = False
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = num_layers

    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
    
        # with padding, always 256
        outputs = self.model(
            input_ids=input,
            attention_mask=mask,
            output_hidden_states=True,
        )
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = None
        features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers

        embedded = features * mask.unsqueeze(-1).float()
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret


def build_bert_backbone():
    """
    Create a Bert instance from the listed parameters.

    Returns:
        BertEncoder: a :class:`BertEncoder` instance.
    """
    return BertEncoder()
