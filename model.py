import torch 

class MultiTaskRoberta(torch.nn.Module):
    def __init__(self, roberta, num_category_labels=2):
        super().__init__()
        self.roberta = roberta
        self.dropout = torch.nn.Dropout(0.1)
        self.deception_classifier = torch.nn.Linear(self.roberta.config.hidden_size, 1)
        self.category_classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_category_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        deception_logits = self.deception_classifier(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        return deception_logits, category_logits