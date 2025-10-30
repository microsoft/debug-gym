import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler


def prefix_tuning():
    # Load the IMDB dataset
    PREFIX_LENGTH = 10

    # Create a simple fake dataset class
    # Create a simple fake dataset class without tokenizer
    class FakeTextDataset(Dataset):
        def __init__(self, num_samples=100, max_length=512 - PREFIX_LENGTH):
            # Fake "tokenized" data: sequences of integers simulating word tokens
            self.max_length = max_length
            self.data = [
                torch.randint(0, 100, (max_length,)) for _ in range(num_samples)
            ]  # Random integers as token ids
            self.labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]  # 1 for positive, 0 for negative

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            input_ids = self.data[idx]
            attention_mask = torch.ones(self.max_length)  # All tokens are attended to (no padding)
            labels = torch.tensor(self.labels[idx], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Load the pre-trained model and tokenizer from Hugging Face
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Freeze the entire model except for the added prefix
    for param in model.parameters():
        param.requires_grad = False

    # Create a fake dataset with 100 samples
    train_dataset = FakeTextDataset(num_samples=16)
    test_dataset = FakeTextDataset(num_samples=8)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(test_dataset, batch_size=8)

    # Define the prefix tuning class
    class PrefixTuning(nn.Module):
        def __init__(self, model, prefix_length=PREFIX_LENGTH, hidden_size=768):
            super().__init__()
            self.prefix_length = prefix_length
            self.hidden_size = hidden_size

            # Learnable prefix vectors
            self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, hidden_size))

            # Get the number of layers in the transformer (distilbert has 6 layers)
            self.num_layers = len(model.distilbert.transformer.layer)

        def forward(self, input_embeds):
            # Expand prefix embeddings to match batch size
            batch_size = input_embeds.size(0)
            prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, prefix_length, hidden_size)

            # Concatenate the prefix embeddings to the input embeddings
            input_embeds_with_prefix = torch.cat([prefix_embeds, input_embeds])

            return input_embeds_with_prefix

    # Create a prefix tuning module
    prefix_tuning = PrefixTuning(model)

    # Set up the optimizer only for prefix embeddings
    optimizer = AdamW(prefix_tuning.parameters(), lr=5e-5)

    # Number of epochs
    num_epochs = 1

    # Set up learning rate scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Set up the device (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    prefix_tuning.to(device)

    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Extract the input embeddings (usually from the token ids)
            input_embeds = model.distilbert.embeddings.word_embeddings(batch["input_ids"])

            # Apply prefix tuning to the input embeddings
            input_embeds_with_prefix = prefix_tuning(input_embeds)

            # Forward pass through the transformer layers manually, with prepended prefixes
            outputs = model(
                inputs_embeds=input_embeds_with_prefix,
                attention_mask=batch["mask"],
                labels=batch["labels"]
            )

            # Compute the loss and backpropagate
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)

            progress_bar.update(1)
