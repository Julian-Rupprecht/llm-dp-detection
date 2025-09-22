import torch
import logging
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from model import MultiTaskRoberta
from utils import preprocess_dataset, plot_metrics, compute_metrics
from tqdm import tqdm


def train(train_dl, validation_dl, model, device, logger):
    loss_hist = np.array([])
    dec_acc_hist = np.array([])
    dec_f1_hist = np.array([])
    cat_acc_hist = np.array([])
    cat_f1_hist = np.array([])

    epochs = 10
    num_no_improvements = 0
    early_stop = 3
    prev_avg_loss = float('inf')

    num_training_steps = epochs * len(train_dl)
    num_warmup_steps = num_training_steps * 0.05

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        avg_loss = 0.0
        iteration = 0

        for batch in tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            deceptive_labels = batch['deceptive_labels'].to(device)
            category_labels = batch['category_labels'].to(device)

            deception_logits, category_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_deception = torch.nn.BCEWithLogitsLoss()(deception_logits.squeeze(), deceptive_labels.float())
            loss_category = torch.nn.CrossEntropyLoss()(category_logits, category_labels)

            loss = loss_deception + loss_category

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            iteration += 1

        avg_loss = round(avg_loss / iteration, 5)
        loss_hist = np.append(loss_hist, avg_loss)

        results = validate(validation_dl, model, device, logger)

        logger.info(f"Deception Validation results of epoch {epoch + 1}:")
        logger.info("----------------------------")
        logger.info(f"Accuracy: {results['dec_acc']:.4f}, Precision: {results['dec_prec']:.4f}, Recall: {results['dec_rec']:.4f}, F1: {results['dec_f1']:.4f}\n")

        logger.info(f"Category Validation results of epoch {epoch + 1}:")
        logger.info("----------------------------")
        logger.info(f"Accuracy: {results['cat_acc']:.4f}, Precision: {results['cat_prec']:.4f}, Recall: {results['cat_rec']:.4f}, F1: {results['cat_f1']:.4f}\n")


        dec_acc_hist = np.append(dec_acc_hist, results["dec_acc"])
        dec_f1_hist = np.append(dec_f1_hist, results["dec_f1"])
        cat_acc_hist = np.append(cat_acc_hist, results["cat_acc"])
        cat_f1_hist = np.append(cat_f1_hist, results["cat_f1"])


        if np.abs(avg_loss - prev_avg_loss) < 0.02:
            num_no_improvements += 1
            logger.info(f"No improvement count: {num_no_improvements}")
        else:
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": cpu_state,
                    "roberta_name": "roberta-base",
                    "num_category_labels": model.category_classifier.out_features,
                },
                "weights.pth",
            )
            logger.info(f"Model checkpoint saved at epoch {epoch + 1} to weights.pth")
            num_no_improvements = 0

        prev_avg_loss = avg_loss

        if num_no_improvements >= early_stop:
            logger.info("Early stopping triggered")
            break

    return loss_hist, dec_acc_hist, dec_f1_hist, cat_acc_hist, cat_f1_hist


def validate(validation_dl, model, device, logger):
    dec_preds = np.array([])
    cat_preds = np.array([])
    true_dec_labels = np.array([])
    true_cat_labels = np.array([])

    model.eval()

    with torch.no_grad():
        for batch in validation_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            deceptive_labels = batch['deceptive_labels'].to(device)
            category_labels = batch['category_labels'].to(device)

            deception_logits, category_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            deceptive_probs = torch.sigmoid(deception_logits)
            category_probs = torch.softmax(category_logits, dim=1)

            deceptive_preds = (deceptive_probs > 0.5).long().squeeze()
            category_preds = torch.argmax(category_probs, dim=1)

            dec_preds = np.concatenate((dec_preds, deceptive_preds.cpu()), axis=0)
            cat_preds = np.concatenate((cat_preds, category_preds.cpu()), axis=0)

            true_dec_labels = np.concatenate((true_dec_labels, deceptive_labels.cpu().numpy()), axis=0)
            true_cat_labels = np.concatenate((true_cat_labels, category_labels.cpu().numpy()), axis=0)

    return {
        "dec_acc": compute_metrics("accuracy", None, dec_preds, true_dec_labels),
        "dec_prec": compute_metrics("precision", "binary", dec_preds, true_dec_labels),
        "dec_rec": compute_metrics("recall", "binary", dec_preds, true_dec_labels),
        "dec_f1": compute_metrics("f1", "binary", dec_preds, true_dec_labels),
        "cat_acc": compute_metrics("accuracy", None, cat_preds, true_cat_labels),
        "cat_prec": compute_metrics("precision", "macro", cat_preds, true_cat_labels),
        "cat_rec": compute_metrics("recall", "macro", cat_preds, true_cat_labels),
        "cat_f1": compute_metrics("f1", "macro", cat_preds, true_cat_labels)
    }



def test(test_dl, model, device, logger):
    results = validate(test_dl, model, device, logger)

    logger.info("Deception Test results:")
    logger.info("----------------------------")
    logger.info(f"Accuracy: {results['dec_acc']:.4f}, Precision: {results['dec_prec']:.4f}, Recall: {results['dec_rec']:.4f}, F1: {results['dec_f1']:.4f}\n")

    logger.info("Category Test results:")
    logger.info("----------------------------")
    logger.info(f"Accuracy: {results['cat_acc']:.4f}, Precision: {results['cat_prec']:.4f}, Recall: {results['cat_rec']:.4f}, F1: {results['cat_f1']:.4f}\n")


def __main__():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_ds, validation_ds, test_ds = preprocess_dataset("datasets\\dataset.csv", tokenizer, logger)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    roberta = RobertaModel.from_pretrained('roberta-base')
    model = MultiTaskRoberta(roberta, 8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_hist, dec_acc_hist, dec_f1_hist, cat_acc_hist, cat_f1_hist = train(
        train_dl, validation_dl, model, device, logger
    )


    ckpt = torch.load("weights.pth", map_location="cpu")
    roberta = RobertaModel.from_pretrained(ckpt["roberta_name"])
    modelR = MultiTaskRoberta(roberta, num_category_labels=ckpt["num_category_labels"])
    modelR.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelR.to(device)
    modelR.eval()

    test(test_dl, modelR, device, logger)

    x = range(1, len(dec_acc_hist) + 1)
    plot_metrics(x, loss_hist, 'Training Loss', 'Epoch', 'Loss', 'plots\\training_loss.png')
    plot_metrics(x, dec_acc_hist, 'Deception Accuracy', 'Epoch', 'Accuracy', 'plots\\deception_accuracy.png')
    plot_metrics(x, dec_f1_hist, 'Deception F1', 'Epoch', 'F1', 'plots\\deception_f1.png')
    plot_metrics(x, cat_acc_hist, 'Category Accuracy', 'Epoch', 'Accuracy', 'plots\\category_accuracy.png')
    plot_metrics(x, cat_f1_hist, 'Category F1', 'Epoch', 'F1', 'plots\\category_f1.png')


if __name__ == "__main__":
    __main__()
