import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm


def evaluate_model(model, data_loader, device):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        progress_bar = tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
        )

        for batch in progress_bar:

            images = batch[0]
            labels = batch[1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    num_classes = model.classifier[-1].out_features
    
    confusion = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
    )

    return (accuracy,f1,confusion,all_labels,all_predictions,)