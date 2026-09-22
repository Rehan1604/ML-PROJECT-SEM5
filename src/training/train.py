import torch
import torch.nn as nn
from tqdm import tqdm
from src.training.checkpoint import save_checkpoint


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for batch in progress_bar:
        # Works with both:
        # (images, labels)
        # (images, labels, metadata)
        images = batch[0]
        labels = batch[1]

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

        progress_bar.set_postfix(
            loss=loss.item()
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate_one_epoch(
    model,
    validation_loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        validation_loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress_bar:
        images = batch[0]
        labels = batch[1]

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs=10,
    learning_rate=0.001,
    checkpoint_path=None,
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.to(device)

    best_validation_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(epochs):

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        validation_loss, validation_accuracy = (
            validate_one_epoch(
                model=model,
                validation_loader=validation_loader,
                criterion=criterion,
                device=device,
            )
        )

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "train_accuracy"
        ].append(train_accuracy)

        history[
            "validation_loss"
        ].append(validation_loss)

        history[
            "validation_accuracy"
        ].append(validation_accuracy)

        

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            if checkpoint_path is not None:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    validation_accuracy=(
                        validation_accuracy
                    ),
                    path=checkpoint_path,
                )

                print(
                    "Best model saved to: "
                    f"{checkpoint_path}"
                )

    return history