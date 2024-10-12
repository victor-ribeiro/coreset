import torch
from typing import Generator
import time

import torch.utils


__all__ = ["train_loop", "eval_train"]


def _valid_model(train_fn: Generator, /):
    def inner(*data_valid):
        def decorator(data_train, loss_fn, optmizer, model, epochs, agg_fn=sum):
            # training loop
            train_loop = train_fn(
                data_train, loss_fn, optmizer, model, epochs, agg_fn=sum
            )
            # validation loop
            for e_loss in train_loop:
                model.eval()
                with torch.no_grad():
                    valid_loss = []
                    for data in data_valid:
                        data_loss = []
                        for ftr, tgt in data:
                            pred = model(ftr)
                            loss = loss_fn(pred, tgt)
                            data_loss += [loss.item()]
                        valid_loss += [agg_fn(data_loss)]
                    yield (e_loss, *valid_loss)
                print(f"[TRAIN] \t {model.__name__} :: {e_loss:.4f}")

        return decorator

    return inner


def train_loop(data_train, loss_fn, optmizer, model, epochs, agg_fn=sum):
    elapsed = 0
    for i in range(epochs):
        model.train(True)
        epoch_loss = 0
        total = 0
        ################################################################
        start_time = time.time()
        data_train = [_ for _ in data_train]
        for ftrs, tgt in data_train:
            total += len(ftrs)
            pred = model(ftrs)
            loss = loss_fn(pred.squeeze(), tgt)
            epoch_loss += loss.item()
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        ################################################################
        epoch_loss = epoch_loss / len(data_train) * epochs
        yield epoch_loss
        end_time = time.time()
        elapsed += end_time - start_time
        print(
            f"[{i}] \t {model.__class__.__name__} :: {epoch_loss:.4f} :: {elapsed:.4f} sec :: \t{total} "
        )


eval_train = _valid_model(train_loop)
