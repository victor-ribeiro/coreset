import torch
from typing import Generator

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
                # print(f"[TRAIN] \t {model.__name__}::{e_loss}")

        return decorator

    return inner


def train_loop(data_train, loss_fn, optmizer, model, epochs, agg_fn=sum):
    print(type(model))
    for i in range(epochs):
        model.train(True)
        epoch_loss = 0
        ################################################################
        for ftrs, tgt in data_train:
            optmizer.zero_grad()

            pred = model(ftrs)
            loss = loss_fn(pred.round(), tgt)
            epoch_loss += loss.item()  # / len(ftrs)
            # optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        ################################################################

        yield epoch_loss
        print(f"[{i}] \t {model.__class__.__name__} :: {epoch_loss}")


eval_train = _valid_model(train_loop)
