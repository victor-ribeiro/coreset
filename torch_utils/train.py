import torch
from typing import Generator

__all__ = ["train", "valid_train"]


def valid_model(train_fn: Generator, /):
    def inner(*data_valid):
        def decorator(data_train, loss_fn, optmizer, model, epochs, agg_fn=sum):
            train_loop = train_fn(
                data_train, loss_fn, optmizer, model, epochs, agg_fn=sum
            )
            for e_loss, grad in train_loop:
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
                    yield grad, (e_loss, *valid_loss)

        return decorator

    return inner


def _train_loop(epoch_fn):
    def decorator(data_train, loss_fn, optmizer, model, epochs, agg_fn=sum):
        for _ in range(epochs):
            model.train(True)
            epoch_loss = epoch_fn(data_train, loss_fn, optmizer, model)
            epoch_loss = zip(*epoch_loss)
            yield [*map(agg_fn, epoch_loss)]

    return decorator


def _train_batch(data_train, loss_fn, optmizer, model):
    def _norm_grad():
        try:
            out_layer = [*model.modules()].pop()
            grad_norm = out_layer.weight.grad.data.norm(2)
            return grad_norm.item()
        except:
            pass

    for ftrs, tgt in data_train:
        pred = model(ftrs)
        loss = loss_fn(pred, tgt)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        yield loss.item(), _norm_grad()


train = _train_loop(_train_batch)
valid_train = valid_model(train)
