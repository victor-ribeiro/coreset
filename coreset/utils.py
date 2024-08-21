from datetime import datetime


def timeit(f_):
    def inner(*args, **kwargs):
        start = datetime.now().timestamp()
        out = f_(*args, **kwargs)
        end = datetime.now().timestamp()
        print(f"[RUNNING] {f_.__name__} :: {end - start:.4f}.S")
        return out

    return inner
