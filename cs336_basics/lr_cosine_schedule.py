import math

def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    assert warmup_iters <= cosine_cycle_iters
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + (1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi)) * (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate
