from cifar100 import native_er
from utils import create_default_args

native_er_cifar100_args = create_default_args(
    {
        "cuda": 0,
        "task_num": 20,
        "mem_size": 2000,
        "lr": 0.1,
        "train_mb_size": 10,
        "eval_mb_size": 500,
        "train_epoch": 5,
        "seed": 123,
        "batch_size_mem": 10,
    })

native_er.native_er_cifar100(native_er_cifar100_args)