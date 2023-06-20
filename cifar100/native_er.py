import torch
from avalanche.benchmarks import SplitCIFAR100
from avalanche.evaluation.metrics import (accuracy_metrics,
                                          confusion_matrix_metrics,
                                          forgetting_metrics, loss_metrics)
from avalanche.logging import InteractiveLogger
from avalanche.models import pytorchcv_wrapper
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.supervised import Naive
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from utils import get_device, set_seed


def native_er_cifar100(args):
    set_seed(args.seed)
    device = get_device(args.cuda)

    benchmark = SplitCIFAR100(
        args.task_num,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=list(range(100)),
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )

    model = pytorchcv_wrapper.resnet("cifar100", depth=20, pretrained=False)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            num_classes=benchmark.n_classes, save_image=False, stream=True
        ),
        loggers=[InteractiveLogger()],
        strict_checks=False,
    )

    replay_plugin = ReplayPlugin(
        mem_size=args.mem_size,
        batch_size_mem=args.batch_size_mem,
        storage_policy=ReservoirSamplingBuffer(args.mem_size)
    )

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = CrossEntropyLoss()

    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.train_mb_size,
        train_epochs=args.train_epoch,
        eval_mb_size=args.eval_mb_size,
        device=device,
        plugins=[replay_plugin],
        evaluator=eval_plugin,
    )

    for t, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        cl_strategy.train(experience, eval_streams=[])
        cl_strategy.eval(benchmark.test_stream[:t+1])

    print("Computing accuracy on the whole test set")
    cl_strategy.eval(benchmark.test_stream)





