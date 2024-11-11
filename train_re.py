import os
import copy
import time
import math
import yaml
import tqdm
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from threading import Thread
import torch.distributed as dist
from torch.utils import tensorboard
from torch.nn.parallel import DistributedDataParallel

import test_re
from models import yolo_re, experimental_re
from utils import (
    g_var_re, utils_re, wandb_utils_re, torch_utils_re, metrics_re, plots_re,
    google_utils_re, datasets_re, autoanchor_re, loss_re
)


def train(hyper, args, tb_writer):
    device = g_var_re.Var.get("device")
    title = utils_re.style_str("hyperparameters")
    info = ", ".join(f"{k}={v}" for (k, v) in hyper.items())
    logger.info(f"{title}: ({info})")

    save_dir, epochs, batch_size = Path(args.save_dir), args.epochs, args.batch_size
    total_batch_size, weights, rank = args.total_batch_size, args.weights, args.global_rank

    # Directories
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyper.yaml", "w") as f:
        yaml.safe_dump(hyper, f, sort_keys=False)
    with open(save_dir / "args.yaml", "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    # Configure
    plots = not args.evolve  # create plots
    cuda = g_var_re.Var.get("device").type == "cuda"
    utils_re.init_seeds(2 + args.global_rank)
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = args.data.endswith("coco.yaml")

    # Logging-Doing this before checking the dataset. Might update data_dict
    loggers = {"wandb": None}  # loggers dict
    wandb_logger = None  # TODO, 外部要用
    if args.global_rank in [-1, 0]:
        args.hyper = hyper  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith(".pt") and os.path.isfile(weights) else None
        wandb_logger = wandb_utils_re.WandbLogger(args, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            # WandbLogger might update weights, epochs if resuming
            weights, epochs, hyper = args.weights, args.epochs, args.hyper

    nc = 1 if args.single_cls else int(data_dict["nc"])  # number of classes
    names = ["item"] if args.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {args.data}"

    # Model
    pretrained = weights.endswith(".pt")
    ckpt = None  # TODO, 外部要用
    if pretrained:
        with torch_utils_re.torch_distributed_zero_first(rank):
            google_utils_re.attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = yolo_re.Model(args.config or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyper.get("anchors")).to(device)
        exclude = ["anchor"] if (args.config or hyper.get("anchors")) and not args.resume else []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = torch_utils_re.intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {weights}")
    else:
        model = yolo_re.Model(args.config, ch=3, nc=nc, anchors=hyper.get("anchors")).to(device)  # create
    with torch_utils_re.torch_distributed_zero_first(rank):
        utils_re.check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path = data_dict["val"]

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyper["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyper['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if args.adam:
        optimizer = optim.Adam(pg0, lr=hyper["lr0"], betas=(hyper["momentum"], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyper["lr0"], momentum=hyper["momentum"], nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": hyper["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    logger.info(f"Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other")
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if args.linear_lr:
        lf = utils_re.linear_lr(hyper["lrf"], epochs)
    else:
        lf = utils_re.one_cycle(1, hyper['lrf'], epochs)  # cosine 1->hyper['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = torch_utils_re.ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get("training_results") is not None:
            results_file.write_text(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if args.resume:
            assert start_epoch > 0, f"{weights} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            logger.info(
                f"{weights} has been trained for {ckpt['epoch']} epochs. "
                f"Fine-tuning for {epochs} additional epochs."
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyper['obj'])
    # verify img_size are gs-multiples
    img_size, img_size_test = [utils_re.check_img_size(x, gs) for x in args.img_size]

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if args.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # Train loader
    dataloader, dataset = datasets_re.create_dataloader(
        train_path, img_size, batch_size, gs, args,
        hyper=hyper, augment=True, cache=args.cache_images, rect=args.rect,
        rank=rank, world_size=args.world_size, workers=args.workers,
        image_weights=args.image_weights, quad=args.quad,
        prefix=utils_re.style_str("train")
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    print(mlc)
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {args.data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    test_loader, test_data = None, None  # TODO, 外部要用
    if rank in [-1, 0]:
        test_loader, test_data = datasets_re.create_dataloader(
            test_path, img_size_test, batch_size * 2, gs, args, hyper=hyper,
            cache=args.cache_images and not args.notest, rect=True, rank=-1,
            world_size=args.world_size, workers=args.workers, pad=0.5,
            prefix=utils_re.style_str("val")
        )

        if not args.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                plots_re.plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram("classes", c, 0)

            # Anchors
            if not args.noautoanchor:
                autoanchor_re.check_anchors(dataset, model=model, thr=hyper["anchor_t"], img_size=img_size)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
            find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        )

    # Model parameters
    hyper["box"] *= 3. / nl  # scale to layers
    hyper["cls"] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyper["obj"] *= (img_size / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyper["label_smoothing"] = args.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyper = hyper  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = utils_re.labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyper["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    compute_loss = loss_re.ComputeLoss(model)  # init loss class
    logger.info(
        f"Image sizes {img_size} train, {img_size_test} test\n"
        f"Using {dataloader.num_workers} dataloader workers\n"
        f"Logging results to {save_dir}\n"
        f"Starting training for {epochs} epochs..."
    )
    epoch = -1  # TODO, 外部要用
    for epoch in range(start_epoch, epochs):
        model.train()

        g_var_re.Var.set("s", torch.randperm(len(dataset)).tolist())
        print(" TRAIN SAMPLER ")
        print(list(dataloader.sampler))

        # Update image weights (optional)
        if args.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = utils_re.labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mean_loss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        msg = ("\n" + "{:10s}" * 8).format("Epoch", "gpu_mem", "box", "obj", "cls", "total", "labels", "img_size")
        logger.info(msg)
        if rank in [-1, 0]:
            pbar = tqdm.tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        desc = ""  # TODO, 外部要用
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())  # noqa
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi,
                        [
                            hyper["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch)
                        ]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi,
                            [
                                hyper["warmup_momentum"],
                                hyper["momentum"]
                            ]
                        )

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(img_size * 0.5, img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.amp.autocast("cuda", enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode
                if args.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mean_loss = (mean_loss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g} GB"
                fmt = "{:10s}" * 2 + "{:10.4g}" * 6
                desc = fmt.format(f"{epoch}/{epochs - 1}", mem, *mean_loss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(desc)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f"train_batch{ni}.jpg"  # filename
                    Thread(target=plots_re.plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({
                        "Mosaics": [
                            wandb_logger.wandb.Image(str(x), caption=x.name)
                            for x in save_dir.glob("train*.jpg") if x.exists()
                        ]
                    })

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        g_var_re.Var.set("s", torch.randperm(len(test_data)).tolist())
        print(" TEST SAMPLER ")
        print(list(test_loader.sampler))

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=["yaml", "nc", "hyper", "gr", "names", "stride", "class_weights"])
            final_epoch = epoch + 1 == epochs
            if not args.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test_re.test(
                    data_dict, batch_size=batch_size * 2, img_size=img_size_test, model=ema.ema,
                    single_cls=args.single_cls, dataloader=test_loader, save_dir=save_dir,
                    verbose=nc < 50 and final_epoch, plots=plots and final_epoch,
                    wandb_logger=wandb_logger, compute_loss=compute_loss, is_coco=is_coco
                )

            # Write
            with open(results_file, "a") as f:
                f.write(desc + ("{:10.4g}" * 8).format(*results) + "\n")  # append metrics, val_loss
            if len(args.name) and args.bucket:
                os.system(f"gsutil cp {results_file} gs://{args.bucket}/results/results{args.name}.txt")

            # Log
            tags = [
                "train/box_loss", "train/obj_loss", "train/cls_loss",  # train loss
                "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.75",
                "metrics/mAP_0.5:0.95",
                "val/box_loss", "val/obj_loss", "val/cls_loss",  # val loss
                "x/lr0", "x/lr1", "x/lr2"
            ]  # params
            for x, tag in zip(list(mean_loss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = metrics_re.fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not args.nosave) or (final_epoch and not args.evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "training_results": results_file.read_text(),
                    "model": copy.deepcopy(model.module if torch_utils_re.is_parallel(model) else model).half(),
                    "ema": copy.deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "wandb_id": wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % args.save_period == 0 and not final_epoch) and args.save_period != -1:
                        wandb_logger.log_model(last.parent, args, epoch, fi, best_model=best_fitness == fi)
                del ckpt

    if rank in [-1, 0]:
        # Plots
        if plots:
            plots_re.plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ["results.png", "confusion_matrix.png", *[f"{x}_curve.png" for x in ("F1", "PR", "P", "R")]]
                wandb_logger.log({
                    "Results": [
                        wandb_logger.wandb.Image(str(save_dir / f), caption=f)
                        for f in files if (save_dir / f).exists()
                    ]
                })
        # Test best.pt
        logger.info(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n")
        if args.data.endswith("coco.yaml") and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else last:  # speed, mAP tests
                results, _, _ = test_re.test(
                    args.data, batch_size=batch_size * 2, img_size=img_size_test, conf_thres=0.001,
                    iou_thres=0.7, model=experimental_re.attempt_load(m, device).half(),
                    single_cls=args.single_cls, dataloader=test_loader, save_dir=save_dir,
                    save_json=True, plots=False, is_coco=is_coco
                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                utils_re.strip_optimizer(f)  # strip optimizers
        if args.bucket:
            os.system(f"gsutil cp {final} gs://{args.bucket}/weights")  # upload
        if wandb_logger.wandb and not args.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(
                str(final), type="model",
                name="run_" + wandb_logger.wandb_run.id + "_model",
                aliases=["last", "best", "stripped"]
            )
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


def train_rgb_ir(hyper, args, tb_writer):
    device = g_var_re.Var.get("device")

    logger.info(utils_re.style_str("hyperparameters") + ": " + ", ".join(f"{k}={v}" for k, v in hyper.items()))
    save_dir, epochs, batch_size = Path(args.save_dir), args.epochs, args.batch_size
    total_batch_size, weights, rank = args.total_batch_size, args.weights, args.global_rank

    # Directories
    w_dir = save_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last = w_dir / "last.pt"
    best = w_dir / "best.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyper.yaml", "w") as f:
        yaml.safe_dump(hyper, f, sort_keys=False)
    with open(save_dir / "args.yaml", "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    # Configure
    plots = not args.evolve  # create plots
    cuda = device.type != "cpu"
    utils_re.init_seeds(2 + rank)
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = args.data.endswith("coco.yaml")

    # Logging-Doing this before checking the dataset. Might update data_dict
    loggers = {"wandb": None}  # loggers dict
    wandb_logger = None  # TODO, 外部要用
    if rank in [-1, 0]:
        args.hyper = hyper  # add hyperparameters
        run_id = None
        if weights.endswith(".pt") and os.path.isfile(weights):
            run_id = torch.load(weights, weights_only=False).get("wandb_id")
        wandb_logger = wandb_utils_re.WandbLogger(args, save_dir.stem, run_id, data_dict)
        loggers["wandb"] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            # WandbLogger might update weights, epochs if resuming
            weights, epochs, hyper = args.weights, args.epochs, args.hyper

    nc = 1 if args.single_cls else int(data_dict["nc"])  # number of classes
    names = ["item"] if args.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {args.data}"  # check

    # Model
    pretrained = weights.endswith(".pt")
    ckpt = {}  # TODO: `ckpt`之后会使用到，但是这里的if分支并没有确保`ckpt`始终存在
    if pretrained:
        with torch_utils_re.torch_distributed_zero_first(rank):
            google_utils_re.attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, weights_only=False, map_location=device)  # load checkpoint
        model = yolo_re.Model(
            args.config or ckpt["model"].yaml,
            ch=6, nc=nc, anchors=hyper.get("anchors")
        ).to(device)  # create
        # exclude keys
        exclude = ["anchor"] if (args.config or hyper.get("anchors")) and not args.resume else []
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = torch_utils_re.intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        model = yolo_re.Model(args.config, ch=3, nc=nc, anchors=hyper.get("anchors")).to(device)  # create
    with torch_utils_re.torch_distributed_zero_first(rank):
        utils_re.check_dataset(data_dict)  # check
    train_path_rgb = data_dict["train_rgb"]
    test_path_rgb = data_dict["val_rgb"]
    train_path_ir = data_dict["train_ir"]
    test_path_ir = data_dict["val_ir"]

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyper["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if args.adam:
        # adjust beta1 to momentum
        optimizer = optim.Adam(pg0, lr=hyper["lr0"], betas=(hyper["momentum"], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyper["lr0"], momentum=hyper["momentum"], nesterov=True)

    # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg1, "weight_decay": hyper["weight_decay"]})
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if args.linear_lr:
        lf = utils_re.linear_lr(hyper["lrf"], epochs)  # linear
    else:
        lf = utils_re.one_cycle(1, hyper["lrf"], epochs)  # cosine 1->hyper['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = torch_utils_re.ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]

        # Results
        if ckpt.get("training_results") is not None:
            results_file.write_text(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if args.resume:
            assert start_epoch > 0, f"{weights} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            logger.info(
                f"{weights} has been trained for {ckpt['epoch']} epochs. "
                f"Fine-tuning for {epochs} additional epochs."
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyper['obj'])
    # verify img_size are gs-multiples
    img_size, img_size_test = [utils_re.check_img_size(x, gs) for x in args.img_size]

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if args.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # Train loader
    dataloader, dataset = datasets_re.create_dataloader_rgb_ir(
        train_path_rgb, train_path_ir, img_size, batch_size, gs, args,
        hyper=hyper, augment=True, cache=args.cache_images, rect=args.rect,
        rank=rank, world_size=args.world_size, workers=args.workers,
        image_weights=args.image_weights, quad=args.quad, prefix=utils_re.style_str("train")
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, (
        f"Label class {mlc} exceeds nc={nc} in {args.data}. Possible class labels are 0-{nc - 1}"
    )

    # Process 0
    test_loader = None  # TODO, 外部要用
    if rank in [-1, 0]:
        test_loader, test_data = datasets_re.create_dataloader_rgb_ir(
            test_path_rgb, test_path_ir, img_size_test, batch_size * 2, gs,
            args, hyper=hyper, cache=args.cache_images and not args.notest,
            rect=True, rank=-1, world_size=args.world_size, workers=args.workers,
            pad=0.5, prefix=utils_re.style_str("val")
        )

        if not args.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                plots_re.plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram("classes", c, 0)

            # Anchors
            if not args.no_auto_anchor:
                autoanchor_re.check_anchors(dataset, model=model, thr=hyper["anchor_t"], img_size=img_size)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        )

    # Model parameters
    hyper["box"] *= 3. / nl  # scale to layers
    hyper["cls"] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyper["obj"] *= (img_size / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyper["label_smoothing"] = args.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyper = hyper  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # attach class weights
    model.class_weights = utils_re.labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = max(round(hyper["warmup_epochs"] * nb), 1000)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    compute_loss = loss_re.ComputeLoss(model)  # init loss class
    logger.info(
        f"Image sizes {img_size} train, {img_size_test} test\n"
        f"Using {dataloader.num_workers} dataloader workers\n"
        f"Logging results to {save_dir}\n"
        f"Starting training for {epochs} epochs..."
    )
    epoch = -1  # TODO, 外部要用
    desc = ""  # TODO, 外部要用
    for epoch in range(start_epoch, epochs):
        model.train()
        # Update image weights (optional)
        if args.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = utils_re.labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mean_loss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        fmt = "\n" + "{:10s}" * 8
        logger.info(fmt.format("Epoch", "gpu_mem", "box", "obj", "cls", "total", "labels", "img_size"))
        if rank in [-1, 0]:
            pbar = tqdm.tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_rgb = imgs[:, :3, :, :]
            imgs_ir = imgs[:, 3:, :, :]

            # FQY my code 训练数据可视化
            if g_var_re.Var.get("flag_visual_training_dataset"):
                from torchvision import transforms
                un_loader = transforms.ToPILImage()
                for num in range(batch_size):
                    image = imgs[num, :3, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = un_loader(image)
                    image.save(f"example_{epoch}_{i}_{num}_color.jpg")
                    image = imgs[num, 3:, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = un_loader(image)
                    image.save(f"example_{epoch}_{i}_{num}_ir.jpg")

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())  # noqa
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi,
                        [
                            hyper["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch)
                        ]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi,
                            [
                                hyper["warmup_momentum"],
                                hyper["momentum"]
                            ]
                        )

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(img_size * 0.5, img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.amp.autocast("cuda", enabled=cuda):
                pred = model(imgs_rgb, imgs_ir)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode
                if args.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mean_loss = (mean_loss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB"
                fmt = "{:10s}" * 2 + "{:10.4g}" * 6
                desc = fmt.format(f"{epoch}/{epochs - 1}", mem, *mean_loss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(desc)

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=["yaml", "nc", "hyper", "gr", "names", "stride", "class_weights"])
            final_epoch = epoch + 1 == epochs
            if not args.no_test or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test_re.test(
                    data_dict, batch_size=batch_size * 2, img_size=img_size_test, model=ema.ema,
                    single_cls=args.single_cls, dataloader=test_loader, save_dir=save_dir,
                    verbose=nc < 50 and final_epoch, plots=plots and final_epoch,
                    wandb_logger=wandb_logger, compute_loss=compute_loss, is_coco=is_coco
                )

            # Write
            with open(results_file, "a") as f:
                f.write(desc + ("{:10.4g}" * 8).format(*results) + "\n")  # append metrics, val_loss
            if len(args.name) and args.bucket:
                os.system(f"gsutil cp {results_file} gs://{args.bucket}/results/results{args.name}.txt")

            # Log
            tags = [
                "train/box_loss", "train/obj_loss", "train/cls_loss",  # train loss
                "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.75",
                "metrics/mAP_0.5:0.95",
                "val/box_loss", "val/obj_loss", "val/cls_loss",  # val loss
                "x/lr0", "x/lr1", "x/lr2"  # params
            ]
            for x, tag in zip(list(mean_loss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = metrics_re.fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not args.no_save) or (final_epoch and not args.evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "training_results": results_file.read_text(),
                    "model": copy.deepcopy(model.module if torch_utils_re.is_parallel(model) else model).half(),
                    "ema": copy.deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "wandb_id": wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % args.save_period == 0 and not final_epoch) and args.save_period != -1:
                        wandb_logger.log_model(last.parent, args, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plots_re.plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ["results.png", "confusion_matrix.png", *[f"{x}_curve.png" for x in ("F1", "PR", "P", "R")]]
                wandb_logger.log({
                    "Results": [
                        wandb_logger.wandb.Image(str(save_dir / f), caption=f)
                        for f in files if (save_dir / f).exists()
                    ]
                })
        # Test best.pt
        logger.info(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n")
        if args.data.endswith("coco.yaml") and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else last:  # speed, mAP tests
                results, _, _ = test_re.test(
                    args.data, batch_size=batch_size * 2, img_size=img_size_test, conf_thres=0.001, iou_thres=0.7,
                    model=experimental_re.attempt_load(m, device).half(), single_cls=args.single_cls,
                    dataloader=test_loader, save_dir=save_dir, save_json=True, plots=False, is_coco=is_coco
                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                utils_re.strip_optimizer(f)  # strip optimizers
        if args.bucket:
            os.system(f"gsutil cp {final} gs://{args.bucket}/weights")  # upload
        if wandb_logger.wandb and not args.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(
                str(final), type="model",
                name="run_" + wandb_logger.wandb_run.id + "_model",
                aliases=["last", "best", "stripped"]
            )
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


def parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str,
        default="yolo_weight/yolov5l.pt",
        help="initial weights path"
    )
    parser.add_argument(
        "--config", type=str,
        default="models/hsi/yolov5l_fusion_transformerx3_hsi.yaml",
        help="model.yaml path"
    )
    parser.add_argument(
        "--data", type=str,
        default="data/hsi/hsi_twostream.yaml",
        help="data.yaml path"
    )
    parser.add_argument(
        "--hyper", type=str,
        default="data/hyp.finetune.yaml",
        help="hyperparameters path"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="total batch size for all GPUs")
    parser.add_argument(
        "--img-size", nargs="+", type=int,
        default=[640, 640], help="[train, test] image sizes"
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume", nargs="?", const=True, default=False,
        help="resume most recent training"
    )
    parser.add_argument("--no-save", action="store_true", help="only save final checkpoint")
    parser.add_argument("--no-test", action="store_true", help="only test final epoch")
    parser.add_argument("--no-auto-anchor", action="store_true", help="disable auto-anchor check")
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default='', help="gsutil bucket")
    parser.add_argument("--cache-images", action="store_true", help="cache images for faster training")
    parser.add_argument(
        "--image-weights", action="store_true",
        help="use weighted image selection for training"
    )
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%")
    parser.add_argument(
        "--single-cls", action="store_true", help="train multi-class data as single-class"
    )
    parser.add_argument("--adam", action="store_true", help="use torch.optim.Adam() optimizer")
    parser.add_argument(
        "--sync-bn", action="store_true",
        help="use SyncBatchNorm, only available in DDP mode"
    )
    parser.add_argument("--local-rank", type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument("--workers", type=int, default=8, help="maximum number of dataloader workers")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true",
        help="existing project/name ok, do not increment"
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument(
        "--upload-dataset", action="store_true",
        help="Upload dataset as W&B artifact table"
    )
    parser.add_argument(
        "--bbox-interval", type=int, default=-1,
        help="Set bounding-box image logging interval for W&B"
    )
    parser.add_argument(
        "--save-period", type=int, default=-1,
        help="Log model after every \"save_period\" epoch"
    )
    parser.add_argument(
        "--artifact-alias", type=str, default="latest",
        help="version of dataset artifact to be used"
    )

    args = parser.parse_args()
    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    return args


def main():
    g_var_re.Var.set("flag_visual_training_dataset", False)

    args = parse_args()
    utils_re.set_logging(args.global_rank)
    if args.global_rank in [-1, 0]:
        utils_re.check_requirements()

    # Resume
    wandb_run = wandb_utils_re.check_wandb_resume(args)
    if args.resume and not wandb_run:  # resume an interrupted run
        # specified or most recent path
        ckpt = args.resume if isinstance(args.resume, str) else utils_re.get_latest_run()
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        apriori = args.global_rank, args.local_rank
        with open(Path(ckpt).parent.parent / "args.yaml") as f:
            args = argparse.Namespace(**yaml.safe_load(f))  # replace
        args.config, args.weights, args.resume, args.batch_size = "", ckpt, True, args.total_batch_size
        args.global_rank, args.local_rank = apriori
        logger.info(f"Resuming training from {ckpt}")
    else:
        args.data = utils_re.check_file(args.data)
        args.config = utils_re.check_file(args.config)
        args.hyper = utils_re.check_file(args.hyper)
        assert len(args.config) or len(args.weights), "either --config or --weights must be specified"
        args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))  # extend to 2 sizes (train, test)
        args.name = "evolve" if args.evolve else args.name
        args.save_dir = str(
            utils_re.increment_path(
                Path(args.project) / args.name,
                exist_ok=args.exist_ok | args.evolve
            )
        )

    # DDP mode
    args.total_batch_size = args.batch_size
    device = torch_utils_re.select_device(args.device, batch_size=args.batch_size)
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")  # distributed backend
        assert args.batch_size % args.world_size == 0, "--batch-size must be multiple of CUDA device count"
        args.batch_size = args.total_batch_size // args.world_size
    g_var_re.Var.set("device", device)

    # Hyperparameters
    with open(args.hyper) as f:
        hyper = yaml.safe_load(f)  # load hyper

    # Train
    logger.info(args)
    if not args.evolve:
        if args.global_rank in [-1, 0]:
            prefix = utils_re.style_str("tensorboard")
            logger.info(f"{prefix}: Start with `tensorboard --logdir {args.project}`, view at http://localhost:6006/")
            tb_writer = tensorboard.SummaryWriter(args.save_dir)  # Tensorboard
            train_rgb_ir(hyper, args, tb_writer)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0)  # image mixup (probability)
        }

        assert args.local_rank == -1, "DDP mode not implemented for --evolve"
        args.no_test, args.no_save = True, True  # only test/save final epoch
        yaml_file = Path(args.save_dir) / "hyp_evolved.yaml"  # save best result here
        if args.bucket:
            os.system(f"gsutil cp gs://{args.bucket}/evolve.txt .")  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path("evolve.txt").exists():  # if evolve.txt exists: select best hyper and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-metrics_re.fitness(x))][:n]  # top n mutations
                w = metrics_re.fitness(x) - metrics_re.fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # noqa, mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyper.keys()):  # plt.hist(v.ravel(), 300)
                    hyper[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyper[k] = max(hyper[k], v[1])  # lower limit
                hyper[k] = min(hyper[k], v[2])  # upper limit
                hyper[k] = round(hyper[k], 5)  # significant digits

            # Train mutation
            results = train(hyper.copy(), args, device)

            # Write mutation results
            utils_re.print_mutation(hyper.copy(), results, yaml_file, args.bucket)

        # Plot results
        plots_re.plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyper {yaml_file}"
        )


if __name__ == "__main__":
    logger = g_var_re.Var.get("logger")
    main()
