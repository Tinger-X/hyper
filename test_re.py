import os
import json
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread

from models import experimental_re
from utils import datasets_re, utils_re, metrics_re, plots_re, torch_utils_re


def test(
        data, weights=None, batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.6,
        save_json=False, single_cls=False, augment=False, verbose=False, model=None,
        dataloader=None, save_dir=Path(""), save_txt=False, save_hybrid=False, save_conf=True,
        plots=False, wandb_logger=None, compute_loss=None, half_precision=True, is_coco=False, args=None
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        utils_re.set_logging()
        device = torch_utils_re.select_device(args.device, batch_size=batch_size)

        # Directories
        save_dir = utils_re.increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = experimental_re.attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        img_size = utils_re.check_img_size(img_size, s=gs)  # check img_size

    # Half
    half = device.type != "cpu" and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith("coco.yaml")
        with open(data) as f:
            data = yaml.safe_load(f)
    utils_re.check_dataset(data)  # check
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        task = args.task if args.task in ("train", "val", "test") else "val"  # path to train/val/test images
        val_path_rgb = data["val_rgb"]
        val_path_ir = data["val_ir"]
        dataloader = datasets_re.create_dataloader_rgb_ir(
            val_path_rgb, val_path_ir, img_size, batch_size, gs, args,
            pad=0.5, rect=True, prefix=utils_re.style_str(task)
        )[0]

    seen = 0
    confusion_matrix = metrics_re.ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, "names") else model.module.names)}
    coco91class = utils_re.coco80_to_coco91_class()
    desc = ("{:20s}" + "{:12s}" * 7).format(
        "Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.75", "mAP@.5:.95"
    )
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=desc)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]

        with torch.no_grad():
            # Run model
            t = torch_utils_re.time_synchronized()
            out, train_out = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            t0 += torch_utils_re.time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = torch_utils_re.time_synchronized()
            out = utils_re.non_max_suppression(
                out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls
            )
            t1 += torch_utils_re.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            utils_re.scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # normalized xywh
                    xywh = (utils_re.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / "labels" / (path.stem + ".txt"), "a") as f:
                        f.write(("{:g} " * len(line))[:-1].format(*line) + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [
                        {
                            "position": {
                                "minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]
                            },
                            "class_id": int(cls),
                            "box_caption": f"{names[cls]} {conf:.3f}",
                            "scores": {"class_score": conf},
                            "domain": "pixel"
                        } for *xyxy, conf, cls in pred.tolist()
                    ]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    image_ndarray = img[si].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    rgb1, rgb2 = image_ndarray[:, :, :3], image_ndarray[:, :, 3:]
                    wandb_images.append(wandb_logger.wandb.Image(rgb1, boxes=boxes, caption=path.name))
                    wandb_images.append(wandb_logger.wandb.Image(rgb2, boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = utils_re.xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({
                        "image_id": image_id,
                        "category_id": coco91class[int(p[5])] if is_coco else int(p[5]),
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5)
                    })

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = utils_re.xywh2xyxy(labels[:, 1:5])
                utils_re.scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = utils_re.box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f"test_batch{batch_i}_labels.jpg"  # labels
            Thread(target=plots_re.plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f"test_batch{batch_i}_pred.jpg"  # predictions
            Thread(
                target=plots_re.plot_images,
                args=(
                    img, plots_re.output_to_target(out),
                    paths, f, names
                ),
                daemon=True
            ).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = metrics_re.ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    fmt = "{:20s}" + "{:12d}" * 2 + "{:12.3g}" * 5  # print format
    print(fmt.format("all", seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(fmt.format(names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
    if not training:
        print(
            "Speed: {:.1f}/{:.1f}/{:.1f} ms inference/NMS/total per {:g}x{:g} image at batch-size {:g}".format(*t)
        )

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [
                wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob("test*.jpg"))
            ]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = "../coco/annotations/instances_val2017.json"  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f"\nEvaluating pycocotools mAP... saving {pred_json}")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = (
            f"""\n{len(list(save_dir.glob("labels/*.txt")))} """
            f"""labels saved to {save_dir / "labels"}"""
        ) if save_txt else ""
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_args():
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--weights", nargs="+", type=str, default="./weights/20-best.pt", help="model.pt path(s)"
    )
    parser.add_argument("--data", type=str, default="./data/hsi/hsi_twostream.yaml", help="*.data path")
    parser.add_argument("--batch-size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.4, help="IOU threshold for NMS")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", default=False, action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", default=True, action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt"
    )
    parser.add_argument(
        "--save-conf", default=True, action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="save a coco api-compatible JSON results file"
    )
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="existing project/name ok, do not increment"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_json |= args.data.endswith("coco.yaml")
    args.data = utils_re.check_file(args.data)  # check file
    print(args)
    utils_re.check_requirements()

    if args.task in ("train", "val", "test"):  # run normally
        test(
            args.data, args.weights, args.batch_size, args.img_size, args.conf_thres, args.iou_thres,
            args.save_json, args.single_cls, args.augment, args.verbose, save_txt=args.save_txt | args.save_hybrid,
            save_hybrid=args.save_hybrid, save_conf=args.save_conf, args=args
        )
    elif args.task == "speed":  # speed benchmarks
        for w in args.weights:
            test(
                args.data, w, args.batch_size, args.img_size, 0.25,
                0.45, save_json=False, plots=False, args=args
            )
    elif args.task == "study":  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in args.weights:
            f = f"study_{Path(args.data).stem}_{Path(w).stem}.txt"  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f"\nRunning {f} point {i}...")
                r, _, t = test(
                    args.data, w, args.batch_size, i, args.conf_thres,
                    args.iou_thres, args.save_json, plots=False, args=args
                )
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
        plots_re.plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    main()
