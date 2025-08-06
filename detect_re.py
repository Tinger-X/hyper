import cv2
import time
import torch
import argparse
from pathlib import Path
from torch.backends import cudnn

from models import experimental_re
from utils import utils_re, datasets_re, plots_re, torch_utils_re


def detect(args: "argparse.Namespace"):
    source1, source2, weights = args.source1, args.source2, args.weights
    view_img, save_txt, img_size = args.view_img, args.save_txt, args.img_size

    save_img = not args.no_save and not source1.endswith(".txt")  # save inference images
    webcam = (
            source1.isnumeric() or
            source1.endswith(".txt") or
            source1.lower().startswith(("rtsp://", "rtmp://", "http"))
    )

    # Directories
    save_dir = utils_re.increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    utils_re.set_logging()
    device = torch_utils_re.select_device(args.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = experimental_re.attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = utils_re.check_img_size(img_size, s=stride)  # check img_size
    names = model.module.names if hasattr(model, "module") else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    model_c = None  # TODO，外部需要
    if classify:
        model_c = torch_utils_re.load_classifier(name="resnet101", n=2)  # initialize
        model_c.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = utils_re.check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = datasets_re.LoadStreams(source1, img_size=img_size, stride=stride)
        dataset2 = None  # TODO，外部需要
    else:
        dataset = datasets_re.LoadImages(source1, img_size=img_size, stride=stride)
        dataset2 = datasets_re.LoadImages(source2, img_size=img_size, stride=stride)
        
    t0 = time.time()
    img_num = 0
    fps_sum = 0
    for (path, img, im0s, vid_cap), (path_, img2, im0s_, vid_cap_) in zip(dataset, dataset2):
        img = torch.from_numpy(img).to(device)
        img2 = torch.from_numpy(img2).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
        img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)

        # Inference
        t1 = torch_utils_re.time_synchronized()
        pred = model(img, img2, augment=args.augment)[0]

        # Apply NMS
        pred = utils_re.non_max_suppression(
            pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms
        )
        t2 = torch_utils_re.time_synchronized()

        # Apply Classifier
        if classify:
            pred = utils_re.apply_classifier(pred, model_c, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)
                p, s, im0_, frame = path, "", im0s_.copy(), getattr(dataset2, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # img.txt
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = utils_re.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # normalized xywh
                        xywh = (utils_re.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or args.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if args.hide_labels else (names[c] if args.hide_conf else f"{names[c]} {conf:.2f}")

                        plots_re.plot_one_box(
                            xyxy, im0,
                            label=label,
                            color=plots_re.colors(c, True),
                            line_thickness=args.line_thickness
                        )
                        plots_re.plot_one_box(
                            xyxy, im0_,
                            label=label,
                            color=plots_re.colors(c, True),
                            line_thickness=args.line_thickness
                        )
                        if args.save_crop:
                            utils_re.save_one_box(
                                xyxy, im0s, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True
                            )

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.6f}s, {1 / (t2 - t1):.6f}Hz)")
            # add all the fps
            img_num += 1
            fps_sum += 1 / (t2 - t1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    save_path_rgb = save_path.split(".")[0] + "_rgb." + save_path.split(".")[1]
                    save_path_ir = save_path.split(".")[0] + "_ir." + save_path.split(".")[1]
                    print(save_path_rgb)
                    cv2.imwrite(save_path_rgb, im0)
                    cv2.imwrite(save_path_ir, im0_)
                else:  # "video" or "stream"
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        print(f"Results saved to {save_dir}{s}")

    print(f"Done. ({time.time() - t0:.3f}s)")
    print(f"Average Speed: {fps_sum / img_num:.6f}Hz")


def parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="weights/100-best.pt", help="model.pt path(s)"
    )
    # file/folder, 0 for webcam
    parser.add_argument(
        "--source1", type=str,
        default="./Dataset/hsi_dataset/hsidetection/sa_information/images/val/", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--source2", type=str,
        default="./Dataset/hsi_dataset/hsidetection/se_information/images/val/", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", default=False, action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--no-save", action="store_true", help="do not save images/videos")
    parser.add_argument(
        "--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3"
    )
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="existing project/name ok, do not increment"
    )
    parser.add_argument("--line-thickness", default=2, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=True, action="store_true", help="hide confidences")
    return parser.parse_args()


def main():
    args = parse_args()
    utils_re.check_requirements(exclude=("pycocotools", "thop"))

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect(args=args)
                utils_re.strip_optimizer(args.weights)
        else:
            detect(args=args)


if __name__ == "__main__":
    main()
