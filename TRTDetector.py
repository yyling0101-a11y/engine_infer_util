import os
import json
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional

import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import time


@dataclass
class SimpleBoxes:
    xyxy: torch.Tensor  # CPU, [N, 4]
    conf: torch.Tensor  # CPU, [N]
    cls: torch.Tensor  # CPU, [N]

    @property
    def xywh(self) -> torch.Tensor:
        if self.xyxy.numel() == 0:
            return torch.empty((0, 4), dtype=self.xyxy.dtype)
        x1, y1, x2, y2 = (
            self.xyxy[:, 0],
            self.xyxy[:, 1],
            self.xyxy[:, 2],
            self.xyxy[:, 3],
        )
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        return torch.stack([cx, cy, w, h], dim=1)


class SimpleResult:
    def __init__(
            self,
            boxes: SimpleBoxes,
            names: Optional[dict] = None,
            img_gpu: Optional[torch.Tensor] = None,  # HWC BGR CUDA
            path: str = "",
    ):
        self.boxes = boxes
        self.names = names or {}
        self.img_gpu = img_gpu
        self.path = path

    def plot(self, line_width: int = 2, font_scale: float = 0.5) -> "cv2.Mat":
        if self.img_gpu is None:
            raise ValueError(
                "No image stored. Set keep_image_for_plot=True when creating TRTDetector."
            )

        img = self.img_gpu.detach()
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"plot() expects stored HWC BGR image, got {tuple(img.shape)}"
            )

        if torch.is_floating_point(img):
            if img.max() <= 1.0:
                img = (img * 255.0).round()
            img = img.clamp(0, 255).to(torch.uint8)
        else:
            img = img.clamp(0, 255).to(torch.uint8)

        img = img.cpu().numpy()  # 仅 plot 时搬运图像

        if self.boxes is None or self.boxes.xyxy.numel() == 0:
            return img

        xyxy = self.boxes.xyxy.numpy().astype("int32")
        conf = self.boxes.conf.numpy()
        cls = self.boxes.cls.numpy().astype("int32")

        for box, score, cls_id in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = box.tolist()
            name = self.names.get(int(cls_id), str(cls_id))
            label = f"{name} {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_width)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            y_text = max(y1, th + 2)
            cv2.rectangle(
                img,
                (x1, y_text - th - 2),
                (x1 + tw, y_text + baseline - 2),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return img


class TRTDetector:
    def __init__(
            self,
            engine_path: str,
            device: str = "cuda:0",
            imgsz: int = 640,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            max_det: int = 300,
            class_agnostic: bool = False,
            keep_image_for_plot: bool = False,
    ):
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.class_agnostic = bool(class_agnostic)
        self.keep_image_for_plot = bool(keep_image_for_plot)

        self.logger = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"engine not found: {engine_path}")

        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata_bytes = f.read(meta_len)
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                self.metadata = metadata
                self.names = (
                    {int(k): v for k, v in metadata.get("names", {}).items()}
                    if isinstance(metadata.get("names", {}), dict)
                    else {}
                )
                engine_bytes = f.read()
                print(
                    f"[INFO] Detected Ultralytics metadata header, meta_len={meta_len}"
                )
            except Exception:
                f.seek(0)
                engine_bytes = f.read()
                self.metadata = {}
                self.names = {}
                print("[INFO] Raw TensorRT engine loaded (no Ultralytics metadata).")

        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed")

        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = []
        self.output_names = []
        for name in self.tensor_names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expect 1 input tensor, got {self.input_names}")
        if len(self.output_names) == 0:
            raise RuntimeError("No output tensor found")

        self.input_name = self.input_names[0]
        self.input_dtype = self.engine.get_tensor_dtype(self.input_name)

        # 静态 shape 信息
        self.engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(self.engine_input_shape) != 4:
            raise RuntimeError(
                f"Only 4D input supported, got {self.engine_input_shape}"
            )

        self.engine_batch = int(self.engine_input_shape[0])
        self.engine_h = int(self.engine_input_shape[2])
        self.engine_w = int(self.engine_input_shape[3])

        # 若用户没特意传 imgsz，以 engine 为准
        if self.imgsz != self.engine_h or self.imgsz != self.engine_w:
            print(
                f"[WARN] imgsz={self.imgsz} but engine expects "
                f"({self.engine_h}, {self.engine_w}), using engine size."
            )
            self.imgsz = self.engine_h

        # 预分配输出缓存
        self.output_buffers = {}
        self.output_specs = {}
        self._allocate_output_buffers(self.engine_batch)

    @staticmethod
    def _trt_dtype_to_torch(dtype):
        if dtype == trt.DataType.FLOAT:
            return torch.float32
        if dtype == trt.DataType.HALF:
            return torch.float16
        if dtype == trt.DataType.INT32:
            return torch.int32
        if dtype == trt.DataType.INT8:
            return torch.int8
        if dtype == trt.DataType.BOOL:
            return torch.bool
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")

    def _allocate_output_buffers(self, batch_size: int):
        self.output_buffers = {}
        self.output_specs = {}

        # 静态 engine 下，batch 必须匹配
        if batch_size != self.engine_batch:
            raise RuntimeError(
                f"Engine fixed batch is {self.engine_batch}, but requested {batch_size}"
            )

        ok = self.context.set_input_shape(
            self.input_name, (batch_size, 3, self.imgsz, self.imgsz)
        )
        if not ok:
            raise RuntimeError(
                f"set_input_shape failed for {(batch_size, 3, self.imgsz, self.imgsz)}"
            )

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_dtype_to_torch(dtype)
            buf = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.output_buffers[name] = buf
            self.output_specs[name] = {"shape": shape, "dtype": torch_dtype}

    def warmup(self, num_iters: int = 5):
        dummy = torch.zeros(
            (self.engine_batch, 3, self.imgsz, self.imgsz),
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(num_iters):
            self._infer_raw(dummy)
        torch.cuda.synchronize(self.device)

    def _prepare_one(self, img_gpu: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        输入:
            HWC, BGR, CUDA Tensor
        输出:
            x: (1,3,imgsz,imgsz), RGB, float32, CUDA
            meta: 后处理还原信息
        """
        if not isinstance(img_gpu, torch.Tensor):
            raise TypeError("Each input must be torch.Tensor")
        if not img_gpu.is_cuda:
            raise ValueError("Input image must be CUDA tensor")
        if img_gpu.ndim != 3 or img_gpu.shape[-1] != 3:
            raise ValueError(
                f"Expect HWC BGR CUDA tensor, got shape={tuple(img_gpu.shape)}"
            )

        h, w = img_gpu.shape[:2]

        if img_gpu.dtype != torch.uint8:
            if torch.is_floating_point(img_gpu):
                if img_gpu.max() <= 1.0:
                    img_gpu = (img_gpu * 255.0).round().clamp(0, 255).to(torch.uint8)
                else:
                    img_gpu = img_gpu.round().clamp(0, 255).to(torch.uint8)
            else:
                img_gpu = img_gpu.to(torch.uint8)

        if h == self.imgsz and w == self.imgsz:
            x = img_gpu[..., [2, 1, 0]]  # BGR -> RGB
            x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
            x = x.to(torch.float32) / 255.0
            meta = {"orig_shape": (h, w), "ratio": 1.0, "pad": (0, 0)}
            return x, meta

        r = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        dw = (self.imgsz - new_w) // 2
        dh = (self.imgsz - new_h) // 2

        x = img_gpu.permute(2, 0, 1).contiguous().unsqueeze(0).float()  # 1,3,H,W BGR
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = x.round().clamp(0, 255).to(torch.uint8)

        canvas = torch.full(
            (1, 3, self.imgsz, self.imgsz),
            114,
            dtype=torch.uint8,
            device=img_gpu.device,
        )
        canvas[:, :, dh: dh + new_h, dw: dw + new_w] = x
        canvas = canvas[:, [2, 1, 0], :, :].contiguous()  # BGR -> RGB
        canvas = canvas.to(torch.float32) / 255.0

        meta = {"orig_shape": (h, w), "ratio": r, "pad": (dw, dh)}
        return canvas, meta

    def _prepare_batch(
            self, imgs_gpu: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[dict]]:
        batch_size = len(imgs_gpu)
        if batch_size != self.engine_batch:
            raise ValueError(
                f"This engine requires batch={self.engine_batch}, got {batch_size}"
            )

        xs = []
        metas = []
        for img in imgs_gpu:
            x, meta = self._prepare_one(img)
            xs.append(x)
            metas.append(meta)

        x = torch.cat(xs, dim=0)  # [B,3,H,W]
        return x, metas

    def _infer_raw(self, x: torch.Tensor):
        if not x.is_cuda or x.ndim != 4:
            raise ValueError(f"Expect CUDA BCHW tensor, got {tuple(x.shape)}")

        b, c, h, w = x.shape
        if c != 3 or h != self.imgsz or w != self.imgsz:
            raise ValueError(
                f"Expect shape [B,3,{self.imgsz},{self.imgsz}], got {tuple(x.shape)}"
            )
        if b != self.engine_batch:
            raise ValueError(f"Engine fixed batch is {self.engine_batch}, got {b}")

        ok = self.context.set_input_shape(self.input_name, tuple(x.shape))
        if not ok:
            raise RuntimeError(
                f"set_input_shape failed: {self.input_name}, shape={tuple(x.shape)}"
            )

        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        for name in self.output_names:
            self.context.set_tensor_address(
                name, int(self.output_buffers[name].data_ptr())
            )

        stream = torch.cuda.current_stream(device=self.device)
        self.context.execute_async_v3(stream.cuda_stream)

        return self.output_buffers

    @staticmethod
    def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
        out = boxes.clone()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
        out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
        out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
        out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
        return out

    @staticmethod
    def _scale_boxes_to_original(boxes_xyxy: torch.Tensor, meta: dict) -> torch.Tensor:
        dw, dh = meta["pad"]
        r = meta["ratio"]
        orig_h, orig_w = meta["orig_shape"]

        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= r

        boxes[:, 0].clamp_(0, orig_w - 1)
        boxes[:, 1].clamp_(0, orig_h - 1)
        boxes[:, 2].clamp_(0, orig_w - 1)
        boxes[:, 3].clamp_(0, orig_h - 1)
        return boxes

    def _postprocess_one(
            self,
            _pred_one: torch.Tensor,  # [C, N] or [84, 8400]
            _meta: dict,
            _img_gpu: Optional[torch.Tensor],
            _classes=None,
    ) -> SimpleResult:
        pred = _pred_one.transpose(0, 1).contiguous()  # [N, C]
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]

        conf, cls = cls_scores.max(dim=1)

        keep = conf > self.conf_thres

        # 类别过滤
        if _classes is not None:
            classes_tensor = torch.as_tensor(_classes, device=cls.device)
            keep = keep & (cls.unsqueeze(1) == classes_tensor).any(dim=1)

        if keep.sum() == 0:
            return SimpleResult(
                boxes=SimpleBoxes(
                    xyxy=torch.empty((0, 4), dtype=torch.float32),
                    conf=torch.empty((0,), dtype=torch.float32),
                    cls=torch.empty((0,), dtype=torch.float32),
                ),
                names=self.names,
                img_gpu=_img_gpu if self.keep_image_for_plot else None,
            )

        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls = cls[keep]

        boxes_xyxy = self._xywh2xyxy(boxes_xywh)

        if self.class_agnostic:
            nms_boxes = boxes_xyxy
        else:
            max_wh = 7680
            nms_boxes = boxes_xyxy + cls.unsqueeze(1) * max_wh

        keep_idx = nms(nms_boxes, conf, self.iou_thres)
        keep_idx = keep_idx[: self.max_det]

        boxes_xyxy = boxes_xyxy[keep_idx]
        conf = conf[keep_idx]
        cls = cls[keep_idx]

        boxes_xyxy = self._scale_boxes_to_original(boxes_xyxy, _meta)

        # 仅 boxes/conf/cls 搬到 CPU
        return SimpleResult(
            boxes=SimpleBoxes(
                xyxy=boxes_xyxy.detach().cpu(),
                conf=conf.detach().cpu(),
                cls=cls.float().detach().cpu(),
            ),
            names=self.names,
            img_gpu=_img_gpu if self.keep_image_for_plot else None,
        )

    def _postprocess_batch(
            self,
            _outputs: dict,
            _metas: List[dict],
            _imgs_gpu: List[torch.Tensor],
            _classes=None
    ) -> List[SimpleResult]:
        # 常见 Ultralytics detect 输出: [B, 84, 8400]
        pred = _outputs[self.output_names[0]]
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected output shape: {tuple(pred.shape)}")

        b = pred.shape[0]
        if b != len(_metas):
            raise RuntimeError(f"Output batch={b}, but metas={len(_metas)}")

        _results = []
        for i in range(b):
            img_ref = _imgs_gpu[i] if self.keep_image_for_plot else None
            _results.append(self._postprocess_one(pred[i], _metas[i], img_ref, _classes))
        return _results

    @torch.no_grad()
    @torch.no_grad()
    def infer(
            self, imgs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
            classes=None,
    ):
        """
        支持：
            result = det.infer(img_gpu)          # engine_batch 必须为 1
            results = det.infer([img0, img1])   # engine_batch 必须匹配输入长度
        输入：
            HWC BGR CUDA Tensor
        返回：
            单张 -> SimpleResult
            多张 -> list[SimpleResult]
        """
        single = isinstance(imgs, torch.Tensor)
        imgs_list = [imgs] if single else list(imgs)

        x, _metas = self._prepare_batch(imgs_list)
        _outputs = self._infer_raw(x)

        _results = self._postprocess_batch(_outputs, _metas, imgs_list, classes)

        if single:
            return _results[0]
        return _results


if __name__ == "__main__":
    det = TRTDetector(
        "/home/nvidia/engine_infer_util/model/yolo11s_batch2.engine",
        device="cuda:0",
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        keep_image_for_plot=True,
    )

    det.warmup(3)

    img0 = cv2.imread("bus.jpg")
    img0_gpu = torch.from_numpy(img0).to("cuda:0", non_blocking=True)
    print(det.engine_batch)
    while True:
        try:
            start = time.time()
            if det.engine_batch == 1:
                result = det.infer(img0_gpu)
                print("num boxes =", len(result.boxes.conf))
                vis = result.plot()
                cv2.imshow("result", vis)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    exit()
            else:
                imgs_gpu = [img0_gpu for _ in range(det.engine_batch)]
                results = det.infer(imgs_gpu)
                for i, result in enumerate(results):
                    print(f"[{i}] num boxes =", len(result.boxes.conf))
                    vis = result.plot()
                    cv2.imshow("result", vis)
                    key = cv2.waitKey(0)
                    if key == ord("q"):
                        exit()
            end = time.time()
            print(f"infer time = {(end - start) * 1000:.4f} ms")

        except KeyboardInterrupt:
            break
