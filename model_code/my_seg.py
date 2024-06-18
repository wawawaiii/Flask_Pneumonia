import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from PIL import Image
import io

class MySEGModel(nn.Module):
  def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
    super().__init__()
    self.model = smp.create_model(
        arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
    )

    params = smp.encoders.get_preprocessing_params(encoder_name)
    
    self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
    self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE,from_logits=True)
    
  def forward(self, image):
    # normalize
    image = (image - self.mean) / self.std
    mask = self.model(image)
    return mask

  def shared_step(self, batch, stage):

    image = batch[0]
    mask = batch[1]

    logits_mask = self.forward(image)
    loss = self.loss_fn(logits_mask, mask)

    prob_mask = logits_mask.sigmoid()
    pred_mask = (prob_mask > 0.5).float()

    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

    return {
        "loss": loss,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

  def shared_epoch_end(self, outputs, stage):
      tp = torch.cat([x["tp"] for x in outputs])
      fp = torch.cat([x["fp"] for x in outputs])
      fn = torch.cat([x["fn"] for x in outputs])
      tn = torch.cat([x["tn"] for x in outputs])

      per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
      
      dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

      metrics = {
          f"{stage}_per_image_iou": per_image_iou,
          f"{stage}_dataset_iou": dataset_iou,
      }
      
      self.log_dict(metrics, prog_bar=True)

  def training_step(self, batch, batch_idx):
      return self.shared_step(batch, "train")            

  def training_epoch_end(self, outputs):
      return self.shared_epoch_end(outputs, "train")

  def test_step(self, batch, batch_idx):
      return self.shared_step(batch, "test")  

  def test_epoch_end(self, outputs):
      return self.shared_epoch_end(outputs, "test")

class my_seg_model():
    def __init__(self):
      self.cuda_available()
      self.my_segmodel_load()
      
    def cuda_available(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def my_segmodel_load(self):
    # 저장된 모델 상태를 로드할 경로
        lung_seg_model_save_path = 'C:/Users/HKIT/Desktop/Flask_Server0520/model/lung_seg_10_1e-4.pth'

        # 모델 초기화
        self.lung_seg_model = MySEGModel("FPN", "resnet50", in_channels=3, out_classes=1)

        # 저장된 모델 상태 로드
        self.lung_seg_model.load_state_dict(torch.load(lung_seg_model_save_path, map_location=self.device))
        self.lung_seg_model.to(self.device)

    def create_mask(self,image: Image.Image) -> bytes:
        image = image.convert("RGB")
        data = MySEGData()
        transform_image = data.process_image(image)
        with torch.no_grad():
            self.lung_seg_model.eval()
            pred_mask = self.lung_seg_model(transform_image)
            pr_masks = pred_mask.sigmoid()
        np_mask = pr_masks.cpu().numpy().squeeze()

        # mask로 폐사진에서 폐만 추출 ==========
        re_image = transform_image.numpy().transpose(1,2,0)
        mask2 = np_mask.copy()
        mask2[mask2 < 0.1] = 0
        mask2 = mask2.astype(bool)
        re_image[~mask2] = [0,0,0]
        # mask로 폐사진에서 폐만 추출 ==========

        re_image = (re_image * 255)
        re_image = np.array(re_image, dtype=np.uint8)
        print(np.min(np_mask), np.max(np_mask))
        lung_image = Image.fromarray(re_image)
        print(type(lung_image))
        # PIL.Image.Image
        output_buffer = io.BytesIO()
        lung_image.save(output_buffer, format="JPEG")

        return output_buffer.getvalue()
       

class MySEGData(Dataset):
    def __init__(self):
        self.image_transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((256,256)),
            T.ToTensor()
        ])
    def process_image(self, img):
        return self.image_transform(img)