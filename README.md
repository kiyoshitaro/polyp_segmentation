Train

```sh
CUDA_VISIBLE_DEVICES=0 python my_train.py -c configs/gcpa_gald_net_config.yaml
```

Test

```sh

CUDA_VISIBLE_DEVICES=0 python my_test.py -c configs/gcpa_gald_net_config.yaml
```

## Option

### [Dataloader](dataloader) (dataloader. name)

All data must put in ./data, follow the tree:

```bash
.
├── Kvasir_SEG_Training_880
│   ├── images
│   │   ├── cju0qkwl35piu0993l0dewei2.png
│   │   ├── cju0qoxqj9q6s0835b43399p4.png
│   └── masks
│       ├── cju0qkwl35piu0993l0dewei2.png
│       ├── cju0qoxqj9q6s0835b43399p4.png
└── Kvasir_SEG_Validation_120
    ├── images
    │   ├── cju0s690hkp960855tjuaqvv0.png
    │   ├── cju0sr5ghl0nd08789uzf1raf.png
    └── masks
        ├── cju0s690hkp960855tjuaqvv0.png
        ├── cju0sr5ghl0nd08789uzf1raf.png
```

with BraTS2018 run:

```bash
    python utils/preprocess_nii.py
```

- KvasirDataset:

  - Kvasir-SEG: [test](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing), [train](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing)
  - Kvasir-SEG, CVC-ColonDB, EndoScene, ETIS-Larib Polyp DB and CVC-Clinic DB: [test](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing), [train](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
  - Kvasir-Instrument : [data](https://datasets.simula.no/kvasir-instrument/)

- ISICDataset:

  - [data](https://challenge.isic-archive.com/data#2018)
  - Submit :
    - 1000 images with name ISIC\_<image_id>\_segmentation.png
      - 0: representing the background of the image, or areas outside the primary lesion
      - 255: representing the foreground of the image, or areas inside the primary lesion
    - metric : MeanIou
    - [link](https://challenge.isic-archive.com/task/49)

- USNerveDataset:

  - [data](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data?fbclid=IwAR3Rly_-HfPylAAHSbEiX5a9Pt42VSXPwou4WEnuNHjl5GML5VOKrhLH2Ik)
  - Submit:
    - run mytest_usnerve.py
    - 5508 images encoded with rle into submission.csv

- BraTSDataset:
  - [data](https://www.med.upenn.edu/sbia/brats2018/data.html)
  - Submit
    - 66 file endwith .nii.gz
    - [link](https://ipp.cbica.upenn.edu/jobs/306528931856371887)

### [Loss](network/optim/losses) (loss)

- dice_loss, structure_loss
- GeneralizedDiceLoss : for 3D & multiclass

### [Schedule](network/optim/schedulers.py) (scheduler)

- cosine, cosine_warmup

### Pretrain (put all in folder ./pretrained)

[pretrain](https://drive.google.com/drive/folders/1RO6e7j3LRgGp2HQalZejxvVuKu0jZZPq?usp=sharing)
