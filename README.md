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

- KvasirDataset:

  - Kvasir-SEG: [test](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing), [train](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing)
  - Kvasir-SEG, CVC-ColonDB, EndoScene, ETIS-Larib Polyp DB and CVC-Clinic DB: [test](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing), [train](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
  - Kvasir-Instrument : [data](https://datasets.simula.no/kvasir-instrument/)
  - Kvasir-5folds: [data](https://drive.google.com/drive/folders/1-2RperHzW0Ea6ijBajx5fOhqM0_eg6BU?usp=sharing)

- ISICDataset:

  - [data](https://challenge.isic-archive.com/data#2018)
  - [code](utils/preprocess_isic.py) to split training isic2018 to train-val-test: 1815-259-520 ~ 70:10:20 (follow [paper](https://github.com/rezazad68/BCDU-Net/blob/master/Skin%20Lesion%20Segmentation/Prepare_ISIC2018.py))
  - [code](utils/augment_isic.py) to augment offline and resize image to low-resolution because of bottleneck
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
  - [code](utils/preprocess_nii.py) to preprocess nii file to pkl 3D image
  - Submit
    - 66 file endwith .nii.gz
    - [link](https://ipp.cbica.upenn.edu/jobs/306528931856371887)

### [Loss](network/optim/losses) (loss)

- dice_loss, structure_loss
- GeneralizedDiceLoss : for 3D & multiclass

### [Schedule](network/optim/schedulers.py) (scheduler)

- cosine, cosine_warmup

### [Model](network/model)

- [GCEE-Lambda](network/model/gcpanet/scws_lambda.py): [weight](), ~/hung/polyp_segmentation/snapshots/SCWSLambdaNet_kfold
- [GCEE-CC](network/model/gcpanet/scws_rcca.py): [weight]() , ~/hung/polyp_segmentation/snapshots/SCWSRCCANet_kfold

- [GCEE-PSP](network/model/gcpanet/scws_psp.py): [weight]()

- [GCPA-CC: GCPARCCANet](network/model/gcpanet/gcpa_rcca.py): [weight]() , ~/hung/polyp_segmentation/snapshots/GCPARCCANet_kfold
- [GCPA-PSP: GCPAPSPNet](network/model/gcpanet/gcpa_psp.py): [weight]() , /mnt/data/hungnt/snapshots/GCPAPSPNet_kfold/
- [GCPA-ASPP: GCPAASPPNet](network/model/gcpanet/gcpa_aspp.py): [weight]() , /mnt/data/hungnt/snapshots/GCPAASPPNet_kfold

- [GCPA-CGNL: GCPAGALDNetv8](network/model/gcpanet/gcpa_gald_v8.py): [weight]() , /mnt/data/hungnt/snapshots/GCPAGALDNetv8_kfold

- [GCEE-PSP](network/model/gcpanet/scws_psp.py) in Kvasir-SEG with img_size = 512: [weight]() , ~/hung/polyp_segmentation/snapshots/SCWSPSPNet_512_SEG/
- [GCEE-CC](network/model/gcpanet/scws_rcca.py) in Kvasir-instrument: [weight]() , ~/hung/polyp_segmentation/snapshots/SCWSRCCANet_instrument/

### Pretrain (put all in folder ./pretrained)

[pretrain](https://drive.google.com/drive/folders/1RO6e7j3LRgGp2HQalZejxvVuKu0jZZPq?usp=sharing)
