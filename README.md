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

- KvasirDataset:

  - Kvasir-SEG: [test](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing), [train](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing)
  - Kvasir-SEG, CVC-ColonDB, EndoScene, ETIS-Larib Polyp DB and CVC-Clinic DB: [test](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing), [train](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
  - Kvasir-Instrument : [data](https://datasets.simula.no/kvasir-instrument/)

- ISICDataset:

  - [data](https://challenge.isic-archive.com/data#2018)

- USNerveDataset:

  - [data](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data?fbclid=IwAR3Rly_-HfPylAAHSbEiX5a9Pt42VSXPwou4WEnuNHjl5GML5VOKrhLH2Ik)

- BraTSDataset:
  - [data](https://www.med.upenn.edu/sbia/brats2018/data.html)

### [Loss](network/optim/losses) (loss)

- dice_loss, structure_loss
- GeneralizedDiceLoss : for 3D & multiclass

### [Schedule](network/optim/schedulers.py) (scheduler)

- cosine, cosine_warmup
