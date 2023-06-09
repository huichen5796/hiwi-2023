# hiwi-2023

DONE:

- `data-local\bin\labels_correct.py` for correction of labels in all csvs and txts under `data-local\labels\chexpert`. Corrected datei will be saved under `data-local\labels\chexpert_correct_labels`. In 'log.txt' there is the correction history.
- corrected labels will transformed to ONEHOT in `data-local\bin\labels_correct.py` to characterize 4 classes of each of 5 illnesses:
   ```python
   def onehot_trans(self, label):
        if label == 1.0:
            return [1.0, 0.0, 0.0, 0.0]
        elif label == 0.0:
            return [0.0, 1.0, 0.0, 0.0]
        elif label == -1.0:
            return [0.0, 0.0, 1.0, 0.0]
        elif pandas.isna(label):
            return [0.0, 0.0, 0.0, 1.0]
        else:
            return label
   ```
- so a data will be saved in csv as form:
  ```csv
  Path,Sex,Age,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion
  CheXpert-v1.0-small/train/patient40810/study4/view1_frontal.jpg,Male,74,"[1.0, 0.0, 0.0, 0.0]","[1.0, 0.0, 0.0, 0.0]","[0.0, 0.0, 0.0, 1.0]","[1.0, 0.0, 0.0, 0.0]","[1.0, 0.0, 0.0, 0.0]"
  ```
- in `timm2\experiments\chexpert_bitm_50_1_5000_bs32 correct.py` change num_classes to 20 (5*4):
  ```python
  def parameters(): 
    defaults = {
        'dataset': 'chexpert',
        'num-classes': 20,
        ...
  ```
  So the number of channels of the final FC layer will become 20:
  ```
  (head): ClassifierHead(
    (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Identity())
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Conv2d(6144, 5, kernel_size=(1, 1), stride=(1, 1))
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  ```
  to
  ```
  (head): ClassifierHead(
    (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Identity())
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Conv2d(6144, 20, kernel_size=(1, 1), stride=(1, 1))
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  ```
  
- in `timm2\my_timm\dataset_factory.py` change function `_make_dataset(self)` unter class CheXpert:
  ```python
  def _make_dataset(self):
        df = pd.read_csv(self.file)
        imgfiles = df['Path'].values
        targets = df[self.classes].values
        targets = np.array([self.flatten(item) for item in targets], dtype=np.float64)
        samples = [s for s in zip(imgfiles, targets)]
        return samples

    def flatten(self, item):
        out = []
        for cla in item:
            out.extend(eval(cla))
        return out
  ```
  In this way, the five ONEHOT tags can be synthesized into a vector of length 20 as the target:
  ```python
  print(loader_train)
  ```
  ```
  [('CheXpert-v1.0-small/train/patient40810/study4/view1_frontal.jpg', array([1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.]))]
  ```
  
  ```
  Target shape: torch.Size([batch_size, 20])
  Target: tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                   ...], device='cuda:0')
  ```

- as loss_fn is still bce used

TODO:

- Train model in LUIS
  - training-code anpassen
  - loss_fn anpassen
