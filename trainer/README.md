# Registering new model for training

1. register new model in class `TrainerKeys` in `keys.py`
2. register appropriate dataset association with model in `getDataset(model)` function in `trainer/main.py`
3. register appropriate model parameter defaults class in `trainer/params.py`
4. create model training file using the provided template below
5. register new model in `train(model)` function in `trainer/main.py`

---

**Model training template**

```python
from trainer.params import xyz

# Must define function "train_main"
def train_main() -> str:
    model_params = xyz()
    cwd = Path(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = # template model class initialization
    # 1. train model with model_params on training set
    # 2. perform inference on test set
    # 3. create folder ./runs
    # 4. save model weights, inferences, graphs, results, etc in the runs folder
    #    for inferences calculations, use batch size of 1 on the training set
    # 5. create string with summary results as follows and return it
    # """class 1 iou: ...
    #    class 2 iou: ...
    #    ...
    #    Average IoU: ...
    #    Average inference time: ..."""
```
