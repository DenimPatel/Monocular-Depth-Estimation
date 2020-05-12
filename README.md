# Monocular Depth Estimation

Estimate Depth Map from a Single Image

## Training

Make changes in the params.py according to your system. 

```bash
python train.py
```
To train with temporal consistency loss
```bash
python train_with_temporal_smoothness.py
```

## Testing

Make changes in the params.py according to your system. 

```bash
python test.py
```

## Results

## Requirements
```bash
pytorch
numpy
cv2
matplotlib
tqdm
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)