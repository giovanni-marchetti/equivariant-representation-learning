# A Universal Approach to Equivariant Representation Learning

## Setup

```
python 3.6+
pip install -f requirements.txt
```

## Data

Sprites:
`python get_data.py --dataset sprites`

Multi-sprites:
`python get_data.py --dataset multi-sprites`

Color-shift:
`python get_data.py --dataset colorshift`

Platonics:
`python get_data.py --dataset platonics`

### Licenses
The dataset with the room is constructed using 3D-models from the following sources with licenses:
https://free3d.com/3d-model/room-48457.html (Personal Use License)
https://www.cgtrader.com/free-3d-models/interior/hall/hall-room (Royalty Free License)
https://www.cgtrader.com/free-3d-models/furniture/bed/bed-free8 (Royalty Free License)
https://free3d.com/3d-model/room-93514.html8 (Personal Use License)




## Experiments

### Sprites
`python main.py --dataset 'sprites' --action-dim 3 --extra-dim 2 --model-name sprites --decoder`

### Color-shift
`python main.py --dataset color-shift --action-dim 3 --extra-dim 2 --model-name color-shift --decoder`

### Multi-sprites
`python main.py --dataset multi-sprites --action-dim 6 --extra-dim 2 --model-name multi-sprites --decoder`

### Platonics 
`python main.py --dataset 'platonics' --action-dim 4 --extra-dim 2 --model-name platonics --decoder`

### Platonics with linear action
`python main.py --dataset 'platonics' --action-dim 3 --extra-dim 0 --model-name platonics-naive --decoder --method naive`

### Room
`python main.py --dataset room_combined action-dim 4 --extra-dim 2 --model-name room`
