from datasets import Dataset, DatasetDict, Audio, ClassLabel, Features, Value
import os
from pathlib import Path

""" 
Schema dataset dossier

data/
├── train/
│   ├── oui/
│   │   ├── oui_001.wav
│   │   └── oui_002.wav
│   ├── non/
│   │   ├── non_001.wav
│   └── maybe/
│       └── maybe_001.wav
├── validation/
│   └── ...
└── test/
    └── ... """



def load_split(split_dir):
    files, labels = [], []
    
    classes = sorted(os.listdir(split_dir))  # ["maybe", "non", "oui"]
    
    for label_name in classes:
        label_dir = Path(split_dir) / label_name
        if not label_dir.is_dir():
            continue
        for audio_file in label_dir.glob("*.wav"):  # ou *.mp3, *.flac...
            files.append(str(audio_file))
            labels.append(label_name)
    
    return {"file": files, "label": labels}

# Charger chaque split
train_data      = load_split("data/train")
validation_data = load_split("data/validation")
test_data       = load_split("data/test")

# Définir les classes (doit être identique pour tous les splits)
all_classes = sorted(set(train_data["label"]))

# Définir le schéma des features
features = Features({
    "file":  Value("string"),
    "audio": Audio(sampling_rate=16000),   # adapte le sample rate
    "label": ClassLabel(names=all_classes)
})

# Créer les Dataset objects
def make_dataset(data, features):
    ds = Dataset.from_dict(data)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))  # charge l'audio depuis "file"
    ds = ds.cast_column("label", features["label"])
    return ds

train_ds = make_dataset(train_data, features)
val_ds   = make_dataset(validation_data, features)
test_ds  = make_dataset(test_data, features)

# Assembler le DatasetDict final
dataset = DatasetDict({
    "train":      train_ds,
    "validation": val_ds,
    "test":       test_ds
})

print(dataset)