import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#csv column : "name", "label" (0 ou 1), "path"
def csv_for_AudioLSTM(csv_path, name=str, label=str, path=str, output=str):
    df_audio = pd.read_csv(csv_path)
    df_out = df_audio[name, label, path]
    df_out.to_csv(output)

#extrait le csv pour pytorch
csv_path = ""
name = ""
label = ""
path = ""
csv_torch = "torchaudio_processing.csv"
csv_for_AudioLSTM(csv_path, name, label, path, csv_torch)



class AudioLSTM(Dataset):
    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []
        # récupere les infos du csv
        for i in range(0, len(csvData)):
            if csvData.iloc[i, 0] in folderList: #folder_list = liste des fichiers du folder
                self.file_names.append(csvData.iloc[i, 0]) #i = ligne et 5 = colonne
                self.labels.append(csvData.iloc[i, 1])
                self.folders.append(csvData.iloc[i, 2])

        self.file_path = file_path
        self.folderList = folderList

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        sound, sample_rate = torchaudio.load(path, out=None, normalization=True)
        soundData = torch.mean(sound, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])  # tempData accounts for audio clips that are too short

        if soundData.numel() < 160000:
            tempData[:, :soundData.numel()] = soundData
        else:
            tempData = soundData[:, :160000]

        soundData = tempData

        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(soundData)  # (channel, n_mels, time)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)  # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        feature = torch.cat([mel_specgram, mfcc], axis=1)
        return feature[0].permute(1, 0), self.labels[index]

    def __len__(self):
        return len(self.file_names)
    

class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden
    

def train(model, epoch):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output, _ = model(data, model.init_hidden(hyperparameters["batch_size"]))

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        total_loss += loss.item()

    print(f"Train Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")

            
def test(model, epoch):
    model.eval()
    correct = 0
    y_pred, y_target = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output, _ = model(data, model.init_hidden(hyperparameters["batch_size"]))

            pred = torch.max(output, dim=1).indices
            correct += pred.eq(target).sum().item()

            y_pred += pred.cpu().tolist()
            y_target += target.cpu().tolist()

    acc = 100 * correct / len(test_loader.dataset)
    print(f"\nTest Epoch {epoch} | Accuracy: {acc:.2f}%\n")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_target, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # --- Report ---
    print(classification_report(y_target, y_pred))


hyperparameters = {"lr": 0.01, "weight_decay": 0.0001, "batch_size": 128, "in_feature": 168, "out_feature": 10}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#Entrainement
csv_path = csv_torch
file_path = ''

train_set = AudioLSTM(csv_path, file_path, range(1, 10))
test_set = AudioLSTM(csv_path, file_path, [10])
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)

model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping

log_interval = 10
for epoch in range(1, 41):
    # scheduler.step()
    train(model, epoch)
    test(model, epoch)