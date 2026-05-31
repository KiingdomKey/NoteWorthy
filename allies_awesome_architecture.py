from constants import *

import torch
import lightning.pytorch as pl
import numpy as np



class ConvLayer(pl.LightningModule):
    def __init__(self,
                 in_size,
                 out_size,
                 residual: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.residual = residual
        self.conv2d = torch.nn.Conv2d(in_size, out_size, (3, 3), padding=(1, 1))
        self.batchnorm = torch.nn.BatchNorm2d(out_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        y = x
        y = self.conv2d(y)
        y = self.batchnorm(y)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y



class RecurrentResidualLayer(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(latent_size)
        self.rnn_layer = torch.nn.LSTM(latent_size,
                                       latent_size,
                                       batch_first=True)
    def forward(self, x):
        return x + self.rnn_layer(self.layer_norm(x))[0]



def calculate_loss(predictions, targets):
    
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []

    if not torch.is_tensor(predictions):
        predictions = torch.tensor(predictions)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    
    for i in range(targets.shape[1]):
        targets_slice = targets[:, :, i]
        targets_slice = torch.tensor(targets_slice).to(int)
        if (len(targets_slice.shape) < 2):
            targets_slice = targets_slice.unsqueeze(dim=0)

        loss = loss_function(predictions[:, :, :, i], targets_slice[:, :]).sum()
        losses += [loss]

    return losses



class RCNN(pl.LightningModule):

    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)


        latent_size = 16

        self.projection = torch.nn.Conv2d(1, latent_size,
                                          kernel_size=(1, 1))

        conv_layers = []
        for _ in range(2):
            conv_layers += [ConvLayer(in_size=latent_size,
                                            out_size=latent_size,
                                            residual=True)]
            conv_layers += [ConvLayer(in_size=latent_size,
                                           out_size=latent_size * 2,
                                           residual=False)]
            conv_layers += [torch.nn.MaxPool2d(2, 2)]
            latent_size *= 2

        conv_layers += [torch.nn.MaxPool2d((2, 2))]

        self.conv_layers = torch.nn.Sequential(*conv_layers)

        self.flatten = torch.nn.Flatten()

        POST_FLATTEN_SIZE = 64 * 11 * 2

        lstm_layers = []

        for _ in range(1):
            lstm_layers += [RecurrentResidualLayer(latent_size=POST_FLATTEN_SIZE)]

        self.lstm_layers = torch.nn.Sequential(*lstm_layers)
        self.output_layer = torch.nn.LSTM(input_size=POST_FLATTEN_SIZE,
                                          hidden_size=88 * 4,
                                          batch_first=True)

        self.output_activation = torch.nn.Softmax(dim=2)

    def forward(self, x):
        # Enters as batch, pitch, time
        y = x
        total_time_steps = y.shape[2]
        predictions = []

        for i in range(0, total_time_steps, TIME_STEP_LENGTH):

            if i + TIME_STEP_LENGTH >= total_time_steps:
                time_slice = y[:, :, i:]
                padding_amount = ((i + TIME_STEP_LENGTH) % total_time_steps)
                time_slice = torch.nn.functional.pad(time_slice, (0, padding_amount, 0, 0, 0, 0,), value=0)
            else:
                time_slice = y[:, :, i: i + TIME_STEP_LENGTH]

            time_slice = time_slice.unsqueeze(dim=1)

            time_slice = self.projection(time_slice)
            time_slice = self.conv_layers(time_slice)
            time_slice = self.flatten(time_slice).unsqueeze(dim=1)

            predictions += [time_slice]

        predictions = torch.cat(predictions, axis=1)

        predictions = self.lstm_layers(predictions)
        predictions = self.output_layer(predictions)[0]

        predictions = torch.unflatten(predictions, dim=2, sizes=(4, 88))

        predictions = predictions.permute(0, 2, 3, 1)

        return predictions

    def predict(self,
                x):
        y = x
        y = self(y)
        y = self.output_activation(y)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0
        }

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        predictions = self(data)

        loss = sum(calculate_loss(predictions, labels))
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        predictions = self(data)

        with torch.no_grad():
            loss = sum(calculate_loss(predictions, labels))

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    