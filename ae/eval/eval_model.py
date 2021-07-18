import os
import numpy as np
import pandas as pd
import seaborn as sns
from ae.model.simple_ae_model import SimpleAEModel
from ae.model.vae_model import VAEModel
from ae.trainer.simple_ae_trainer import SimpleAETrainer


def eval_model(x_train, x_test, config, df_test, load=None):
    if load is not None:
        MODEL_PATH="/home/swei20/AE/trained_model"
        MODEL_PATH = os.path.join(MODEL_PATH, load) 
    if config.model.type == "ae":
        mm = SimpleAEModel()    
        mm.build_model(config)
        mm.load(MODEL_PATH)
        pred = mm.encoder.predict(x_test)

    elif config.model.type == "vae":
        mm = VAEModel()
        mm.build_model(config)
        mm.load(MODEL_PATH)
        pred = mm.encoder.predict(x_test)[-1]

    # tt = SimpleAETrainer(mm, config)
    # data = (x_train, x_train)
    # tt.train(data)
    idx = np.arange(len(x_test)) if df_test is None else df_test.index
    df_encode = pd.DataFrame(data= pred, columns = [f"{config.model.type}{i}" for i in range(1, mm.latent_dim+1)], index = idx)
    df = pd.concat([df_test, df_encode], axis=1)
    return df

def plot_model(df, para, type):
    sns.pairplot(
        df,
        x_vars=[f"{type}{i}" for i in range(1,6)],
        y_vars=[f"{type}{i}" for i in range(1,6)],
        hue=para,
        plot_kws=dict(marker="o", s=2, edgecolor="none"),
        diag_kws=dict(fill=False),
        palette="Spectral"
    )