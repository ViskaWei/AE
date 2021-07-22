import os
import numpy as np
import pandas as pd
import seaborn as sns
from ae.model.simple_ae_model import SimpleAEModel
from ae.model.vae_model import VAEModel
from ae.trainer.simple_ae_trainer import SimpleAETrainer


def eval_model(x_train, x_test, config, df_test, load=None):
    model_type = config.model.type
    mm = SimpleAEModel() if model_type == "ae" else VAEModel()
    mm.build_model(config)
    if load is not None:
        MODEL_PATH=os.path.join(os.environ["home"], "trained_model", model_type,load, "")
        # f"/home/swei20/AE/trained_model/{model_type}/{load}/"
        mm.load(MODEL_PATH)

    pred = mm.encoder.predict(x_test)
    if config.model.type == "vae":
        pred = pred[0]
    # return pred
    
    # _,s,v = np.linalg.svd(pred)
    # v = v.T
    idx = np.arange(len(x_test)) if df_test is None else df_test.index
    df_encode = pd.DataFrame(data= pred, columns = [f"{model_type}{i}" for i in range(1, mm.latent_dim+1)], index = idx)
    df = pd.concat([df_test, df_encode], axis=1)
    return df, mm



def plot_model(df, para, type):
    sns.pairplot(
        df,
        x_vars=[f"{type}{i}" for i in range(1,6)],
        y_vars=[f"{type}{i}" for i in range(1,6)],
        hue=para,
        plot_kws=dict(marker="o", s=2, edgecolor="none"),
        diag_kws=dict(fill=False, alpha=0.3),
        palette="Spectral"
    )