import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from ae.base.base_trainer import BaseTrain

class SimpleAETrainer(BaseTrain):
    def __init__(self, model, config):
        super(SimpleAETrainer, self).__init__(model, config)
        self.callbacks = []
        self.log_dir = None
        self.x_train = None

        self.init_log_dir()
        self.init_callbacks()

    def init_log_dir(self):
        self.log_dir = os.path.join(self.root, "logs", self.config.trainer.logdir, self.model.name)
        try: 
            os.mkdir(self.log_dir)
        except:
            pass


    def init_callbacks(self):
        # self.callbacks.append(
        #     ModelCheckpoint(
        #         filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
        #         monitor=self.config.callbacks.checkpoint_monitor,
        #         mode=self.config.callbacks.checkpoint_mode,
        #         save_best_only=self.config.callbacks.checkpoint_save_best_only,
        #         save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
        #         verbose=self.config.callbacks.checkpoint_verbose,
        #     )
        # )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
                histogram_freq=1,
                embeddings_freq=1
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor='loss', 
                patience=12
            )
        )

        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss', 
                patience=4, 
                min_lr=0., 
                factor=0.1),
        )

    def train(self, data, ep=None):
        # if self.model.model is None:
        #     self.model.build_model(self.config)
        self.x_train = data[0]
        epochs=ep or self.config.trainer.epoch
        history = self.model.model.fit(
            data[0], data[1],
            epochs=epochs,
            verbose=self.config.trainer.verbose,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
            shuffle=True
        )

    def eval(self, full_eigv):
        self.eigv = full_eigv[:, :self.model.input_dim]    
        self.ae_pred = self.model.model.predict(self.x_train)
        self.flux_org = self.x_train.dot(self.eigv.T)
        self.flux_rec = self.ae_pred.dot(self.eigv.T)
        self.abs_err = abs(self.flux_org - self.flux_rec)
