import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from ae.base.base_trainer import BaseTrain

class SimpleAETrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleAETrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.log_dir = None

        self.init_log_dir()
        self.init_callbacks()

    def init_log_dir(self):
        self.log_dir = os.path.join(self.root, 'logs/fit/', self.model.name)
        os.mkdir(self.log_dir)

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
                patience=6
            )
        )

        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='loss', 
                patience=3, 
                min_lr=0., 
                factor=0.1),
        )

    def train(self, ep=None):
        # if self.model.model is None:
        #     self.model.build_model(self.config)

        epochs=ep or self.config.trainer.epoch
        history = self.model.model.fit(
            self.data[0], self.data[1],
            epochs=epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
            shuffle=True
        )