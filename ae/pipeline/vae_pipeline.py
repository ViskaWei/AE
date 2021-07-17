from ae.pipeline.simple_ae_pipeline import AEPipeline

class VAEPipeline(AEPipeline):
    def __init__(self, logging=True, trace=None):
        super().__init__(logging=logging, trace=trace)


    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument("--stddev", type=float, default=None)

    def prepare(self):
        super().prepare()
        
    def apply_model_args(self):
        self.update_config("model", "stddev")
        super().apply_model_args()

    def execute(self):
        super().execute()