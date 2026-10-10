# type: ignore

from src.models import BaseModel, ModelBuilderEngine, FeatureProcessingPolicy


class ModelEngine(ModelBuilderEngine):
    
    def __init__(self, policy: FeatureProcessingPolicy, model: BaseModel) -> None:
        super().__init__(policy, model)
        
    def build(self) -> BaseModel:
        modelDataset = self.policy.fetch()
        self.model.train(modelDataset)
        #self.model.evaluate(modelDataset)
        #self.model.save()
        return self.model