# directly extract embeddings 
# after move to env: pip install modelgenerator 
from modelgenerator.tasks import Embed
import torch

model = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
transformed_batch = model.transform({"sequences": ["ACGT", "AGCT"]})
embedding = model(transformed_batch)
torch.save(embedding.cpu(), "embeddings.pt") # first time 
# embedding = torch.load("embeddings.pt")  # later 

print(embedding.shape)
print(embedding)
