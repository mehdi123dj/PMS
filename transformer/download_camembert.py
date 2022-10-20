
from farm.modeling.language_model import LanguageModel
import os 


SAVE_DIR = os.path.join(os.path.dirname(__file__), "models",'camembert_large')
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
print(SAVE_DIR)

L=LanguageModel.load(pretrained_model_name_or_path = "camembert/camembert-large")
L.save(SAVE_DIR)

