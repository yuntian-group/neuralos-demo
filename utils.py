import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import json
import os
import time

def load_model_from_config(config_path, model_name, device='cuda'):
    # Load the config file
    config = OmegaConf.load(config_path)
    
    # Instantiate the model
    model = instantiate_from_config(config.model)
    
    # Download the model file from Hugging Face
    model_file = hf_hub_download(repo_id=model_name, filename="model.safetensors", token=os.getenv('HF_TOKEN'))
    
    print(f"Loading model from {model_name}")
    # Load the state dict
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def sample_frame(model: LatentDiffusion, prompt: str, image_sequence: torch.Tensor):
    sampler = DDIMSampler(model)
    
    with torch.no_grad():
        u_dict = {'c_crossattn': "", 'c_concat': image_sequence}
        uc = model.get_learned_conditioning(u_dict)
        uc = model.enc_concat_seq(uc, u_dict, 'c_concat')
        
        c_dict = {'c_crossattn': prompt, 'c_concat': image_sequence}
        c = model.get_learned_conditioning(c_dict)
        c = model.enc_concat_seq(c, c_dict, 'c_concat')

        print ('sleeping')
        time.sleep(120)
        print ('finished sleeping')
        samples_ddim = model.p_sample_loop(cond=c, shape=[1, 3, 64, 64], return_intermediates=False, verbose=True)
        #samples_ddim, _ = sampler.sample(S=999,
        #                                 conditioning=c,
        #                                 batch_size=1,
        #                                 shape=[3, 64, 64],
        #                                 verbose=False,
        #                                 unconditional_guidance_scale=5.0,
        #                                 unconditional_conditioning=uc,
        #                                 eta=0)
        
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        #x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)
        
        return x_samples_ddim.squeeze(0).cpu().numpy()

# Global variables for model and device
#model = None
#device = None

def initialize_model(config_path, model_name):
    #global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config_path, model_name, device)
    return model