import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import RobertaModel, ViTModel, AutoTokenizer
import torch.backends.cudnn as cudnn
from torch.distributions import Normal
from utils.libs import *


class CyclopsNet(nn.Module):

    def __init__(self, parser):
        super(CyclopsNet, self).__init__()
        self.parser = parser
        self.grad_clip = parser.grad_clip
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.state_encoder = KAN([12, 256, 512, 768])
        self.img_encoder = ViTModel.from_pretrained('vit-base-patch16-224', add_pooling_layer=False)
        self.text_encoder = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        self.img_mu_var = LatentParams(parser.d_model, parser.latent_dim)
        self.txt_mu_var = LatentParams(parser.d_model, parser.latent_dim)
        self.img_state_fusion = Img_State_Fusion(parser.d_model)
        self.fusion = QueryGuidedFusion(parser)
        self.joint_encoder = KAN([parser.d_model + 2 * parser.latent_dim, parser.d_model + parser.latent_dim, parser.d_model])
        self.joint_mu_var = LatentParams(parser.d_model, parser.latent_dim)
        self.classifier = KAN([parser.latent_dim, int(parser.latent_dim / 2), 1])
       
        cudnn.deterministic = True     
        cudnn.benchmark = True
        
        # freeze parameters
        freeze_encoders = [self.text_encoder, self.img_encoder]
        for freeze_encoder in freeze_encoders:
            for param in freeze_encoder.parameters():
                param.requires_grad = False      
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
            
    def forward(self, images, states, queries):
        image_features = self.img_encoder(images).last_hidden_state
        states_embeddings = self.state_encoder(states)      # (2*bs, d_model)
        fused = self.img_state_fusion(image_features, states_embeddings)
        queries_inputs = self.roberta_tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to(self.parser.device)
        queries_features = self.text_encoder(**queries_inputs).last_hidden_state
        # alignment
        fuses_cls = fused[: , 0, :]                         # (2*bs, d_model)
        queries_cls = queries_features[: , 0, :]            # (2*bs, d_model)
        mu_image, logvar_image = self.img_mu_var(fuses_cls)  # (2*bs, latent_dim)
        mu_text, logvar_text = self.txt_mu_var(queries_cls)     # (2*bs, latent_dim)
        z_image = self.reparameterize(mu_image, logvar_image)   # (2*bs, latent_dim)
        z_text = self.reparameterize(mu_text, logvar_text)      # (2*bs, latent_dim)
        image_normalized = F.normalize(z_image, dim=1)              # (2*bs, latent_dim)
        text_normalized = F.normalize(z_text, dim=1)                # (2*bs, latent_dim)
        # fusion
        joint_input= self.fusion(fused, queries_features, image_normalized, text_normalized)  # (2*bs, d_model + 2 * latent_dim)
        joint_embeddings = self.joint_encoder(joint_input)              # (2*bs, d_model)
        mu_joint, logvar_joint = self.joint_mu_var(joint_embeddings)    # (2*bs, latent_dim)
        z_joint = self.reparameterize(mu_joint, logvar_joint)           # (2*bs, latent_dim)
        logits = self.classifier(z_joint).squeeze(1)
        q_image = Normal(mu_image, torch.exp(0.5 * logvar_image)) 
        q_text = Normal(mu_text, torch.exp(0.5 * logvar_text))    
        q_joint = Normal(mu_joint,torch.exp(0.5 * logvar_joint))
        return image_normalized, text_normalized, q_image, q_text, q_joint, logits