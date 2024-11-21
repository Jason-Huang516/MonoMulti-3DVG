import os
import torch
from tqdm import tqdm
from model import CyclopsNet
import sys
import arguments
from torch.nn.utils.clip_grad import clip_grad_norm_
from loss_factory import CyclopsNetLoss  
from utils.datasetprocessor import *
from utils.utils import lr_lambda
import functools
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = arguments.get_argument_parser().parse_args()
    num_epochs = parser.num_epochs
    train_loader = get_loader(parser.data_path, parser.dataset, "train", parser.batch_size, True, 8)
    print("****************Load data completed!!******************")
    model = CyclopsNet(parser).to(parser.device)
    print("****************Load model completed!!******************")
    criterion = CyclopsNetLoss(parser).to(parser.device)
    params_to_update = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(
                                params_to_update,
                                lr=parser.learning_rate,
                                weight_decay=parser.decay_factor
                                )
    lr_lambda_func = functools.partial(lr_lambda, warmup_epochs=parser.warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_func)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=parser.learning_rate * 0.2, verbose=True)
    writer = SummaryWriter(log_dir='logs')
    minimum_loss = float('inf')
    start_epoch = 0
    
    if parser.resume:
        if os.path.isfile(parser.resume):
            print("=> loading checkpoint '{}'".format(parser.resume))
            checkpoint = torch.load(parser.resume, map_location=parser.device)
            start_epoch = checkpoint['epoch']
            epoch_loss = checkpoint['epoch_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}, epoch_loss {})"
                    .format(parser.resume, start_epoch, epoch_loss))
        else:
            print("=> no checkpoint found at '{}'".format(parser.resume))

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        model.train()
        train_bar = tqdm(train_loader, file=sys.stderr)
        for _, (images, states, queries, labels) in enumerate(train_bar):
            optimizer.zero_grad()  

            image_embeddings, text_embeddings, q_image, q_text, q_joint, logits = model(images.to(parser.device), states.to(parser.device), queries)
            loss = criterion(image_embeddings, text_embeddings, q_image, q_text, q_joint, logits, labels.to(parser.device))

            loss.backward()
            if model.grad_clip > 0:
                clip_grad_norm_(params_to_update, model.grad_clip)
            optimizer.step()
                
            running_loss += loss.item()

            train_bar.desc = (
                "train_epoch[{}/{}] ITC_Loss:{:.4f} KL_Loss:{:.4f} "
                "ITM_Loss:{:.4f} Loss_total:{:.4f}"
                .format(
                    epoch + 1,
                    num_epochs,
                    criterion.itc_value.item(),
                    criterion.kl_value.item(),
                    criterion.itm_value.item(),
                    loss.item()
                )
            )

        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Learning Rate: {current_lr}")
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        if epoch < parser.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            save_dir = r"./training/model"
            os.makedirs(save_dir, exist_ok=True)  
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch_loss': epoch_loss,
                'parser': parser,
            }, os.path.join(save_dir, f'model_{epoch + 1}epoch_checkpoint.pth.tar'))
            
        if epoch_loss < minimum_loss :
            minimum_loss = epoch_loss
            save_dir = r"./training/model"
            os.makedirs(save_dir, exist_ok=True)  
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch_loss': epoch_loss,
                'parser': parser,
            }, os.path.join(save_dir, f'model_best.pth.tar'))
    
    writer.close()
    print("Training completed.")

if __name__ == '__main__':
    main()