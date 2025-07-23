from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import omegaconf
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from kasearch import SearchDB, AlignSequences
from anarci import number

from MDLM.diffusion import Diffusion
from MDLM.data import OASDataModule
from MDLM.utils.tokenizer import ProteinTokenizer
from MDLM.trainer import DiffusionTrainer


def main():    
    parser = argparse.ArgumentParser(description='Train MDLM Antibody Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Data loader
    oas_data = OASDataModule()
    data_file = Path('data/processed/poas_human_all_aho.pkl')
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found. Run preprocess.py first.")
    oas_data.load_saved(data_file)
    ab = oas_data.to_antibody(save=False)     
    print(f"Loaded OAS data with {len(ab)} proteins")
    train_data, test_data, val_data = random_split(ab, lengths=[0.8, 0.1, 0.1]) 
    
    config = omegaconf.OmegaConf.load(args.config)
    assert config.model.hidden_size % config.model.n_heads == 0 # d_model must be divisible by n_heads
    
    max_len = config.model.max_len
    tokenizer = ProteinTokenizer()    
    train_loader = DataLoader(
        tokenizer.batch_tokenize_pad(data=train_data, max_len=max_len),
        shuffle=True,
        batch_size=config.training.batch_size, 
        num_workers=config.training.num_workers)
    test_loader = DataLoader(
        tokenizer.batch_tokenize_pad(data=test_data, max_len=max_len), 
        shuffle=True,
        batch_size=config.training.batch_size, 
        num_workers=config.training.num_workers)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    diffusion_model = Diffusion(config, tokenizer, device).to(device)
    trainer = DiffusionTrainer(diffusion_model, config, tokenizer, max_steps=config.training.epochs * len(train_loader))

    searchdb = SearchDB(
        database_path='data/oasdb_small_20230111',
        allowed_species='Human',
        regions=['cdrs', 'cdr3'],
        length_matched=[False, False])    
    
    if config.mode == 'train':
        wandb.init(
            entity="fyng", 
            project="mdlm_antibody", 
            config=omegaconf.OmegaConf.to_container(config, resolve=True))
    
    # dispatch training
    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch+1}/{config.training.epochs}")
        train_loss = trainer.fit(train_loader)
        test_loss = trainer.validate(test_loader)

        # CALCULATE METRICS
        seq_list = []
        # sample 100 sequences
        for _ in tqdm(range(100)):
            seq = diffusion_model.restore_model_and_sample(
                num_steps = 1000,
                store_traj = False
            )
            seq_list.append(seq)
        seq_tokens = torch.cat(seq_list)
        vdj_tokens = None
        anarci_tokens = None
        proteins = tokenizer.batch_detokenize(seq_tokens, vdj_tokens, anarci_tokens)

        success = 0
        sample_dict=[]
        with open(f'output/sample_epoch{epoch}.txt', 'w') as f:
            for i, p in enumerate(proteins):
                f.write(f"{p.sequence}'\n'")
                f.write(f"{p.vdj_label}'\n'")
                f.write(f"{p.anarci_label}'\n'")
                # strict version of parsing function:
                try:
                    vl, vh = p.to_aa(format='aho')
                    # align with anarci
                    _, chain1 = number(vl)
                    _, chain2 = number(vh)
                    if not chain1 and chain2:         
                        continue   
                    try:                        
                        # Sequence similarity search against known human antibodies
                        # using KA search. https://github.com/oxpig/kasearch
                        aligned = AlignSequences(allowed_species=['Human'])([vl, vh])
                        searchdb.search(aligned)
                        res = searchdb.current_best_identities # this automatically resets at the next search()
                        res = np.max(res, axis=1) # take best metric found
                        
                        sample_dict.append({
                            "VL": vl,
                            "VH": vh,
                            "VL_similarity_cdr": res[0][0],
                            "VL_similarity_cdr3": res[0][1],
                            "VH_similarity_cdr": res[1][0],
                            "VH_similarity_cdr3": res[1][1],
                            # "file_path": pdb_fp,
                        })
                        
                        # write to yaml file for boltz
                        yaml_data = {
                            "version": 1,
                            "sequences": [
                                {
                                    "protein": {
                                        "id": "VL",
                                        "sequence": vl
                                    }
                                },
                                {
                                    "protein": {
                                        "id": "VH", 
                                        "sequence": vh
                                    }
                                }
                            ]
                        }
                        
                        yaml_filename = f'output/yaml/epoch{epoch}_sample{i}.yaml'
                        with open(yaml_filename, 'w') as yaml_file:
                            yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)
                    except KeyError as e:
                        print(f"Missing sequence from chain {str(e)}")
                except:
                    continue    
        df = pd.DataFrame(sample_dict)
        df.to_csv(f'output/sample_epoch{epoch}.csv', index=False)
    
        # log validation metrics
        wandb.log({
            'test_loss': test_loss,
            'epoch': epoch,
        })
        
        # save model weights
        diffusion_model.save(f"model_weights/diffusion_epoch_{epoch}.pt")
    
    
if __name__ == "__main__":
    main()