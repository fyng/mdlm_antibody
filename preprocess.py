# script to preprocess OAS data
from pathlib import Path
from MDLM.data import OASDataModule

def main():
    oas_data = OASDataModule()
    # data_dir = "data/test"
    data_dir = "data/oas_pair_human"
    oas_data.load_data_folder(data_dir)
    df = oas_data.to_dataframe()
    
    # Aho alignment for pOAS human data (1.9M rows) takes around 21 hours 
    # not for the faint-hearted
    ab = oas_data.to_antibody(save=True, alignment_method='aho')
    
    print(ab[0].sequence)
    print(df.shape)
    print(f"{df.shape[0] - len(ab)} sequences not aligned")
    
if __name__ == '__main__':
    main()