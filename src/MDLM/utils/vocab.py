BOS_TOKEN = 0
PAD_TOKEN = 1
EOS_TOKEN = 2
MASK_TOKEN = 3 
CHAIN_BREAK_TOKEN = 4
UNK_TOKEN = 8

BOS_STR = "<cls>"
PAD_STR = "-"
EOS_STR = "<eos>"
UNK_STR = "<unk>"
MASK_STR = "<mask>"
CHAIN_BREAK_STR = "|"

SEQUENCE_VOCAB_SIZE = 25
VDJ_VOCAB_SIZE = 9
ANARCI_VOCAB_SIZE = 45

SEQUENCE_VOCAB = [
    "<cls>", "-", "<eos>", "<mask>", "|",
    "L", "A", "G", "V", "S", 
    "E", "R", "T", "I", "D", 
    "P", "K", "Q", "N", "F", 
    "Y", "M", "H", "W", "C",
]

VDJ_VOCAB = [
    "<cls>", "-", "<eos>", "<mask>", "|",
    "V", "D", "J", "<unk>"
]

ANARCI_VOCAB = [
    "<cls>", "-", "<eos>", "<mask>", "|",
    "fwk1", "fwk2", "fwk3", "fwk4",   
    "cdrk1", "cdrk2", "cdrk3", "cdrk4",
    "fwl1", "fwl2", "fwl3", "fwl4",   
    "cdrl1", "cdrl2", "cdrl3", "cdrl4",
    "fwh1", "fwh2", "fwh3", "fwh4",   
    "cdrh1", "cdrh2", "cdrh3", "cdrh4",
    "fwa1", "fwa2", "fwa3", "fwa4",   
    "cdra1", "cdra2", "cdra3", "cdra4",
    "fwb1", "fwb2", "fwb3", "fwb4",   
    "cdrb1", "cdrb2", "cdrb3", "cdrb4",
]

