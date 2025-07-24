PAD_TOKEN = 0
MASK_TOKEN = 1 

PAD_STR = "-"
MASK_STR = "<mask>"

SEQUENCE_VOCAB_SIZE = 22
VDJ_VOCAB_SIZE = 5
ANARCI_VOCAB_SIZE = 42

SEQUENCE_VOCAB = [
    "-", "<mask>",
    "L", "A", "G", "V", "S", 
    "E", "R", "T", "I", "D", 
    "P", "K", "Q", "N", "F", 
    "Y", "M", "H", "W", "C",
]

VDJ_VOCAB = [
    "-", "<mask>",
    "V", "D", "J"
]

ANARCI_VOCAB = [
    "-", "<mask>",
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