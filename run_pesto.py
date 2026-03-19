import argparse
import os
import sys
import numpy as np
import torch as pt
from textwrap import dedent
from tqdm import tqdm
from glob import glob
from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from src.structure import data_to_structure, encode_bfactor, concatenate_chains, split_by_chain
from src.structure_io import save_pdb, read_pdb, save_text
from src.scoring import bc_scoring, bc_score_names
from model.config import config_model, config_data
from model.data_handler import Dataset
from model.model import Model

parser = argparse.ArgumentParser(
    description= '''Script usage: ''',
    epilog     = """test output files for 2CUA_A.pdb are found in examples/examples_in.
    Files ending in -i0.pdb or -iout0.pdb contain the PPI-ness probability [0,1] in the last but one column"""
    )
parser.add_argument('--ipath', default=None, help=' input pdb file list path')
parser.add_argument('--opath', default=None, help=' output pdb file list path')
parser.add_argument('--iosame', default=False, action='store_true', help=' create output pdb in the same path as input pdb (overrides --opath)')
parser.add_argument('--device', default='cpu', type=str, help=' device cpu or cuda (defaults to cpu)')
parser.add_argument('--allpreds', default=False, action='store_true', help=' return all predictions (PPI, lipid, etc), as opposed to the default of returning PPI only (_iout0.pdb) (defaults to False)')
parser.add_argument('--PPIasTEXT', default=False, action='store_true', help=dedent('''\
output the aminoacid sequence and PPI prediction of all input as a single .txt file called PPIpred.txt.
Configured only for --allpreds False.
Column names:
- file: basename of the corresponding pdb file
- chain: pdb chain ID
- AA: single-letter aminoacid code
- pos: index of the aminoacid in the protein
- PPI: probability [0,1] of being engaged in a PPI'''))

args = parser.parse_args()

# check arguments
def exit_program():
    print("No valid pdb files found, exiting...")
    sys.exit(0)

def exit_if_badpath(i, p):
    if not os.path.isdir(i):
        print(p)
        sys.exit(0)

if args.ipath is None:
    print('--ipath is required')
    sys.exit(0)
else:
    exit_if_badpath(args.ipath, 'path to --ipath does not exist, exiting...')

if args.opath is None:
    if not args.iosame:
        print('--opath is required unless --iosame is used')
        sys.exit(0)
    else:
         args.opath = args.ipath
else:
    exit_if_badpath(args.opath, 'path to --opath does not exist, exiting...')
    if args.iosame:
        print('--iosame will override --opath and output files written to --ipath')
        args.opath = args.ipath

# script body
ppitextpath = None
if args.PPIasTEXT:
    if args.iosame:
        ppitextpath = args.ipath
    else:
        ppitextpath = args.opath

def predict_and_save(i, z, structure, output_basepath, ppitextpath):
    # prediction
    p = pt.sigmoid(z[:,i])
    # encode result
    structure = encode_bfactor(structure, p.cpu().numpy())
    # save results
    output_filepath = output_basepath+'_iout{}.pdb'.format(i)
    save_pdb(split_by_chain(structure), output_filepath)
    # output text file
    if ppitextpath is not None:
       out_txt = save_text(split_by_chain(structure), output_filepath)
       return(out_txt)
    else:
       return(None)

# data parameters
input_path = args.ipath

# find pdb files and ignore already predicted pdbs (containing "iout" in the filename)
pdb_filepaths = glob(os.path.join(input_path, "*.pdb"), recursive=True)
pdb_filepaths = [fp for fp in pdb_filepaths if "_iout" not in fp]

# exit if no valid pdbs are found
if len(pdb_filepaths) == 0:
    exit_program()

# write output PPI as text file
if not args.allpreds and args.PPIasTEXT:
    ppitext = True
else:
    ppitext = False

# model parameters
model_path = "model/model_ckpt_i_v4_1_2021-09-07_11-21.pt"

# define device
device = pt.device(args.device)

# create model
model = Model(config_model)

# reload model
model.load_state_dict(pt.load(model_path, map_location=device))

# set model to inference
model = model.eval().to(device)

# create dataset loader with preprocessing
dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

# debug print
print(len(dataset))

# run model on all subunits

with pt.no_grad():
    outext = []
    for subunits, filepath in tqdm(dataset):
        # concatenate all chains together
        structure = concatenate_chains(subunits)
        # encode structure and features
        X, M = encode_structure(structure)
        #q = pt.cat(encode_features(structure), dim=1)
        q = encode_features(structure)[0]
        # extract topology
        ids_topk, _, _, _, _ = extract_topology(X, 64)
        # pack data and setup sink (IMPORTANT)
        X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])
        # run model
        z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))
        # for all predictions
        if args.iosame:
            output_basepath = filepath[:-4]
        else:
            output_basepath = os.path.join(args.opath, os.path.basename(filepath)[:-4])
        if not args.allpreds:
            out = predict_and_save(0, z, structure, output_basepath, ppitextpath)
        else:
            for i in range(z.shape[1]):
                out = predict_and_save(i, z, structure, output_basepath, ppitextpath)
        if out is not None:
            outext.extend(out)
            outexto = [[filepath + '\t' + row[0]] for row in outext]
    if len(outext) > 0:
        with open(args.opath+'/PPIpred.txt', 'w') as f:
            f.write('file\tchain\tAA\tpos\tPPI\n')
            for line in outexto: [f.write(s+'\n') for s in line]
