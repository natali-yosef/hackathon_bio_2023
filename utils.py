from Bio.PDB import *
import numpy as np
import os
from tqdm import tqdm
import pathlib
import torch
from Bio.PDB import PDBParser, Polypeptide, is_aa
from esm import FastaBatchedDataset, pretrained

NB_MAX_LENGTH = 140
AA_DICT = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
           "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20, "-": 21}
FEATURE_NUM = len(AA_DICT)
BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]
OUTPUT_SIZE = len(BACKBONE_ATOMS) * 3
NB_CHAIN_ID = "H"
MODELS_LIST = ['esm2_t36_3B_UR50D', 'esm2_t6_8M_UR50D', 'esm1b_t33_650M_UR50S']
              # 'esm1b_t33_650M_UR50S_with_contact','esm2_t6_8M_UR50D_with_contact']
LAYERS_NUMBER = {'esm2_t36_3B_UR50D': 36, 'esm2_t6_8M_UR50D': 6, 'esm1b_t33_650M_UR50S': 33}
EMBENDING_DIM = {'esm2_t36_3B_UR50D': 2560, 'esm2_t6_8M_UR50D': 320, 'esm1b_t33_650M_UR50S': 1280}

import os
import sys
os.path.abspath(sys.argv[0])

def path_to_save_emmbending(model_name):
  return f"./input_data/{model_name}/"

def model_saved_path(model_name):
  # return f"/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/models/saved_model_{model_name}.h5"
  return f"./models/saved_model_{model_name}.h5"


def input_path(model_name):
  # return f"/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Save/train_input_{model_name}.npy"
    print("input_path")
    return f"/input_data/{model_name}/train_input.npy"


def output_path(model_name):
  # return f"/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Save/train_labels_{model_name}.npy"
    return f"/input_data/{model_name}/train_labels.npy"


def output_dir_predict():
  return f"./pdb_predictions/"


def get_seq_aa(pdb_file, chain_id):
    """
    returns the sequence (String) and a list of all the aa residue objects of the given protein chain.
    :param pdb_file: path to a pdb file
    :param chain_id: chain letter (char)
    :return: sequence, [aa objects]
    """
    # load model
    chain = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'): # Not amino acid
            continue
        elif aa == "UNK":  # unkown amino acid
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(residue.get_resname())
        aa_residues.append(residue)

    return seq, aa_residues


def generate_label(pdb_file):
    """
    receives a pdb file and returns its backbone + CB coordinates.
    :param pdb_file: path to a pdb file (nanobody, heavy chain has id 'H') already alingned to a reference nanobody.
    :return: numpy array of shape (CDR_MAX_LENGTH, OUTPUT_SIZE).
    """
    # get seq and aa residues
    seq, aa_residues = get_seq_aa(pdb_file, NB_CHAIN_ID)

    # make sure the length does not exceed the nb_max_length
    sequence_length = min(NB_MAX_LENGTH, len(seq))

    # initialize the label matrix
    label_matrix = np.zeros((NB_MAX_LENGTH, OUTPUT_SIZE))

    # loop through the residues of the sequence.
    for i in range(sequence_length):
        residue = aa_residues[i]

        # For each residue, procure the coordinates of the CB  and the backbone
        backbone_coordinates = []
        for atom_name in BACKBONE_ATOMS:
            # If it is the Glycine amino acid, Set its CB coordinates to (0, 0, 0) as instructed
            if atom_name == "CB" and residue.get_resname() == "GLY":
                backbone_coordinates.extend([0.0, 0.0, 0.0])
            else:
                coordinates = residue[atom_name].get_coord()
                backbone_coordinates.extend(coordinates)

        # Put the coordinates inside the label matrix
        label_matrix[i] = backbone_coordinates

    # If the sequence is shorter than 140 aa, pad it with zero rows.
    if sequence_length < NB_MAX_LENGTH:
        label_matrix[sequence_length:NB_MAX_LENGTH, :] = 0.0

    return label_matrix

def matrix_to_pdb(seq, coord_matrix, pdb_name):
    """
    Receives a sequence (String) and the output matrix of the neural network (coord_matrix, numpy array)
    and creates from them a PDB file named pdb_name.pdb.
    :param seq: protein sequence (String), with no padding
    :param coord_matrix: output np array of the nanobody neural network, shape = (NB_MAX_LENGTH, OUTPUT_SIZE)
    :param pdb_name: name of the output PDB file (String)
    """
    ATOM_LINE = "ATOM{}{}  {}{}{} {}{}{}{}{:.3f}{}{:.3f}{}{:.3f}  1.00{}{:.2f}           {}\n"
    END_LINE = "END\n"
    k = 1
    save_path_pdb = os.path.join(output_dir_predict(), f"{pdb_name}.pdb")
    with open(save_path_pdb, "w") as pdb_file:
        for i, aa in enumerate(seq):
            third_space = (4 - len(str(i))) * " "
            for j, atom in enumerate(BACKBONE_ATOMS):
                if not (aa == "G" and atom == "CB"):  # GLY lacks CB atom
                    x, y, z = coord_matrix[i][3*j], coord_matrix[i][3*j+1], coord_matrix[i][3*j+2]
                    b_factor = 0.00
                    first_space = (7 - len(str(k))) * " "
                    second_space = (4 - len(atom)) * " "
                    forth_space = (12 - len("{:.3f}".format(x))) * " "
                    fifth_space = (8 - len("{:.3f}".format(y))) * " "
                    sixth_space = (8 - len("{:.3f}".format(z))) * " "
                    seventh_space = (6 - len("{:.2f}".format(b_factor))) * " "

                    pdb_file.write(ATOM_LINE.format(first_space, k, atom, second_space, Polypeptide.one_to_three(aa) , "H", third_space,
                                                    i, forth_space, x, fifth_space, y, sixth_space, z, seventh_space,
                                                    b_factor, atom[0]))
                    k += 1

        pdb_file.write(END_LINE)
    return save_path_pdb 

# def extract_embeddings(model_name, pdb_dir, output_dir, repr_layers=[36]):
#     model, alphabet = pretrained.load_model_and_alphabet(model_name)
#     print("done download")
#     model.eval()

#     if torch.cuda.is_available():
#         model = model.cuda()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     pdb_dir = pathlib.Path(pdb_dir)
#     pdb_files = list(pdb_dir.glob('*.pdb'))

#     with torch.no_grad():
#         for pdb_file in pdb_files:
#             #print(type(pdb_file))
#             seq, _ = get_seq_aa(pdb_file, CHAIN_ID)

#             batch_converter = alphabet.get_batch_converter()
#             batch_labels, batch_strs, batch_tokens = batch_converter([(str(pdb_file.stem), seq)])

#             if torch.cuda.is_available():
#                 batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

#             out = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)

#             representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

#             for i, label in enumerate(batch_labels):
#                 entry_id = label.split()[0]

#                 filename = output_dir / f"{entry_id}.pt"
#                 result = {"entry_id": entry_id}

#                 # save amino acid embeddings instead of mean representation
#                 result["amino_acid_embeddings"] = {layer: t[i, 1:-1].clone() for layer, t in representations.items()}
#                 torch.save(result, filename)

# def generate_input(pt_file):
#   data = torch.load(pt_file)['amino_acid_embeddings'][36]
#   padded_data= torch.zeros((140, 2560))
#   padded_data[:data.size(0),:] = data
#   return padded_data.numpy()

if __name__ == '__main__':

   #  you can make all the data for the network in this section.
   # you can save the matrices to your drive and load them in your google colab file later.


    input_matrix = []
    labels_matrix = []
    # data_path = "C:\StudiesYes\Computer Science\\76562\Week4\Ex4\Ex4Data"
    data_path = "/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Ex4Data"

    for pdb in tqdm(os.listdir(data_path)):

        nb_one_hot = generate_input(os.path.join(data_path, pdb), feature_num)
        nb_xyz = generate_label(os.path.join(data_path, pdb))

        input_matrix.append(nb_one_hot)
        labels_matrix.append(nb_xyz)
    # save_path =  "C:\StudiesYes\Computer Science\\76562\Week4\Ex4\Save"
    save_path = "/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Save"
    np.save(f"{save_path}/train_input.npy", np.array(input_matrix))
    np.save(f"{save_path}/train_labels.npy", np.array(labels_matrix))

    print(f"Number of samples: {len(input_matrix)}")


