import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


def getDrugGraph(smiles: str) -> Data:
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            raise ValueError("Invalid SMILES string")

        AllChem.ComputeGasteigerCharges(mol)  # Calculate necessary features

        nodes = pd.DataFrame(
            [
                [
                    a.GetAtomicNum(),  # Atomic number
                    a.GetDegree(),  # Degree (number of bonds)
                    int(a.GetHybridization()),  # Hybridization type
                    a.GetIsAromatic(),  # Aromaticity
                    a.GetFormalCharge(),  # Formal charge
                    a.GetMass(),  # Atomic mass
                    float(a.GetProp("_GasteigerCharge")),  # Gasteiger charge
                ]
                for a in mol.GetAtoms()
            ]
        )

        bonds = [
            (
                bond.GetBeginAtomIdx(),  # Index of the starting atom
                bond.GetEndAtomIdx(),  # Index of the ending atom
                int(bond.GetBondType()),  # Bond type
                int(bond.GetStereo()),  # Stereochemistry
            )
            for bond in mol.GetBonds()
        ]

        bonds = pd.DataFrame(bonds).values
        edges = bonds[:, :2]
        edges_attr = bonds[:, 2:]
        node_features = torch.Tensor(nodes.values.astype(float))
        atom_type = torch.tensor(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
        )
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edges_attr = torch.tensor(edges_attr, dtype=torch.float)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edges_attr,
            atom_type=atom_type,
        )

        return data

    except Exception as e:
        print(f"Error creating graph for SMILES {smiles}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a drug graph from a SMILES string"
    )
    parser.add_argument("smiles", type=str, help="SMILES string of the drug molecule")
    args = parser.parse_args()

    data = getDrugGraph(args.smiles)
    if data:
        print(data)
