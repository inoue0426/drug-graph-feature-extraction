# Drug Graph Feature Extraction

This project extracts features from drug molecules represented by SMILES strings and constructs a graph representation suitable for deep learning models.

## Requirements

- Python 3.7+
- RDKit
- Torch
- Torch Geometric

## Usage

To create a graph from a SMILES string, run the following command:

```bash
python src/drug_graph.py "CCO"
```

or

```python
import drug_graph

drug_graph.getDrugGraph("CCO")
```