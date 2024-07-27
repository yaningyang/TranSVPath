# TranSVPath

TranSVPath is a deep learning model based on the Transformer architecture, specifically designed for accurately predicting the pathogenicity of structural variants (SVs).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

TranSVPath leverages the power of Transformer models to predict the pathogenicity of various types of structural variants, including deletions, duplications, insertions, inversions, and microsatellites. Our comprehensive evaluations demonstrate that TranSVPath outperforms existing tools, showcasing its robustness and superior predictive accuracy in genomic variant analysis.

## Features

- Accurate prediction of SV pathogenicity.
- Handles multiple types of structural variants.
- High generalization capability, even with rare variant types.
- Utilizes the TabTransformer deep learning framework.
  
## Installation

To install TranSVPath, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yaningyang/TranSVPath.git
```
The following Python packages and their versions are required for this project:

| Package              | Version |
|----------------------|---------|
| `tensorflow`         | 2.10.0  |
| `tensorflow-addons`  | 0.18.0  |
| `PyVCF`              | 0.6.8   |
| `pysam`              | 0.19.1  |


## Usage

1. Prepare your input data.
    1. Get the histone signals of DNase H2A.Z H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me2 H3K9ac H3K9me3 H4K20me1 of E003, E006, and E008 from https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidatedImputed and put them in the data/bw_files directory.
    2. Get some feature data (excluding histone data) from the PhenoSV project (Xu Z, Li Q, Marchionni L, et al. PhenoSV: interpretable phenotype-aware model for the prioritization of genes affected by structural variants[J]. Nature Communications, 2023, 14(1): 7805.) and put them in the data/bw_files directory.
    3. Get the pre-trained model from https://1drv.ms/f/c/49fb0e2344c180c4/EmyGW93ZpGNJuJrYIXm63w8BdhM6YgoHAfQsIrOG_xDTIQ

2. Run the TranSVPath model:
    ```bash
    cd src
    python transvpath.py -f path/to/your.vcf -m path/to/saved_models/  -o ./result.txt
    ```

3. When preparing your VCF files, please ensure the following:

    - **INFO Field**: The INFO field in the VCF file must clearly specify the type of variant (`SVTYPE`). The acceptable variant types are:
      - Deletion
      - Insertion
      - Inversion
      - Duplication
      - Microsatellite
    
    - **END Field**: The end position (`END`) of the variant must be provided in the INFO field.
    
    - **Molecular Consequence (MC)**: Molecular consequence information should be included and can be annotated using Annovar.
    
    Make sure your VCF files adhere to these specifications to ensure compatibility with the analysis tools used in this project.
   
4. The output results are saved in a txt text file, and the field values ​​are:
   ```
   chrom   start   end   variant_type    pathogenicity_score
   ```

## Contributing
  Contributions are welcome! 

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
We acknowledge the contributions of all team members and collaborators who made this project possible. 
