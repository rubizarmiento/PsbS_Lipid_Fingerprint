# Extract the PsbS monomer from structures
gmx select -s input.pdb -select "atomnr 1 to 668" -ofpdb output.pdb -pdbatoms selected

# Add bonds to PDB structure
gmx trjconv -f initial_protein.pdb -s protein.tpr -o initial_protein_bonds.pdb

# Add bonds to a Go-model 
python scripts/add_bonds_to_pdb_go_martini.py --itp_system protein.itp --itp_nb go_nbparams.itp --ipdb protein.pdb --opdb output.pdb

# Add bonds to a Go-model with multiple itps 
python scripts/add_bonds_to_pdb_go_martini.py --itp_system "protein1.itp protein2.itp" --itp_nb "go_nbparams1.itp go_nbparams2.itp" --ipdb protein.pdb --opdb output.pdb

