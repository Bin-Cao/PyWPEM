from EXAFS import EXAFS
EXAFS('absorb.csv',de_bac = True).fit(first_cutoff_energy=22100,second_cutoff_energy=22400)