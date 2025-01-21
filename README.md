# FEL: Fear Extinction Learning Paragidm Generator and Data Analysis Code for BCI2000
## FEL_paradigm:
- MATLAB code and accompanying files to initially generator parameter files to configure experiments in BCI2000 for data collection.
- Primary file is FEL_parms_generator.m, with permutations doing what they say in the filename.
  - Important to note that paths must be adjusted accordingly in generator files depending on filestructure of the machine that the experiment will be deployed on.
- Future development should compress parm generation to one script with multiple flags for different experiment features. 
## Analysis
- data analysis code for FEL_paradigm recordings
- Currently, existing python code is trivial
## Titration
  - Code that creates an executeable file (.exe) to determine hitrates for the amygdala stim titration paradigm.
      - implemented to check the subjective threshold of stimulation. 
