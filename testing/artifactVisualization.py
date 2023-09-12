from src.processDat import processDat
from pathlib import Path
BCI2kPath = Path("C:\BCI2000.x64\data")
dataPath = BCI2kPath / "NoahTesting000" /"NoahTestingS000R04.dat"
t = processDat(dataPath)
# t.plotStimOn()
# t.visualizeRaw2Col()
t.VisualizeRawOffset()
t.psds()
t.plotAllStates()
t.show()