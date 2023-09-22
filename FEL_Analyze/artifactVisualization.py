from src.processDat import processDat
from pathlib import Path
BCI2kPath = Path("C:\BCI2000.x64\data")
dataPath = BCI2kPath / "NoahTesting000" /"NoahTestingS000R24.dat"
t = processDat(dataPath,notchOrder = 4, commonRef=False)
# t.plotStimOn()
# t.generateTriggerChannel()
t.build_MNE_Epochs()
# t.VisualizeRawOffset()
# t.plotEpochs()
# t.plotEpochsSubplot()
# t.whole_block_psds()
# t.epochFFT()
# t.plotStimulusCode()
# t.plotAllStates()
# t.plotPowerBands_oneChannel(10)
t.show()