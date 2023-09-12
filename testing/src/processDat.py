from BCI2kReader import BCI2kReader as bci
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from src.filters import highpass, notch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import neurokit2 as nk

class processDat():
    def __init__(self,path, commonRef:bool=True):
        with bci.BCI2kReader(path) as fp:
            self.sigs,self.states = fp.readall()
            self.states = self._reshapeStates()
            self.params = fp._parameters()
            self.StimuliID, self.faceValues = self._getStimuliIDs()
            self.StimuliLocs = self._getStimuliPresentationLocations()
            self.fs = self.params['SamplingRate']
            self.data = self._preprocess(self.sigs.copy(),commonRef)
            self.stimulusPresented = self.states['StimulusBegin']
            self.estim = self.states['EstimCurrent']
            target = max(self.estim)*0.7
            self.estimIdx = self._getBoolStateIDXs(self.estim, thresh=target)
            self.t = np.linspace(0,self.sigs.shape[-1]/self.fs,self.sigs.shape[-1])
            print('End Init')
    
    def _getStimuliIDs(self):
        idxs=[]
        faces = {}
        duration = {}
        stimuliVals = {}
        for i,value in enumerate(self.params['Stimuli'][5]):
            if value.find('face') > -1:
                idxs.append(i)
                img = self.params["Stimuli"][1][i].split("\\")[-1]
                faces[i+1] = f'{value} {img}'
                duration[i+1] = self.params['Stimuli'][3][i]
        for idx, face in faces.items():
            if face in stimuliVals:
                a = stimuliVals[face]
                a.append(idx)
                stimuliVals[face] = a
            else:
                stimuliVals[face] = [idx]
        return stimuliVals, faces

    def _getStimuliPresentationLocations(self):
        state = np.round(self.states['StimulusCode'],0)
        faceArray = {}
        for face, idx in self.StimuliID.items():
            temp = []
            for val in idx:
                temp.append(np.where(state == val))
            if len(temp[0][0])>0:
                faceArray[face] = temp
        return faceArray

    def _getBoolStateIDXs(self,state, thresh=0.7):
        peaks = sig.find_peaks(state, height=thresh, distance=int(self.fs*.1))
        return peaks[0]

    def _preprocess(self,data, commonRef:bool = True):
        out = np.empty(data.shape)
        # data = self._commonAverageReferencing(data)
        for i,channel in enumerate(data):
            out[i][:] = highpass(channel, self.fs, 0.2, 8)
            out[i][:] = notch(channel, self.fs, Wn=60, Q=15, order=8)
        if commonRef:
            out = self._commonAverageReferene(out)
        return out

    def _commonAverageReferene(self,data):
        commonAverage= np.average(data, axis=0)
        out = np.empty(data.shape)
        for i,channel in enumerate(data):
            out[i][:] = channel - commonAverage
        return out


    def _reshapeStates(self):
        states = {}
        for state, val in self.states.items():
            out = val[0]
            mode = st.mode(out).count
            if len(out) == mode:
                pass # Exclude states that do not contain any information, ie array contains all of one value. 
            else:
                states[state] = val[0]
        return states

    def visualizeRaw2Col(self):
        fig,ax = plt.subplots(int(self.sigs.shape[0]/2)+1,2)
        col = 0
        row = 0
        gs = ax[-1,0].get_gridspec()
        stims = self._getBoolStateIDXs(self.stimulusPresented)
        for a in ax[-1]:
            a.remove()
        for i,chan in enumerate(self.sigs):
            for stim in stims:
                ax[row][col].axvline(x=self.t[stim])    
            ax[row][col].plot(self.t,chan)
            ax[row][col].plot(self.t,self.data[i])
            row +=1
            if i == 7:
                row = 0
                col = 1
        axBig = fig.add_subplot(gs[-1,:])
        axBig.set_ylim(-.5, 1.5)
        self.plotStimOn(axBig)

    def VisualizeRawOffset(self):
        fig = plt.figure('Offset EEG Channels')
        ax = plt.subplot2grid((10,1),loc=(0,0),rowspan=9)
        for stim in self.estimIdx:
            ax.axvline(x=self.t[stim])        
        for i,chan in enumerate(self.data):
            ax.plot(self.t,chan+(30000*i))
        ax2 = plt.subplot2grid((10,1),loc=(9,0))
        self.plotStimOn(ax2)
        # self.plotEstim(ax)
        ax.get_shared_x_axes().join(ax,ax2)
        fig.add_axes(ax)
        fig.add_axes(ax2)
        
    def psds(self):
        fig = plt.figure('PSD')
        ax = plt.subplot2grid((1,1),loc=(0,0))       
        for i,chan in enumerate(self.data):
            f,Pxx = sig.welch(chan,fs = self.fs,window='hamming',nperseg=512) #hemming window
            ax.plot(f,Pxx)
        ax.set_xlim(0,1000)
            
    def plotStimOn(self,ax=None):
        if ax == None:
            fig, ax = plt.subplots(1,1,num='StimOn')
        ax.plot(self.t,self.stimulusPresented)
    def plotEstim(self,ax=None):
        if ax == None:
            fig, ax = plt.subplots(1,1,num='Estim')
        ax.plot(self.t,self.estim)

    def plotAllStates(self):
        rows = len(self.states)
        fig = plt.figure('states')
        for i,(state, v) in enumerate(self.states.items()):
            if i == 0:
                ax1 = plt.subplot2grid((rows,1),(0,0))
                ax1.plot(self.t,v, c=(0,0,0), alpha =0.3)
                ax1.set_ylabel(state, rotation=0)
                ax1.yaxis.tick_right()
                fig.add_axes(ax1)
            else:
                ax = plt.subplot2grid((rows,1),(i,0),sharex = ax1)
                ax.plot(self.t,v, c=(0,0,0), alpha =0.3)
                ax.set_ylabel(state, rotation=0)
                ax.yaxis.tick_right()
                fig.add_axes(ax)
            if state == 'StimulusCode':
                times, _ = self._getStimulusCodeTimeSeries()
                c = len(times)
                colors = sns.color_palette(None,c)
                count = 0
                for face, t in times.items():
                    value = self.StimuliID[face]
                    for vals in t:
                        y = np.ones(vals.shape)
                        y = value*y
                        ax.plot(vals,y, c=colors[count])
                    count +=1
    def plotStimulusCode(self):
        state = self.states['StimulusCode']
        fig, ax = plt.subplots(1,1,num='stimCode')
        ax.plot(self.t,state,c=(0,0,0),alpha = 0.5)
        times, _ = self._getStimulusCodeTimeSeries()
        c = len(times)
        colors = sns.color_palette(None,c)
        count = 0
        legend = []
        for face, t in times.items():
            legend.append(face)
            value = self.StimuliID[face]
            lab = face
            for vals in t:
                y = np.ones(vals.shape)
                y = value*y
                ax.plot(vals,y, c=colors[count],label=lab)
                lab = '_'
            count +=1
        ax.legend()
    def _getStimulusCodeTimeSeries(self):
        stateTime = {}
        stateIndexes = {}
        for face,vals in self.StimuliLocs.items():
            temp_time = []    
            temp_idx = []
            for i in vals:
                start = 0
                loc = i[0]
                t = False
                for idx,j in enumerate(loc[0:-1]):
                    if j+1 != loc[idx+1]:
                        t= True
                        temp_time.append(self.t[loc[start]:loc[idx]])
                        temp_idx.append([loc[start],loc[idx]])
                        start = idx+1
                if t == False:
                    temp_time.append(self.t[loc[0]:loc[-1]])
                    temp_idx.append([loc[0],loc[-1]])
            stateTime[face] = temp_time
            stateIndexes[face] = temp_idx
        return stateTime, stateIndexes
    

    def _epochDataByStimuli(self):
        _, stimIdx = self._getStimulusCodeTimeSeries()
        popKeys = []
        keys = list(stimIdx.keys())
        sliceLen = stimIdx[keys[0]][0][1] - stimIdx[keys[0]][0][0] 
        for key in stimIdx.keys():
            if key.find('w shock')>-1:
                popKeys.append(key)
        for key in popKeys:
            stimIdx.pop(key)
        data = {}
        for face, val in stimIdx.items():
            channels = {}
            for i, channel in enumerate(self.data):
                epochs = {}
                for j,idx in enumerate(val):
                    epochs[j] = channel[idx[0]:idx[0]+sliceLen]
                channels[i] = epochs            
            data[face] = channels
        return data  

    def plotEpochs(self):
        data = self._epochDataByStimuli()
        for face, signals in data.items():
            fig,ax = plt.subplots(1,1,num=face)
            for i,(channel,signal) in enumerate(signals.items()):
                offset = -20000*i
                epochs = []
                for epoch in signal.values():
                    ax.plot(epoch+offset,c=(0,0,0),alpha=0.3)
                    epochs.append(epoch)
                avg = np.average(epochs,axis=0)
                ax.plot(avg+offset, c = (1,0,0))




    def show(self):
        plt.show()


