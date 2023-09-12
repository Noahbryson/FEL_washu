from BCI2kReader import BCI2kReader as bci
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from lib.filters import highpass, notch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import neurokit2 as nk

class processDat():
    def __init__(self,path):
        with bci.BCI2kReader(path) as fp:
            self.sigs,self.states = fp.readall()
            self.states = self._reshapeStates()
            self.params = fp._parameters()
            self.fs = self.params['SamplingRate']
            self.data = self._preprocess(self.sigs.copy())
            self.stimulusPresented = self.states['StimulusBegin']
            self.estim = self.states['EstimCurrent']
            target = max(self.estim)*0.7
            self.estimIdx = self._getBoolStateIDXs(self.estim, thresh=target)
            self.t = np.linspace(0,self.sigs.shape[-1]/self.fs,self.sigs.shape[-1])
            print('End Init')
    
    def _getBoolStateIDXs(self,state, thresh=0.7):
        peaks = sig.find_peaks(state, height=thresh, distance=int(self.fs*.1))
        return peaks[0]

    def _preprocess(self,data):
        out = np.empty(data.shape)
        for i,channel in enumerate(data):
            out[i][:] = highpass(channel, self.fs, 0.2, 8)
            out[i][:] = notch(channel, self.fs, Wn=60, Q=15, order=8)
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
        stims = self._getBoolStateIDXs(self.stimulusPresented)
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
            f,Pxx = sig.welch(chan,fs = self.fs,nperseg=512,scaling='density')
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
                ax1.plot(self.t,v)
                ax1.set_ylabel(state, rotation=0)
                ax1.yaxis.tick_right()
                fig.add_axes(ax1)
            else:
                ax = plt.subplot2grid((rows,1),(i,0),sharex = ax1)
                ax.plot(self.t,v)
                ax.set_ylabel(state, rotation=0)
                ax.yaxis.tick_right()
                fig.add_axes(ax)
            
        
    def show(self):
        plt.show()


