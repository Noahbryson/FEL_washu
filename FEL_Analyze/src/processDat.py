from BCI2kReader import BCI2kReader as bci
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from src.filters import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import neurokit2 as nk
import mne
import csv
import os


class biosignals():
    def __init__(self,all_signals,all_labels,fs,notchOrder,commonRef, otherSignals:list =['EMG','GSR','EDA']):
        self.CoreChanTypes = {'EEG':('AF3', 'AF4', 'C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T7', 'T8'), 'EMG':['EMG'], 'GSR':['GSR']}
        all_signals = all_signals
        all_labels = all_labels
        self.fs = fs
        notchOrder = notchOrder
        otherSignals = otherSignals
        if all_labels:
            self.chanLabelFlag = True
        else:
            self.chanLabelFlag = False
        self.EEG, self.response_signals = self._isolate_signals(all_signals, all_labels,otherSignals)
        if commonRef:
            self.EEG = self._commonAverageReference(self.EEG)
        self.EEG, self.response_signals = self.preprocessAll(self.EEG,self.response_signals,notchOrder)
        self.allData = self.EEG | self.response_signals
        self.allLabels = tuple(self.allData.keys())
        self.EEGLabels = tuple(self.EEG.keys())
        print(1)
    
    def getChannelTypes(self):
        output = {}
        for channel in self.allData:
            for key,vals in self.CoreChanTypes.items():
                if channel in vals:
                    output[channel] = key
                    break
                else:
                    output[channel] = 'sEEG'
        x,y =  list(output.keys()), [i.lower() for i in output.values()]
        return x,y
    def preprocessAll(self,EEG,Response_sigs,notchOrder=4):
        EEGDat, eegKey = dict_2_numpy(EEG)
        eeg_keyOrder = sorted(eegKey)
        EEGDat = self._preprocessEEG(EEGDat,notchOrder)
        EEG = numpy_2_dict(EEGDat,eegKey)
        EEGout = {k:EEG[k] for k in eeg_keyOrder}
        resp_data,resp_key = dict_2_numpy(Response_sigs)
        respOut = {}
        for i,key in enumerate(resp_key):
            if key.find('EMG') >-1:
                respOut[key] = self._preprocessEMG(resp_data[i],notchOrder) 
            if key.find('GSR') >-1:
                respOut[key] = self._preprocessEDA(resp_data[i])
        return EEGout,respOut
    def _preprocessEEG(self,data,notchOrder=4, commonRef:bool = True):
        out = np.empty(data.shape)
        for i,channel in enumerate(data):
            out[i][:] = highpass(channel, self.fs, 0.5, 8)
            out[i][:] = lowpass(out[i][:], self.fs, 56, 8)
            if notchOrder:
                out[i][:] = notch(channel, self.fs, Wn=60, Q=30, order=notchOrder)
        return out
    def _preprocessEMG(self,data,notchOrder=4, commonRef:bool = True):
        out = highpass(data, self.fs, 12, 8)
        out = lowpass(out, self.fs, 500, 8)
        if notchOrder:
            out = notch(out, self.fs, Wn=60, Q=60, order=notchOrder)
        out = out*10**-6 # convert to V
        return out
    def _preprocessEDA(self,data):
        out = highpass(data, self.fs, 0.001, 8)
        out = lowpass(out, self.fs, 10, 8)
        # out = out*10**3 # convert to mOhm
        return out
    def _isolate_signals(self,allData,labels, conds:list =['EMG','GSR','EDA'])->(dict,dict):
        popList = []
        EEG_out = {}
        otherOut = {}
        if self.chanLabelFlag:
            for i,lab in enumerate(labels):
                for cond in conds:
                    if lab.find(cond) >=0:
                        popList.append(labels.index(lab))
            EEG = allData[0:-len(popList)]
            for lab, chan in zip(labels[0:-len(popList)],EEG):
                EEG_out[lab] = chan
            other = allData[-len(popList):]
            for lab, chan in zip(labels[-len(popList):],other):
                otherOut[lab] = chan
        else:
            EEG = allData
            for lab,chan in zip(labels,EEG):
                EEG_out[lab] = chan
            other = None
        return EEG_out, otherOut   
    def _commonAverageReference(self,data):
        EEG,keys = dict_2_numpy(data)
        commonAverage = np.average(EEG, axis=0)
        for i,channel in enumerate(data.values()):
            EEG[i][:] = channel - commonAverage
        data =numpy_2_dict(EEG,keys)
        return data
class eyetracking():
    def __init__(self):
        self.exists = False
    def setData(self,data,fs):
        states = {}
        if 'EyetrackerTime' in data.keys():
            self.exists = True
            self.fs = fs
            for key,val in data.items():
                if key.find('Eyetracker') >-1:
                    states[key] = val
            self.t = np.linspace(0,len(val)/fs,len(val))
        else:
            print('No Eyetracker Data')
        self.formatData(states)
    def formatData(self,states):
        self.attrs = ('Validity','Pos','Size','Dist','Gaze')
        sides = ('Left', 'Right')
        # self.Pos, self.Validity, self.Size, self.Dist, self.Gaze = {},{},{},{},{}
        for i in self.attrs:
            temp = {}
            for key,val in states.items():
                if key.find(i) > -1:
                    key = key.replace('Eyetracker','')
                    temp[key] = val
            setattr(self,i,temp)
    
    def plotAttr(self,attr, ax:plt.axes=False, smooth:bool=True, dec:str= ''):
        if type(ax) == bool:
            fig, ax = plt.subplots(1,1,num=f'{attr} {dec}')
        data = getattr(self,attr)
        for key,val in data.items():
            if key.find('Size')>-1 and smooth == True:
                smooth = savitzky_golay(val,1001,1,0)
                smooth = windowed_EMavg(val,windowLen=50,smoothing=0.5)
                print(val == smooth)
                ax.plot(self.t,smooth,label=f'{key} sm')
                # val = highpass(val,self.fs,0.2,2)
                # ax.plot(self.t,savitzky_golay(smooth,101,5,1),label=f'{key} hp')
            else:
                ax.plot(self.t,val,label=key)
        ax.legend()
    def plotAllAttr(self, subplot:bool=True):
        if subplot:
            fig, ax = plt.subplots(len(self.attrs),1, num='Eyetracking Data',sharex=True)
            for i,attr in enumerate(self.attrs):
                self.plotAttr(attr,ax=ax[i])
        else:
            for attr in self.attrs:
                self.plotAttr(attr)
        
    

def dict_2_numpy(data):
    keys = tuple(data.keys())
    out = np.empty([len(data),len(list(data.values())[0])])
    for i,value in enumerate(data.values()):
            out[i][:] = value
    return out,keys
def numpy_2_dict(data,keys):
    out = {}
    for i, key in enumerate(keys):
        out[key] = data[i][:]
    return out

class processDat():
    import pathlib
    def __init__(self,path, notchOrder: int = 4, commonRef:bool=True, otherSignals:list =['EMG','GSR','EDA']):
        # plt.style.use('dark_background')
        self.eyetracking = eyetracking()
        with bci.BCI2kReader(path) as fp:
            self.params = self._getParams(fp,path)
            self.sigs,self.states = fp.readall()
            self.states = self._reshapeStates()
            self.fs=int(fp._samplingrate())
        self.eyetracking.setData(self.states,self.fs)
        self.chanLabelFlag = len(self.params['ChannelNames'])
        if self.chanLabelFlag:
            self.ChanLabs = self.params['ChannelNames']
        else:
            self.ChanLabs = [i+1 for i in range(len(self.sigs))]            
        self.StimuliID, self.faceValues = self._getStimuliIDs()
        self.StimuliLocs = self._getStimuliPresentationLocations()
        # self.fs = self.params['SamplingRate']
        self.dataStruct = biosignals(self.sigs,self.ChanLabs,self.fs,notchOrder,commonRef,otherSignals)
        self.ChanLabs = self.dataStruct.allLabels
        self.stimulusPresented = self.states['StimulusBegin']
        self.t = np.linspace(0,self.sigs.shape[-1]/self.fs,self.sigs.shape[-1])
        self.offsetVal = 3*np.std(self.sigs[0]) #uV
        
        if 'EstimCurrent' in self.states:
            self.estim = self.states['EstimStimulus']
            target = max(self.estim)*0.7
            self.estimIdx = self._getBoolStateIDXs(self.estim, thresh=target)
            self.u_s = True
        else: 
            self.estim = np.zeros(self.t.shape)
            self.estimIdx = []
            self.u_s = False
        print('End Init')
    """Processing Functions"""
    def _getParams(self, fp:bci.BCI2kReader,path:pathlib.Path):
        params = fp._parameters()
        if not 'Stimuli' in params:
            stimuli = []
            fname = str(path.name)
            name = fname.split('.')[0]
            stimFile = path.parent / f'{name}.csv'
            if os.path.isfile(stimFile):
                with open(stimFile,'r') as file:
                    for line in csv.reader(file):
                        stimuli.append(line)
                stimuli.pop(0)

            else:
                print('Please Run Matlab Function to Generate Stimuli File')
                exit()
        params['Stimuli'] = stimuli
        return params

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
                stimuliVals[face] = idx
        return stimuliVals, faces
    
    def _getStimuliPresentationLocations(self):
        state = np.round(self.states['StimulusCode'],0)
        faceArray = {}
        for face, idx in self.StimuliID.items():
            temp = []
            if type(idx)==list:
                for val in idx:
                    temp.append(np.where(state == val))
            else:
                temp.append(np.where(state == idx))
            if len(temp[0][0])>0:
                faceArray[face] = temp
        return faceArray
    
    def _getBoolStateIDXs(self,state, thresh=0.7):
        peaks = sig.find_peaks(state, height=thresh, distance=int(self.fs*.3))
        return peaks[0]
    
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
        rawdata,rawKeys = dict_2_numpy(self.dataStruct.allData)
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
            for i, channel in enumerate(rawdata):
                epochs = {}
                for j,idx in enumerate(val):
                    epochs[j] = channel[idx[0]:idx[0]+sliceLen]
                channels[i] = epochs            
            data[face] = channels
        return data, rawKeys
    def generateTriggerChannel(self):
        trig = np.zeros(len(self.sigs[0]))
        times, x = self._getStimulusCodeTimeSeries()
        for face, t in x.items():
            for r in t:
                trig[r[0]:r[1]+1] = self.StimuliID[face]
        trigVals = set(trig)
        ids = {}
        values = list(self.StimuliID.values())
        keys = list(self.StimuliID.keys())
        for val in trigVals:
            if val in values and keys[values.index(val)].find('w shock')<0:
                ids[keys[values.index(val)]] = int(val)
        return trig , ids
    def build_MNE_Epochs(self, plotRaw:bool=False,plotEpochs:bool=False):
        montage = mne.channels.make_standard_montage('easycap-M1')
        ch_names, ch_types = self.dataStruct.getChannelTypes()
        info = mne.create_info(ch_names, self.fs,ch_types)
        raw, rawChan = dict_2_numpy(self.dataStruct.allData)
        for i, data in enumerate(raw[0:len(self.dataStruct.EEG)+1]):
            raw[i][:] = 10**-6*data
        data = mne.io.RawArray(raw,info)
        data.set_montage(montage)
        data.set_eeg_reference(['Cz'])
        events, all_events, ids = self.getMNEvents()
        if plotRaw:
            data.plot(events=all_events, event_id=ids,scalings='auto')
        print(len(all_events))
        epochs = mne.Epochs(data,all_events,detrend=1,tmin=-0.2,tmax=2.5, event_id=ids)
        epochs.apply_baseline(baseline=(None,0))
        if plotEpochs:
            epochs.plot(n_epochs=20,n_channels=4,events=True,scalings=None)
        return epochs
        
    def MNE_evoked(self):
        epochs = self.build_MNE_Epochs()
        events = epochs.event_id
        evoked = {}
        for key in events:
            evoked[key] = epochs[key].average()
        return evoked
    
    def getMNEvents(self):
        eventInfo = mne.create_info(['stim'],self.fs,['stim'])
        # stimData = np.round(self.states['StimulusCode'],0)
        stimData, ids = self.generateTriggerChannel()
        stim = np.empty([1,len(stimData)])
        stim[0][:] = stimData
        eventRaw = mne.io.RawArray(stim,eventInfo)
        all_events = mne.find_events(eventRaw,consecutive=False)
        meaningful = list(self.StimuliID.values())[0:-3]
        event_list = []
        for i,event in enumerate(all_events):
            if event[1] in meaningful:
                event_list.append(event)
        event_list = np.array(event_list)
        return event_list, all_events, ids

    """Visualization Functions"""
    def VisualizeRawOffset(self):
        data = self.dataStruct.EEG
        fig = plt.figure('Offset EEG Channels')
        ax = plt.subplot2grid((10,1),loc=(0,0),rowspan=9)
        colors = ('black', 'orange')
        for stim in self.estimIdx:
            ax.axvline(x=self.t[stim], c=colors[0])        
        for i,(lab,chan) in enumerate(data.items()):
            if i%4 == 0:
                c = colors[1]
            else:
                c = colors[0]
            ax.plot(self.t,chan+(self.offsetVal*i), c=c, alpha=0.9,label = lab)
        ax.set_ylabel('uV')
        ax2 = plt.subplot2grid((10,1),loc=(9,0))
        ax2.set_xlabel('Time (s)')
        self.plotStimulusCode(ax2)
        # self.plotEstim(ax)
        # ax.set_ylim = (-1, 2000)
        ax.get_shared_x_axes().join(ax,ax2)
        fig.add_axes(ax)
        fig.add_axes(ax2)
        fig.suptitle('Raw EEG during FEL Acquisition')        
    def whole_block_psds(self):
        data = self.dataStruct.EEG
        fig = plt.figure('PSD')
        ax = plt.subplot2grid((1,1),loc=(0,0))       
        window = sig.get_window(window='hamming', Nx = int(self.fs))
        for i,chan in enumerate(data):
            f,Pxx = sig.welch(chan,fs = self.fs,window=window, scaling='density') #hamming window
            # f = f[1:]
            # Pxx = Pxx[1:]
            ax.semilogy(f,Pxx, label = self.ChanLabs[i])
        print(f'freq resolution :{self.fs / (2 *len(f))}')
        ax.legend()
        ax.set_xlim(0,100)            
    def plotStimOn(self,ax=None):
        if ax == None:
            fig, ax = plt.subplots(1,1,num='StimPresentation')
        ax.plot(self.t,self.stimulusPresented)
    def plotEstimOn(self,ax=None):
        if ax == None:
            fig, ax = plt.subplots(1,1,num='EstimOn')
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
    def plotStimulusCode(self, ax=None):
        state = self.states['StimulusCode']
        state,_ = self.generateTriggerChannel()
        if ax == None:
            fig, ax = plt.subplots(1,1,num='stimCode')
        ax.plot(self.t,state,c=(0,0,0),alpha = 0.5, linewidth=0.5)
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
                ax.plot(vals,y, c=colors[count],label=lab, linewidth=3)
                lab = '_'
            count +=1
        ax.legend()
    def plotEpochs(self):
        data, channels = self._epochDataByStimuli()
        for face, signals in data.items():
            fig,ax = plt.subplots(1,1,num=face)
            for i,(channel,signal) in enumerate(signals.items()):
                offset = -self.offsetVal*i
                epochs = []
                for epoch in signal.values():
                    ax.plot(epoch+offset,c=(0,0,0),alpha=0.3)
                    epochs.append(epoch)
                avg = np.average(epochs,axis=0)
                ax.plot(avg+offset, c = (1,0,0))
    def plotEpochsSubplot(self):
        data,channels = self._epochDataByStimuli()
        fig,ax = plt.subplots(len(data),len(self.sigs),num='AllEpochs',sharex=True,sharey=True)
        for i,(face, signals) in enumerate(data.items()):
            for j,(channel,signal) in enumerate(signals.items()):
                offset = -self.offsetVal*i
                epochs = []
                for epoch in signal.values():
                    # ax[i][j].plot(epoch,c=(0,0,0),alpha=0.3)
                    epochs.append(epoch)
                avg = np.median(epochs,axis=0)
                ax[i][j].plot(avg, c = (1,0,0))
                ax[0][j].set_title(self.ChanLabs[j])
            ax[i][0].set_ylabel(face)
            labList = face.split("_")
            labList[-1] = labList[-1].split('.')[0] 
            if j > 3:
                labList[-1] = labList[-1] + ' stim'
            dec = labList[-1]
            ax[i][0].set_ylabel(f'{dec}\nuV**2/Hz')
            ax[i][0].yaxis.set_label_coords(-0.7, 0.5)
        # for a in ax.flat:
        #     a.set_ylim(-100,100)
        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            top = 0.93,
            bottom = 0.05,
            hspace = 0.09,
            wspace = 0.09
        )
    def epochFFT(self):
        data,channels = self._epochDataByStimuli()
        fig,ax = plt.subplots(len(data),len(self.sigs),num='Epoch FFT',sharex='col', sharey=True)
        window = sig.get_window(window='hamming', Nx = int(self.fs/2))
        for i,(face, signals) in enumerate(data.items()):
            
            for j,(epoch,signal) in enumerate(signals.items()):
                channel = channels[j]
                offset = -self.offsetVal*j
                epochs = []
                for epoch in signal.values():
                    f, Pxx = sig.welch(epoch,fs = self.fs,window=window)
                    # f = f[1:]
                    # Pxx = Pxx[1:]
                    ax[i][j].semilogy(f,Pxx,c=(0,0,0),alpha=0.3)
                    epochs.append(Pxx)
                avg = np.average(epochs,axis=0)
                med = np.median(epochs,axis=0)
                ax[i][j].semilogy(f,avg, c = (1,0,0))
                ax[i][j].semilogy(f,med, c = (0,0.5,0.5))
                if channel.find('GSR')>-1:
                    ax[i][j].set_xlim(0,10)
                elif channel.find('EMG')>-1:
                    ax[i][j].set_xlim(12,500)
                else:
                    ax[i][j].set_xlim(0.1,55)
                if self.chanLabelFlag:
                    ax[0][j].set_title(f'{channel}')
                else:
                    ax[0][j].set_title(f'Chan {j+1}')
            labList = face.split("_")
            labList[-1] = labList[-1].split('.')[0] 
            if i > 3:
                labList[-1] = labList[-1] + ' stim'
            dec = labList[-1]
            ax[i][0].set_ylabel(f'{dec}\nuV**2/Hz')
            ax[i][0].yaxis.set_label_coords(-0.7, 0.5)
        for a in ax.flat:
            a.label_outer()
        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            top = 0.93,
            bottom = 0.05,
            hspace = 0.09,
            wspace = 0.09
        )
    def plotPowerBands_oneChannel(self, channel, axes:plt.axes or bool = False, legend:bool=True, order:int = 4):
        """Plots all power bands of one channel on same figure. Channel is 1 indexed list of channels, ax is for an ax to be passed to add this figure to another subplot."""
        if axes:
            title = ''
            ax = axes
        else:
            title =  f'Channel {channel} Power Bands'
            fig, ax = plt.subplots(1,1,num =title)
         
        channel = channel -1
        data = self.all_signals[channel]
        delta = getDeltaBand(data,self.fs,order)
        theta = getThetaBand(data,self.fs,order)
        alpha = getAlphaBand(data,self.fs,order)
        beta  = getBetaBand( data,self.fs,order)
        gamma = getGammaBand(data,self.fs,order)
        bands = {'delta':delta, 'theta':theta, 'alpha':alpha, 'beta':beta, 'gamma':gamma}
        c = len(bands)
        colors = sns.color_palette(None,c)
        offset = 0
        keys = list(bands.keys())
        for k in keys[1:]:
            val = bands[k][20000:]
            pp = abs(max(val)) + abs(min(val))
            if pp >offset:
                offset = pp
        offset = offset + 5 
        for i,(band, val) in enumerate(bands.items()):
            ax.plot(self.t,val-(i*offset), label=band, c=colors[i])
        ax.set_xlabel('Times (s)')
        ax.set_ylabel('uV')
        ax.set_title(title)
        ax.set_ylim(-(i+1)*offset, offset)
        if legend:
            ax.legend()
    
    def plotEvokedPSD(self):
        evk = self.MNE_evoked()
        fig,ax = plt.subplots(len(evk),1,num='evoked')
        for i,(event,dat) in enumerate(evk.items()):
            dat.plot_psd(axes=ax[i])
            ax[i].set_xlim([0,55])
    
    def plotEvokedTopo(self,numTimes = 6):
        evk=self.MNE_evoked()
        manualTimes = np.array([-0.2, 0.05, 1, 2, 2.3, 2.4, 2.5])
        times = np.linspace(-.2,2.5,numTimes)
        numTimes = len(manualTimes)
        times = manualTimes
        fig,ax = plt.subplots(len(evk),numTimes+1,num='evoked topo')
        for i,(ev, dat) in enumerate(evk.items()):

            dat.plot_topomap(ch_type='eeg', times=times,colorbar=True,axes=ax[i][:])
            ax[i][0].set_ylabel(ev.split('.')[0])

    def show(self):
        plt.show()


