% BCI2000 Stimulus Presentation Demo Script
% 
% StimulusPresentationScript_Demo creates a parameter fragment that can be
% loaded into BCI2000 to create a stimulus presentation experiment.
% 
% This demo script will take the image files located in the BCI2000 prog
% directory and create a stimuli matrix containing these images, variable
% duration fixation cross stimuli, instructions, and a sync pulse. 
% 
% Change the n_rows and n_stimuli variables to store more information with
% the stimuli or add additional stimuli. Best practice is to separate
% stimuli into banks (e.g. 1-25, 101-125, etc) for easy evaluation later. 
% 
% Note that every stimulus needs to have an index for every row desired,
% even if that row label is not meaningful for the stimulus.
% 
% A sequence is created to alternate the fixation cross stimuli with the
% image stimuli.
% 
% The stimuli and meaningful parameters are written into a param
% variable and stored as a *.prm file using the convert_bciprm function.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: James Swift <swift@neurotechcenter.org>
%
% $BEGIN_BCI2000_LICENSE$
% 
% This file is part of BCI2000, a platform for real-time bio-signal research.
% [ Copyright (C) 2000-2021: BCI2000 team and many external contributors ]
% 
% BCI2000 is free software: you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
% 
% BCI2000 is distributed in the hope that it will be useful, but
%                         WITHOUT ANY WARRANTY
% - without even the implied warranty of MERCHANTABILITY or FITNESS FOR
% A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License along with
% this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% $END_BCI2000_LICENSE$
% http://www.bci2000.org 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the path of the BCI2000 main directory here
BCI2000pathparts = regexp(pwd,filesep,'split');
BCI2000path = '';
for i = 1:length(BCI2000pathparts)-2
    BCI2000path = [BCI2000path BCI2000pathparts{i} filesep];
end
settings.BCI2000path = BCI2000path;
clear BCI2000path BCI2000pathparts i

% Add BCI2000 tools to path
addpath(genpath(fullfile(settings.BCI2000path,'tools')))
blocknames = cell(1,8);
blocknames{1,1} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Habituation_Parameters.prm');
blocknames{1,2} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Acquisition_1_Parameters.prm');
blocknames{1,3} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Acquisition_2_Parameters.prm');
blocknames{1,4} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Acquisition_3_Parameters.prm');
blocknames{1,5} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Extinction_Parameters.prm');
blocknames{1,6} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Recall_1_Parameters.prm');
blocknames{1,7} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Recall_2_Parameters.prm');
blocknames{1,8} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Recall_3_Parameters.prm');
blocknames{1,9} = fullfile(settings.BCI2000path,'McDonnel PTSD paradigm','parms','FEL_Briefing_Parameters.prm');

% Get task images
WM_faces = dir('C:\Users\schalklab\Box\McDonnel PTSD paradigm\Stimuli\face slides');
WM_faces([1 2]) = [];
face_nums = randperm(length(WM_faces),6);
for i = 1:6
    face_paths(i,:) = [WM_faces(face_nums(i)).folder '\' WM_faces(face_nums(i)).name];
end
% face 5 is dummy face for extinction
% face 6 is example face used in briefing

scales = dir('C:\Users\schalklab\Box\McDonnel PTSD paradigm\Stimuli\likert');
scales([1 2]) = [];
for i = 1:4
    likert_paths(i,:) = [scales(i).folder '\' scales(i).name];
end

% Set up the different stimuli so they are represented by unique stimulus codes, separated into banks for easy evaluation later
n_stimuli = 100; % Total events
n_rows    = 8;

for blocknum = 1:9
    
% Settings
settings.SamplingRate          = '2000Hz'; % device sampling rate
settings.SampleBlockSize       = '200';     % number of samples in a block

settings.PreRunDuration        = '1s';
settings.PostRunDuration       = '0s';
settings.TaskDuration          = '2s';
settings.InstructionDuration   = '30s';
settings.SyncPulseDuration     = '1s';
settings.BaselineMinDuration   = '1s';
settings.BaselineMaxDuration   = '2s';
settings.NumberOfSequences     = '1';
settings.StimulusWidth         = '100';
settings.WindowTop             = '0';
settings.WindowLeft            = '1920';
settings.WindowWidth           = '2560';
settings.WindowHeight          = '1440';
settings.BackgroundColor       = '0x808080';
settings.CaptionColor          = '0xFFFFFF';
settings.CaptionSwitch         = '1';
settings.WindowBackgroundColor = '0x808080';
settings.ISIMinDuration        = '0s';
settings.ISIMaxDuration        = '0s';
settings.SubjectName           = 'BCI';
settings.DataDirectory         = fullfile('..','data');
settings.SubjectSession        = 'auto';
settings.SubjectRun            = '01';
settings.parm_filename         = blocknames{1,blocknum};
settings.UserComment           = 'Enter user comment here';
settings.InstructionsCaption   = {'Stimulus Presentation Task. Press space to continue'; 'End of task.'};

% Set up Stimuli
param.Stimuli.Section         = 'Application';
param.Stimuli.Type            = 'matrix';
param.Stimuli.DefaultValue    = '';
param.Stimuli.LowRange        = '';
param.Stimuli.HighRange       = '';
param.Stimuli.Comment         = 'captions and icons to be displayed, sounds to be played for different stimuli';
param.Stimuli.Value           = cell(n_rows,n_stimuli);
param.Stimuli.Value(:)        = {''};
param.Stimuli.RowLabels       = cell(n_rows,1);
param.Stimuli.RowLabels(:)    = {''};
param.Stimuli.ColumnLabels    = cell(1,n_stimuli);
param.Stimuli.ColumnLabels(:) = {''};

param.Stimuli.RowLabels{1}  = 'caption';
param.Stimuli.RowLabels{2}  = 'icon';
param.Stimuli.RowLabels{3}  = 'audio';
param.Stimuli.RowLabels{4}  = 'StimulusDuration';
param.Stimuli.RowLabels{5}  = 'AudioVolume';
param.Stimuli.RowLabels{6}  = 'Category';
param.Stimuli.RowLabels{7}  = 'EarlyOffsetExpression';
param.Stimuli.RowLabels{8}  = 'EstimOn';

% Images 1-5 task faces without shock
img_titles = ['CS+E'; 'CS+N'; 'CS-E'; 'CS-N'; 'dumb'];
for idx = 1:5
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = ''; %caption
    param.Stimuli.Value{2,idx}      = face_paths(idx,:);
    param.Stimuli.Value{3,idx}      = ''; %audio
    param.Stimuli.Value{4,idx}      = '3s'; %stimulusDuration
    param.Stimuli.Value{5,idx}      = '0'; %audioVolume
    param.Stimuli.Value{6,idx}      = 'face only'; %category
    param.Stimuli.Value{7,idx}      = ''; %EarlyOffsetExpression
    param.Stimuli.Value{8,idx}      = '0'; %EstimOn
end

% Images 6-7 task faces shortened faces for shock
for idx = 6:7
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = ''; %caption
    param.Stimuli.Value{2,idx}      = face_paths(idx-5,:);
    param.Stimuli.Value{3,idx}      = ''; %audio
    param.Stimuli.Value{4,idx}      = '2.9s'; %stimulusDuration
    param.Stimuli.Value{5,idx}      = '0'; %audioVolume
    param.Stimuli.Value{6,idx}      = 'face pre shock'; %category
    param.Stimuli.Value{7,idx}      = ''; %EarlyOffsetExpression
    param.Stimuli.Value{8,idx}      = '0'; %EstimOn
end

% Images 8-9 task faces paired with shock
for idx = 8:9
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = ''; %caption
    param.Stimuli.Value{2,idx}      = face_paths(idx-7,:);
    param.Stimuli.Value{3,idx}      = 'C:\bci2000.x64\prog\sounds\Sine.wav'; %audio
    param.Stimuli.Value{4,idx}      = '0.1s'; %stimulusDuration
    param.Stimuli.Value{5,idx}      = '0'; %audioVolume
    param.Stimuli.Value{6,idx}      = 'face w shock'; %category
    param.Stimuli.Value{7,idx}      = ''; %EarlyOffsetExpression
    param.Stimuli.Value{8,idx}      = '1'; %EstimOn;
end

% blanks 12:29
% 3 +/- 1s
SamplingRate = str2double(settings.SamplingRate(1:end-2));
BlockSize    = str2double(settings.SampleBlockSize);
MinDuration  = 2;
MaxDuration  = 4;
for idx = 12:29
    blockvals = MinDuration:BlockSize/SamplingRate:MaxDuration;
    randval   = randi(length(blockvals));
    duration  = blockvals(randval);
    
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = '';
    param.Stimuli.Value{2,idx}      = '';
    param.Stimuli.Value{3,idx}      = '';
    param.Stimuli.Value{4,idx}      = [num2str(duration) 's'];
    param.Stimuli.Value{5,idx}      = '0';      
    param.Stimuli.Value{6,idx}      = 'blank'; 
    param.Stimuli.Value{7,idx}      = '';
    param.Stimuli.Value{8,idx}      = '0';
end

% fixation cross 30:47
% 1.25 +/- 0.25s
SamplingRate = str2double(settings.SamplingRate(1:end-2));
BlockSize    = str2double(settings.SampleBlockSize);
MinDuration  = 1;
MaxDuration  = 1.5;
for idx = 30:47
    blockvals = MinDuration:BlockSize/SamplingRate:MaxDuration;
    randval   = randi(length(blockvals));
    duration  = blockvals(randval);
    
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = '+';
    param.Stimuli.Value{2,idx}      = '';
    param.Stimuli.Value{3,idx}      = '';
    param.Stimuli.Value{4,idx}      = [num2str(duration) 's'];
    param.Stimuli.Value{5,idx}      = '0';      
    param.Stimuli.Value{6,idx}      = 'fixation'; 
    param.Stimuli.Value{7,idx}      = '';
    param.Stimuli.Value{8,idx}      = '0';
end

% Image 70 briefing face
idx = 70;
param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
param.Stimuli.Value{1,idx}      = ''; %caption
param.Stimuli.Value{2,idx}      = face_paths(6,:);
param.Stimuli.Value{3,idx}      = ''; %audio
param.Stimuli.Value{4,idx}      = '3s'; %stimulusDuration
param.Stimuli.Value{5,idx}      = '0'; %audioVolume
param.Stimuli.Value{6,idx}      = 'face only'; %category
param.Stimuli.Value{7,idx}      = ''; %EarlyOffsetExpression
param.Stimuli.Value{8,idx}      = '0'; %EstimOn

% Instructions 84-85
idx_iter = 1;
for idx = 84:83+length(settings.InstructionsCaption)
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = settings.InstructionsCaption{idx_iter};
    param.Stimuli.Value{2,idx}      = '';
    param.Stimuli.Value{3,idx}      = '';
    param.Stimuli.Value{4,idx}      = settings.InstructionDuration;
    param.Stimuli.Value{5,idx}      = '0';    
    param.Stimuli.Value{6,idx}      = 'instruction'; 
    param.Stimuli.Value{7,idx}      = 'KeyDown == 32'; % space key 
    param.Stimuli.Value{8,idx}      = '0'; %EstimOn
    idx_iter = idx_iter + 1;

end

% Sync pulse 87
idx = 87;
param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
param.Stimuli.Value{1,idx}      = '';
param.Stimuli.Value{2,idx}      = '';
param.Stimuli.Value{3,idx}      = '';
param.Stimuli.Value{4,idx}      = settings.SyncPulseDuration;
param.Stimuli.Value{5,idx}      = '0';      
param.Stimuli.Value{6,idx}      = 'sync'; 
param.Stimuli.Value{7,idx}      = '';
param.Stimuli.Value{8,idx}      = '0'; %EstimOn

% Likerts 88-91
for idx = 88:91
    param.Stimuli.ColumnLabels{idx} = sprintf('%d',idx);
    param.Stimuli.Value{1,idx}      = '';
    param.Stimuli.Value{2,idx}      = likert_paths(idx - 87,:);
    param.Stimuli.Value{3,idx}      = '';
    param.Stimuli.Value{4,idx}      = settings.InstructionDuration;
    param.Stimuli.Value{5,idx}      = '0';    
    param.Stimuli.Value{6,idx}      = 'instruction'; 
    param.Stimuli.Value{7,idx}      = 'KeyDown == 49 || KeyDown == 50 || KeyDown == 51 || KeyDown == 52 || KeyDown == 53 || KeyDown == 54 || KeyDown == 55 || KeyDown == 56 || KeyDown == 57'; 
    param.Stimuli.Value{8,idx}      = '0'; %EstimOn
end

for i = 1:n_stimuli
    if isempty(param.Stimuli.Value{8,i})
        param.Stimuli.Value{8,i} = '0';
    end
end

if blocknum == 1
% Habituation Sequence
    faces = repelem(1:4,4);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    seq_hab = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_hab = [seq_hab cross_1 cross_2 faces(f)];
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_hab = [seq_hab cross_1 cross_2 likert_seq 85 87]';

    labeled = cell(length(seq_hab),2);
    for i = 1:length(seq_hab)
        labeled{i,1} = seq_hab(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_hab(i)};
    end
    seq = seq_hab;
elseif blocknum == 2
% Acquisition Sequence 1
    faces = repelem(1:4,16);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    f1shock = [ones(1,8) zeros(1,8)];
    f1shock = f1shock(randperm(length(f1shock)));

    f2shock = [ones(1,8) zeros(1,8)];
    f2shock = f2shock(randperm(length(f2shock)));

    f1s = 0;
    f2s = 0;

    seq_acq1 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_acq1 = [seq_acq1 cross_1 cross_2 faces(f)];
        if seq_acq1(end) == 1
            f1s = f1s+1;
            if f1shock(f1s)
                seq_acq1(end) = [];
                seq_acq1 = [seq_acq1 6 8];
            end
        elseif seq_acq1(end) == 2
            f2s = f2s+1;
            if f2shock(f2s)
                seq_acq1(end) = [];
                seq_acq1 = [seq_acq1 7 9];
            end
        end
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_acq1 = [seq_acq1 cross_1 cross_2 likert_seq 85 87]';

    labeled = cell(length(seq_acq1),2);
    for i = 1:length(seq_acq1)
        labeled{i,1} = seq_acq1(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_acq1(i)};
    end
    seq = seq_acq1;
elseif blocknum == 3
% Acquisition Sequence 2
    faces = repelem(1:4,17);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    f1shock = [ones(1,8) zeros(1,9)];
    f1shock = f1shock(randperm(length(f1shock)));

    f2shock = [ones(1,8) zeros(1,9)];
    f2shock = f2shock(randperm(length(f2shock)));

    f1s = 0;
    f2s = 0;

    seq_acq2 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_acq2 = [seq_acq2 cross_1 cross_2 faces(f)];
        if seq_acq2(end) == 1
            f1s = f1s+1;
            if f1shock(f1s)
                seq_acq2(end) = [];
                seq_acq2 = [seq_acq2 6 8];
            end
        elseif seq_acq2(end) == 2
            f2s = f2s+1;
            if f2shock(f2s)
                seq_acq2(end) = [];
                seq_acq2 = [seq_acq2 7 9];
            end
        end
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_acq2 = [seq_acq2 cross_1 cross_2 likert_seq 85 87]';

    labeled = cell(length(seq_acq2),2);
    for i = 1:length(seq_acq2)
        labeled{i,1} = seq_acq2(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_acq2(i)};
    end
elseif blocknum == 4
% Acquisition Sequence 3
    faces = repelem(1:4,17);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    f1shock = [ones(1,8) zeros(1,9)];
    f1shock = f1shock(randperm(length(f1shock)));

    f2shock = [ones(1,8) zeros(1,9)];
    f2shock = f2shock(randperm(length(f2shock)));

    f1s = 0;
    f2s = 0;

    seq_acq3 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_acq3 = [seq_acq3 cross_1 cross_2 faces(f)];
        if seq_acq3(end) == 1
            f1s = f1s+1;
            if f1shock(f1s)
                seq_acq3(end) = [];
                seq_acq3 = [seq_acq3 6 8];
            end
        elseif seq_acq3(end) == 2
            f2s = f2s+1;
            if f2shock(f2s)
                seq_acq3(end) = [];
                seq_acq3 = [seq_acq3 7 9];
            end
        end
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_acq3 = [seq_acq3 cross_1 cross_2 likert_seq 85 87]';

    labeled = cell(length(seq_acq3),2);
    for i = 1:length(seq_acq3)
        labeled{i,1} = seq_acq3(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_acq3(i)};
    end
    seq = seq_acq3;
elseif blocknum == 5
% Extinction Sequence
    pass_combine = 0;
    tic
    while ~pass_combine
        faces_a = [repelem([1 3],10) 5*ones(1,3)];
        pass_face = 0;
        pass_US = 0;
        while ~(pass_face && pass_US)
            faces_a = faces_a(randperm(length(faces_a)));
            pass_face = repeated_faces(faces_a);
            pass_US = repeated_US(faces_a);
        end

        faces_b = [repelem([1 3],10) 5*ones(1,3)];
        pass_face = 0;
        pass_US = 0;
        while ~(pass_face && pass_US)
            faces_b = faces_b(randperm(length(faces_b)));
            pass_face = repeated_faces(faces_b);
            pass_US = repeated_US(faces_b);
        end

        faces_c = [repelem([1 3],10) 5*ones(1,4)];
        pass_face = 0;
        pass_US = 0;
        while ~(pass_face && pass_US)
            faces_c = faces_c(randperm(length(faces_c)));
            pass_face = repeated_faces(faces_c);
            pass_US = repeated_US(faces_c);
        end
        faces = [faces_a faces_b faces_c];
        pass_combine = repeated_faces(faces) && repeated_US(faces);
    end
    toc

    seq_ext = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_ext = [seq_ext cross_1 cross_2 faces(f)];
    end
    likert_seq = write_likert_seq([1 3 5]);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_ext = [seq_ext cross_1 cross_2 likert_seq 85 87]';

    labeled = cell(length(seq_ext),2);
    for i = 1:length(seq_ext)
        labeled{i,1} = seq_ext(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_ext(i)};
    end
    seq = seq_ext;
elseif blocknum == 6
% Recall Sequence 1
    faces = repelem(1:4,17);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    seq_rec1 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_rec1 = [seq_rec1 cross_1 cross_2 faces(f)];
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_rec1 = [seq_rec1 cross_1 cross_2 likert_seq 85 87];

    labeled = cell(length(seq_rec1),2);
    for i = 1:length(seq_rec1)
        labeled{i,1} = seq_rec1(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_rec1(i)};
    end
    seq = seq_rec1;
elseif blocknum == 7
% Recall Sequence 2
    faces = repelem(1:4,17);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    seq_rec2 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_rec2 = [seq_rec2 cross_1 cross_2 faces(f)];
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_rec2 = [seq_rec2 cross_1 cross_2 likert_seq 85 87];

    labeled = cell(length(seq_rec2),2);
    for i = 1:length(seq_rec2)
        labeled{i,1} = seq_rec2(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_rec2(i)};
    end
    seq = seq_rec2;
elseif blocknum == 8
% Recall Sequence 3
    faces = repelem(1:4,17);
    pass_face = 0;
    pass_US = 0;
    tic
    while ~(pass_face && pass_US)
        faces = faces(randperm(length(faces)));
        pass_face = repeated_faces(faces);
        pass_US = repeated_US(faces);
    end
    toc

    seq_rec3 = [87 84];
    for f = 1:length(faces)
        cross_1 = randi(18)+11;
        cross_2 = randi(18)+29;
        seq_rec3 = [seq_rec3 cross_1 cross_2 faces(f)];
    end
    likert_seq = write_likert_seq(1:4);
    cross_1 = randi(18)+11;
    cross_2 = randi(18)+29;
    seq_rec3 = [seq_rec3 cross_1 cross_2 likert_seq 85 87];

    labeled = cell(length(seq_rec3),2);
    for i = 1:length(seq_rec3)
        labeled{i,1} = seq_rec3(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_rec3(i)};
    end
    seq = seq_rec3;
elseif blocknum == 9
    tic
    seq_brief = [87 84 70 88 70 90 85 87]';
    toc
    labeled = cell(length(seq_brief),2);
    for i = 1:length(seq_brief)
        labeled{i,1} = seq_brief(i);
        labeled{i,2} = param.Stimuli.Value{6,seq_brief(i)};
    end
    seq = seq_brief;
end

%

param.Sequence.Section      = 'Application';
param.Sequence.Type         = 'intlist';
param.Sequence.DefaultValue = '1';
param.Sequence.LowRange     = '1';
param.Sequence.HighRange    = '';
param.Sequence.Comment      = 'Sequence in which stimuli are presented (deterministic mode)/ Stimulus frequencies for each stimulus (random mode)';
param.Sequence.Value        = cellfun(@num2str, num2cell(seq), 'un',0);
param.Sequence.NumericValue = seq;

% UserComment
param.UserComment.Section         = 'Application';
param.UserComment.Type            = 'string';
param.UserComment.DefaultValue    = '';
param.UserComment.LowRange        = '';
param.UserComment.HighRange       = '';
param.UserComment.Comment         = 'User comments for a specific run';
param.UserComment.Value           = {settings.UserComment};

%
param.SamplingRate.Section         = 'Source';
param.SamplingRate.Type            = 'int';
param.SamplingRate.DefaultValue    = '256Hz';
param.SamplingRate.LowRange        = '1';
param.SamplingRate.HighRange       = '';
param.SamplingRate.Comment         = 'sample rate';
param.SamplingRate.Value           = {settings.SamplingRate};

%
param.SampleBlockSize.Section         = 'Source';
param.SampleBlockSize.Type            = 'int';
param.SampleBlockSize.DefaultValue    = '8';
param.SampleBlockSize.LowRange        = '1';
param.SampleBlockSize.HighRange       = '';
param.SampleBlockSize.Comment         = 'number of samples transmitted at a time';
param.SampleBlockSize.Value           = {settings.SampleBlockSize};

%
param.NumberOfSequences.Section         = 'Application';
param.NumberOfSequences.Type            = 'int';
param.NumberOfSequences.DefaultValue    = '1';
param.NumberOfSequences.LowRange        = '0';
param.NumberOfSequences.HighRange       = '';
param.NumberOfSequences.Comment         = 'number of sequence repetitions in a run';
param.NumberOfSequences.Value           = {settings.NumberOfSequences};

%
param.StimulusWidth.Section         = 'Application';
param.StimulusWidth.Type            = 'int';
param.StimulusWidth.DefaultValue    = '0';
param.StimulusWidth.LowRange        = '';
param.StimulusWidth.HighRange       = '';
param.StimulusWidth.Comment         = 'StimulusWidth in percent of screen width (zero for original pixel size)';
param.StimulusWidth.Value           = {settings.StimulusWidth};

%
param.SequenceType.Section              = 'Application';
param.SequenceType.Type                 = 'int';
param.SequenceType.DefaultValue         = '0';
param.SequenceType.LowRange             = '0';
param.SequenceType.HighRange            = '1';
param.SequenceType.Comment              = 'Sequence of stimuli is 0 deterministic, 1 random (enumeration)';
param.SequenceType.Value                = {'0'};

%
param.StimulusDuration.Section           = 'Application';
param.StimulusDuration.Type              = 'float';
param.StimulusDuration.DefaultValue      = '40ms';
param.StimulusDuration.LowRange          = '0';
param.StimulusDuration.HighRange         = '';
param.StimulusDuration.Comment           = 'stimulus duration';
param.StimulusDuration.Value             = {};

%
param.ISIMaxDuration.Section       = 'Application';
param.ISIMaxDuration.Type          = 'float';
param.ISIMaxDuration.DefaultValue  = '80ms';
param.ISIMaxDuration.LowRange      = '0';
param.ISIMaxDuration.HighRange     = '';
param.ISIMaxDuration.Comment       = 'maximum duration of inter-stimulus interval';
param.ISIMaxDuration.Value         = {settings.ISIMaxDuration};

%
param.ISIMinDuration.Section       = 'Application';
param.ISIMinDuration.Type          = 'float';
param.ISIMinDuration.DefaultValue  = '80ms';
param.ISIMinDuration.LowRange      = '0';
param.ISIMinDuration.HighRange     = '';
param.ISIMinDuration.Comment       = 'minimum duration of inter-stimulus interval';
param.ISIMinDuration.Value         = {settings.ISIMinDuration};

%
param.PreSequenceDuration.Section       = 'Application';
param.PreSequenceDuration.Type          = 'float';
param.PreSequenceDuration.DefaultValue  = '2s';
param.PreSequenceDuration.LowRange      = '0';
param.PreSequenceDuration.HighRange     = '';
param.PreSequenceDuration.Comment       = 'pause preceding sequences/sets of intensifications';
param.PreSequenceDuration.Value         = {'0s'};

%
param.PostSequenceDuration.Section       = 'Application';
param.PostSequenceDuration.Type          = 'float';
param.PostSequenceDuration.DefaultValue  = '2s';
param.PostSequenceDuration.LowRange      = '0';
param.PostSequenceDuration.HighRange     = '';
param.PostSequenceDuration.Comment       = 'pause following sequences/sets of intensifications';
param.PostSequenceDuration.Value         = {'0s'};

%
param.PreRunDuration.Section       = 'Application';
param.PreRunDuration.Type          = 'float';
param.PreRunDuration.DefaultValue  = '2000ms';
param.PreRunDuration.LowRange      = '0';
param.PreRunDuration.HighRange     = '';
param.PreRunDuration.Comment       = 'pause preceding first sequence';
param.PreRunDuration.Value         = {settings.PreRunDuration};

%
param.PostRunDuration.Section       = 'Application';
param.PostRunDuration.Type          = 'float';
param.PostRunDuration.DefaultValue  = '2000ms';
param.PostRunDuration.LowRange      = '0';
param.PostRunDuration.HighRange     = '';
param.PostRunDuration.Comment       = 'pause following last squence';
param.PostRunDuration.Value         = {settings.PostRunDuration};


%
param.BackgroundColor.Section      = 'Application';
param.BackgroundColor.Type         = 'string';
param.BackgroundColor.DefaultValue = '0x00FFFF00';
param.BackgroundColor.LowRange     = '0x00000000';
param.BackgroundColor.HighRange    = '0x00000000';
param.BackgroundColor.Comment      = 'Color of stimulus background (color)';
param.BackgroundColor.Value        = {settings.BackgroundColor};

%
param.CaptionColor.Section      = 'Application';
param.CaptionColor.Type         = 'string';
param.CaptionColor.DefaultValue = '0x00FFFF00';
param.CaptionColor.LowRange     = '0x00000000';
param.CaptionColor.HighRange    = '0x00000000';
param.CaptionColor.Comment      = 'Color of stimulus caption text (color)';
param.CaptionColor.Value        = {settings.CaptionColor};

%
param.WindowBackgroundColor.Section      = 'Application';
param.WindowBackgroundColor.Type         = 'string';
param.WindowBackgroundColor.DefaultValue = '0x00FFFF00';
param.WindowBackgroundColor.LowRange     = '0x00000000';
param.WindowBackgroundColor.HighRange    = '0x00000000';
param.WindowBackgroundColor.Comment      = 'background color (color)';
param.WindowBackgroundColor.Value        = {settings.WindowBackgroundColor};

%
param.IconSwitch.Section          = 'Application';
param.IconSwitch.Type             = 'int';
param.IconSwitch.DefaultValue     = '1';
param.IconSwitch.LowRange         = '0';
param.IconSwitch.HighRange        = '1';
param.IconSwitch.Comment          = 'Present icon files (boolean)';
param.IconSwitch.Value            = {'1'};

%
param.AudioSwitch.Section         = 'Application';
param.AudioSwitch.Type            = 'int';
param.AudioSwitch.DefaultValue    = '1';
param.AudioSwitch.LowRange        = '0';
param.AudioSwitch.HighRange       = '1';
param.AudioSwitch.Comment         = 'Present audio files (boolean)';
param.AudioSwitch.Value           = {'0'};

%
param.CaptionSwitch.Section       = 'Application';
param.CaptionSwitch.Type          = 'int';
param.CaptionSwitch.DefaultValue  = '1';
param.CaptionSwitch.LowRange      = '0';
param.CaptionSwitch.HighRange     = '1';
param.CaptionSwitch.Comment       = 'Present captions (boolean)';
param.CaptionSwitch.Value         = {settings.CaptionSwitch};

%
param.WindowHeight.Section        = 'Application';
param.WindowHeight.Type           = 'int';
param.WindowHeight.DefaultValue   = '480';
param.WindowHeight.LowRange       = '0';
param.WindowHeight.HighRange      = '';
param.WindowHeight.Comment        = 'height of application window';
param.WindowHeight.Value          = {settings.WindowHeight};

%
param.WindowWidth.Section        = 'Application';
param.WindowWidth.Type           = 'int';
param.WindowWidth.DefaultValue   = '480';
param.WindowWidth.LowRange       = '0';
param.WindowWidth.HighRange      = '';
param.WindowWidth.Comment        = 'width of application window';
param.WindowWidth.Value          = {settings.WindowWidth};

%
param.WindowLeft.Section        = 'Application';
param.WindowLeft.Type           = 'int';
param.WindowLeft.DefaultValue   = '0';
param.WindowLeft.LowRange       = '';
param.WindowLeft.HighRange      = '';
param.WindowLeft.Comment        = 'screen coordinate of application window''s left edge';
param.WindowLeft.Value          = {settings.WindowLeft};

%
param.WindowTop.Section        = 'Application';
param.WindowTop.Type           = 'int';
param.WindowTop.DefaultValue   = '0';
param.WindowTop.LowRange       = '';
param.WindowTop.HighRange      = '';
param.WindowTop.Comment        = 'screen coordinate of application window''s top edge';
param.WindowTop.Value          = {settings.WindowTop};

%
param.CaptionHeight.Section      = 'Application';
param.CaptionHeight.Type         = 'int';
param.CaptionHeight.DefaultValue = '0';
param.CaptionHeight.LowRange     = '0';
param.CaptionHeight.HighRange    = '100';
param.CaptionHeight.Comment      = 'Height of stimulus caption text in percent of screen height';
param.CaptionHeight.Value        = {'5'};

%
param.WarningExpression.Section      = 'Filtering';
param.WarningExpression.Type         = 'string';
param.WarningExpression.DefaultValue = '';
param.WarningExpression.LowRange     = '';
param.WarningExpression.HighRange    = '';
param.WarningExpression.Comment      = 'expression that results in a warning when it evaluates to true';
param.WarningExpression.Value        = {''};

%
param.Expressions.Section      = 'Filtering';
param.Expressions.Type         = 'matrix';
param.Expressions.DefaultValue = '';
param.Expressions.LowRange     = '';
param.Expressions.HighRange    = '';
param.Expressions.Comment      = 'expressions used to compute the output of the ExpressionFilter';
param.Expressions.Value        = {''};

%
param.SubjectName.Section      = 'Storage';
param.SubjectName.Type         = 'string';
param.SubjectName.DefaultValue = 'Name';
param.SubjectName.LowRange     = '';
param.SubjectName.HighRange    = '';
param.SubjectName.Comment      = 'subject alias';
param.SubjectName.Value        = {settings.SubjectName};

%
param.DataDirectory.Section      = 'Storage';
param.DataDirectory.Type         = 'string';
param.DataDirectory.DefaultValue = strcat('..\data');
param.DataDirectory.LowRange     = '';
param.DataDirectory.HighRange    = '';
param.DataDirectory.Comment      = 'path to top level data directory (directory)';
param.DataDirectory.Value        = {settings.DataDirectory};

%
param.SubjectRun.Section      = 'Storage';
param.SubjectRun.Type         = 'string';
param.SubjectRun.DefaultValue = '00';
param.SubjectRun.LowRange     = '';
param.SubjectRun.HighRange    = '';
param.SubjectRun.Comment      = 'two-digit run number';
param.SubjectRun.Value        = {settings.SubjectRun};

%
param.SubjectSession.Section      = 'Storage';
param.SubjectSession.Type         = 'string';
param.SubjectSession.DefaultValue = '00';
param.SubjectSession.LowRange     = '';
param.SubjectSession.HighRange    = '';
param.SubjectSession.Comment      = 'three-digit session number';
param.SubjectSession.Value        = {settings.SubjectSession};

% write the param struct to a bci2000 parameter file
    
parameter_lines = convert_bciprm( param );
fid = fopen(settings.parm_filename, 'w');

for i=1:length(parameter_lines)
    fprintf( fid, '%s', parameter_lines{i} );
    fprintf( fid, '\r\n' );
end
fclose(fid);
end
% helper functions

function pass = repeated_faces(sequence)
    pass = 1;
    for i = 3:length(sequence)
        segment = sequence(i-2:i);
        if segment(1) == segment(2) && segment(2) == segment(3)
            pass = 0;
        end
    end
end

function pass = repeated_US(sequence)
    pass = 1;
    sequence = sequence == 1 | sequence == 2;
    for i = 3:length(sequence)
        segment = sequence(i-2:i);
        if segment(1) == segment(2) && segment(2) == segment(3)
            pass = 0;
        end
    end
end

function sequence = write_likert_seq(facelist)
    facelist = facelist(randperm(length(facelist)));
    sequence = [];
    for i = 1:length(facelist)
        sequence = [sequence facelist(i) 89];
    end
    for i = 1:length(facelist)
        sequence = [sequence facelist(i) 91];
    end
end