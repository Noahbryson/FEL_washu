function Stimuli2Csv(folder, filename)
%%
%filename is the name of the file without the .dat extension
%%
ext = strfind(filename,'.dat');
if ~isempty(ext)
    ext = '';
else
    ext='.dat';
end
olddir = pwd;
cd('C:\BCI2000.x64');   % The absolute path has to be hardcoded somewhere, and here it is.  Watch out, in case this is (or becomes) incorrect 
cd tools, cd matlab;
bci2000path -AddToMatlabPath tools/matlab;
bci2000path -AddToMatlabPath tools/mex;
bci2000path -AddToSystemPath tools/cmdline;   % required so that BCI2000CHAIN can call the command-line tools
cd(olddir); % change directory back to where we were before
clear olddir;
Path = 'C:\BCI2000.x64\data\';
[~,~,parms]=load_bcidat(strcat(Path,folder,'\',filename,ext),1);
stimuli = parms.Stimuli.Value;
T = cell2table(stimuli(1:end,:));
csv = strcat(Path,folder,'\',filename,'.csv');
writetable(T,csv);
end