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
loadBCI2kTools;
Path = 'C:\BCI2000.x64\data\';
[x,y,parms]=load_bcidat(strcat(Path,folder,'\',filename,ext),1);
stimuli = parms.Stimuli.Value;
T = cell2table(stimuli(1:end,:));
csv = strcat(Path,folder,'\',filename,'.csv');
writetable(T,csv);
end