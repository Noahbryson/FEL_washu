function stimseq = convert_to_stimseq(facelist, sequence, stimorder)
% first four entries in faces are nonstim
% entry 5 replaces entry 1 for stim
% entry 6 replaces entry 2 for stim
% entry 7 follows entry 5 for stim
% entry 8 follows entry 6 for stim

stimorder = stimorder - 1;
j = 0;
stimseq = [];
for i = 1:4:length(sequence)
    j = j+1;
    old_chunk = sequence(i:i+3);
    if stimorder(j)
        pos = find(old_chunk==facelist(2));
        new_chunk = [old_chunk(1:pos-1) facelist(6) facelist(8) old_chunk(pos+1:end)];
    else
        pos = find(old_chunk==facelist(1));
        new_chunk = [old_chunk(1:pos-1) facelist(5) facelist(7) old_chunk(pos+1:end)];
    end
    stimseq = [stimseq new_chunk];
end
end