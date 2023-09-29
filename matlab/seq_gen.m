function [sequence stimorder] = seq_gen(reps)
sequence = randperm(4);
for i = 2:reps
    flag = true;
    while flag
        temp_seq = randperm(4);
        if sequence(end) ~= temp_seq(1)
            flag = false;
        end
    end
    sequence = [sequence temp_seq];
end
stimorder = [];
if ~mod(reps,2)
    inj = zeros(1,1 + reps/2);
    zerocounter = 0;
    flag = true;
    while sum(inj) ~= reps/2 || flag
        flag = false;
        for i = 1:(1+ reps/2)
            inj(i) = randi(3)-1;
            if inj(i) == 0
                zerocounter = zerocounter+1;
            else
                zerocounter = 0;
            end
            if zerocounter >= 2
                flag = true;
            end
        end
    end
    stimorder = 2*ones(1,inj(1));
    for i = 1:(reps/2)
        stimorder = [stimorder 1 2*ones(1,inj(i+1))];
    end
end

end
