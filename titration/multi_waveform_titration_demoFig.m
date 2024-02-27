fs=1000;
duration = 1;
[t,y,base,carrier] = modulateSquareWaves(duration,50,8.33,fs,1,0);

new = zeros(size(carrier));
count = 1;
for i=1:length(carrier)
if count < 5
    new(i) = 1;
elseif count == 1201
       count = 0;
end
count = count+1;
end
tspan = linspace(-1.25,duration,(duration+1.25)*fs);
spes = zeros(1.25*fs,1)';
spes(250:255) = 1;
low = cat(2,spes, carrier);
high = cat(2,spes, base);
theta = cat(2,spes, y);
close all
fig = figure(1);
subplot(3,1,1)
plot(tspan,low,'Color','r');
title('8.33 Hz Stim')
ylim([-0.1,1.1])
xlim([-1.25 duration])
subplot(3,1,2)
plot(tspan,high,'Color','r');
title('50 Hz Stim')
ylim([-0.1,1.1])
xlim([-1.25 duration])
subplot(3,1,3)
plot(tspan,theta,'Color','r');
title('Theta Burst Stim')
ylim([-0.1,1.1])
xlim([-1.25 duration])
exportgraphics(fig,'FEL_washu/titration/stim_titration.png','Resolution',300);