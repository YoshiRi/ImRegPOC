% Because I like matlab to plot image

clear all
close all

%%
gt = load('test1TrueParam.csv');
pocval = load('test1POCEstimation.csv');
pocpeak = load('test1POCpeak.csv');


%% Get Error

err = gt-pocval;
abserr = abs(err./gt);


t = 1:size(gt,1);

set(0,'defaultfigureposition',[-30000 0 300 300]')
figure(1)
plot(t,abserr,t,pocpeak,'r--')
ylim([0 1])
legend('x','y','\theta','\kappa','peak')