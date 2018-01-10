% Because I like matlab to plot image

clear all
close all

%% LOAD 
gt = load('TrueParam.csv');
pocval = load('POCEstimation.csv');
pocpeak = load('POCpeak.csv');


t = 1:size(gt,1);

%% Get Error POC

err = gt-pocval;
gtnz = gt;
gtnz(gt==0) = 0.01
abserrp = abs(err./gtnz);



% set(0,'defaultfigureposition',[25000 0 300 300]')

hfig = figure(1)
plot(t,abserrp,t,pocpeak,'r--')
ylim([0 1])
legend('x','y','$\theta$','$\kappa$','peak')
xlabel('Number of Image')
ylabel('Error [%], poc peak ')
title('Relationship between POC peak and Estimation Error')
grid on
pfig = pubfig(hfig);
pfig.LegendLoc = 'best';
pfig.FigDim = [15 11];
expfig(['POC_Estimation'],'-pdf');

%%  Err for SIFT
SIFTval = load('SIFTEstimation.csv');
SIFTeval = load('FPnumSIFT.csv');

SIFTerr = gt-SIFTval;
SIFTabserrp = abs(SIFTerr./gtnz);

hfig = figure(2)
plot(t,SIFTabserrp)
ylim([0 1])
legend('x','y','$\theta$','$\kappa$','peak')
xlabel('Number of Image')
ylabel('Error [%], poc peak ')
title('Relationship between SIFT peak and Estimation Error')
grid on
pfig = pubfig(hfig);
pfig.LegendLoc = 'best';
pfig.FigDim = [15 11];
% expfig(['POC_Estimation'],'-pdf');


Epoc = sum(abserrp,2);
Esift = sum(SIFTabserrp,2);


hfig = figure(3)
plot(t,Epoc,t,Esift)
ylim([0 1])
legend('poc','sift')
xlabel('Number of Image')
ylabel('Error [%], poc peak ')
title('Relationship between SIFT peak and Estimation Error')
grid on
pfig = pubfig(hfig);
pfig.LegendLoc = 'best';
pfig.FigDim = [15 11];


%% Eval for ORB

ORBval = load('ORBEstimation.csv');
ORBeval = load('FPnumORB.csv');

ORBerr = gt-ORBval;
ORBabserrp = abs(ORBerr./gtnz);

hfig = figure(4)
plot(t,ORBabserrp)
ylim([0 1])
legend('x','y','$\theta$','$\kappa$','peak')
xlabel('Number of Image')
ylabel('Error [%], poc peak ')
title('Relationship between ORB peak and Estimation Error')
grid on
pfig = pubfig(hfig);
pfig.LegendLoc = 'best';
pfig.FigDim = [15 11];
