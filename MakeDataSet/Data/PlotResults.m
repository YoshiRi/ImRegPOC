% Because I like matlab to plot image

clear all
close all

%% LOAD 

folder = 'Test1/'


gt = load([folder,'TrueParam.csv']);
pocval = load([folder,'POCEstimation.csv']);
pocpeak = load([folder,'POCpeak.csv']);


t = 1:size(gt,1);

%% Get Error POC

err = gt-pocval;
gtnz = gt;
gtnz(gt==0) = 0.01;
abserr = abs(err);
abserrp = abs(err./gtnz);

abserr(pocpeak<0.05,:) = 0;
AvgErr = sum(abserr)/sum(pocpeak>0.05)

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
% expfig(['POC_Estimation'],'-pdf');

%%  Err for SIFT
SIFTval = load([folder,'SIFTEstimation.csv']);
SIFTeval = load([folder,'FPnumSIFT.csv']);

SIFTerr = gt-SIFTval;
SIFTabserrp = abs(SIFTerr./gtnz);
SIFTabserr = abs(SIFTerr);

SIFTAvgErr = sum(SIFTabserr)/size(SIFTabserr,1)

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

ORBval = load([folder,'ORBEstimation.csv']);
ORBeval = load([folder,'FPnumORB.csv']);

ORBerr = gt-ORBval;
ORBabserrp = abs(ORBerr./gtnz);
ORBabserr = abs(ORBerr);

ORBAvgErr = sum(ORBabserr)/size(ORBabserr,1)

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

%%
AvgErr
SIFTAvgErr
ORBAvgErr