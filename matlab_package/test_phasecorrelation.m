%% read
ref = imread('../python_package/imgs/ref.png');
con = imtranslate(ref,[10,0.5]);


% fuga
[dx,dy,peak,Pt]=PhaseCorrelation(ref,con);

display(peak)
display(dy)