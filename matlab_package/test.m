%% test for matlab
% you need Image Processing toolbox for MATLAB

%% read images with gray scale
ref = imread('../python_package/ref.png');
cmp = imread('../python_package/cmp.png');

% if it is not grayscale
if ndims(ref)>2
    ref = rgb2gray(ref);
end
if ndims(cmp)>2
    cmp = rgb2gray(cmp);
end

%% put in poc function
poc_prototype(ref,cmp)
