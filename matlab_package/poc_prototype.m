% 2016/9/12 Yoshi Ri @ Univ Tokyo
% 2017/8/11 Updated

%%%%%%%%%%%%%%%%%%
% input : 2 gray scale images with the same size 
% output : translation , rotation , scaling, poc_peaks
%%%%%%%%%%%%%%%%%%


function [dx,dy,theta,scale,peak] = poc_prototype(AI,BI)

%% Get size
[height, width ] = size(AI);
 cy = height/2;
 cx = floor(width/2);


%% Create window
% hannig window and root of hanning window
han_win = zeros(width);
Rhan_win = zeros(width);

% make window 
for i = 1 :height
    for j = 1:width
            han_win(i,j) = 0.25 * (1.0 + cos(pi*abs(cy- i) / height))*(1.0 + cos(pi*abs(cx - j) / width));
            % Root han_win
            Rhan_win(i,j)=abs(cos(pi*abs(cy - i) / height)*cos(pi*abs(cx - j) / width));
            if i >height/8  &&  i<height*7/8
                fi = 1;
            elseif i <= height/8
                fi = (i-1)/height * 8;
            elseif i >= height*7/8
                fi = (height - i)/height *8;
            end
            if j >width/8  &&  j<width*7/8
                fj = 1;
            elseif j <= width/8
                fj = (j-1)/width * 8;
            elseif j >= width*7/8
                fj = (width - j)/width *8;
            end
            trapez_win(i,j)=fi*fj;
    end
end

%% Apply Windowing 
 IA = han_win .* double((AI));
 IB = han_win .* double(BI);

%% 2DFFT
A=fft2(IA);
B=fft2(IB);

At = A./abs(A);
Bt = (conj(B))./abs(B);

%% get magnitude and whitening
As= fftshift(log(abs(A)+1));
Bs= fftshift(log(abs(B)+1));

%% Log-Poler Transformation
% need bilinear interpolation

lpcA = zeros(height,width);
lpcB = zeros(height,width);
cx = width / 2;
cy = height / 2;

% cut off val of LPF 
LPmin = width*(1-log2(2*pi)/log2(width));
LPmin = width*(1-log2(4*pi)/log2(width));

%start logplolar 
for i= 0:width-1
        r =power(width,(i)/width);
    for j= 0:height-1
        x=r*cos(2*pi*j/height)+cx;
        y=r*sin(2*pi*j/height)+cy;
        if r < cx * 3 / 5 % in the circle
             x0 = floor(x);
             y0 = floor(y);
             x1 = x0 + 1.0;
             y1 = y0 + 1.0;
            w0=x1-x;
            w1=x-x0;
            h0=y1-y;
            h1=y-y0;
            %@Bilinear Interpolation
            val=As(y0+1,x0+1)*w0*h0 + As(y0+1,x1+1)*w1*h0+ As(y1+1,x0+1)*w0*h1 + As(y1+1,x1+1)*w1*h1;
            % High pass
            if i > LPmin 
                 lpcA(j+1,i+1)=val;
            else
                 lpcA(j+1,i+1)=0;
            end
            val=Bs(y0+1,x0+1)*w0*h0 + Bs(y0+1,x1+1)*w1*h0+ Bs(y1+1,x0+1)*w0*h1 + Bs(y1+1,x1+1)*w1*h1;
            if i > LPmin 
                 lpcB(j+1,i+1)=val;
            else
                 lpcB(j+1,i+1)=0;
            end
        end
    end
end

%%%% end LogPoler %%%


% phase correlation to get rotation and scaling
PA = fft2(lpcA);
PB = fft2(lpcB);
Ap = PA./abs(PA);
Bp = (conj(PB))./abs(PB);
Pp = fftshift(ifft2(Ap.*Bp));


[mm,x]=max(Pp);
[mx,y]=max(mm);
px=y;
py=x(y);
mg =1;
%% Bilinear Interpolation
Rect = Pp(py-mg:py+mg,px-mg:px+mg);
vert = sum(Rect,1); horz = sum(Rect,2);
sum11=sum(vert);

pyy=[py-mg:py+mg] * horz /sum11;
pxx=[px-mg:px+mg] * vert' /sum11;

dx = floor(width/2) - pxx + 1;
dy = floor(height/2 )- pyy + 1;


%% Translation Extraction
theta1 = 360 * dy / height;
theta2 = theta1 + 180;
scale = 1/power(width,dx/width)
figure(1);
mesh(Pp);
ylabel('rotation axis')
xlabel('scaling axis')
zlabel('correlation value')

%% Compensate Rotation and scaling
IB_recover1 = ImageRotateScale(IB, theta1,scale,width,height);
IB_recover2 = ImageRotateScale(IB, theta2,scale,width,height);

%% Translation estimation

IB_R1=fft2(IB_recover1);
IB_R2=fft2(IB_recover2);
IB1p = (conj(IB_R1))./abs(IB_R1);
IB2p = (conj(IB_R2))./abs(IB_R2);

App = A./abs(A);
Pp1 = fftshift(ifft2(App.*IB1p));
Pp2 = fftshift(ifft2(App.*IB2p));

[mm1,x1]=max(Pp1);
[mx1,y1]=max(mm1);
px1=y1;
py1=x1(y1);

[mm2,x2]=max(Pp2);
[mx2,y2]=max(mm2);
px2=y2;
py2=x2(y2);

%% Comparison to get True rotation
if mx1 > mx2
theta = theta1

Rect = Pp1(py1-mg:py1+mg,px1-mg:px1+mg);
vert = sum(Rect,1); horz = sum(Rect,2);
sum11=sum(vert);

pyy1=[py1-mg:py1+mg] * horz /sum11;
pxx1=[px1-mg:px1+mg] * vert' /sum11;


% get translation from center
dx = floor(width/2) - pxx1 + 1
dy = floor(height/2 )- pyy1 + 1

peak = mx1
% show result
f1 = figure(2);
result = imtranslate(IB_recover1,[-dx, -dy]);
imshow(abs(double(IA)-result),[0 255]);
title('difference of images')
SaveFigPDF(f1,'diff_image');
f2 = figure(3);
imshow(result,[0 255]);
title('rotation compensated compared image')
SaveFigPDF(f2,'rotation_compensated');
f3 = figure(4);
imshow(IA,[0 255]);
title('refrence image')
SaveFigPDF(f3,'ref');
f4 = figure(5);
imshow(IB,[0 255]);
title('compared image')
SaveFigPDF(f4,'compared');
figure(6);
mesh(Pp1);
title('POC peak')


else
theta = theta2

Rect = Pp2(py2-mg:py2+mg,px2-mg:px2+mg);
vert = sum(Rect,1); horz = sum(Rect,2);
sum11=sum(vert);

pyy2=[py2-mg:py2+mg] * horz /sum11;
pxx2=[px2-mg:px2+mg] * vert' /sum11;


dx = floor(width/2) - pxx2 + 1
dy = floor(height/2 )- pyy2 + 1

peak = mx2
% show result 
f1 = figure(2);
result = imtranslate(IB_recover2,[-dx, -dy]);
imshow(abs(double(IA)-result),[0 255]);
title('difference of images')
SaveFigPDF(f1,'diff_image');
f2 = figure(3);
imshow(result,[0 255]);
title('rotation compensated compared image')
SaveFigPDF(f2,'rotation_compensated');
f3 = figure(4);
imshow(IA,[0 255]);
title('refrence image')
SaveFigPDF(f3,'ref');
f4 = figure(5);
imshow(IB,[0 255]);
title('compared image')
SaveFigPDF(f4,'compared');
figure(6);
mesh(Pp2);
title('POC peak')

end


%% figure
figure(7);
stiched = [IA,IB;result,abs(double(IA)-result)];
imshow(stiched,[0 255])

stiched = [IA;IB];
imshow(stiched,[0 255])
end