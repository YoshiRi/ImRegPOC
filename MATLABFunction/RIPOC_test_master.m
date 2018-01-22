% 2016/11/28 Yoshi Ri @ Univ Tokyo
% RIPOC program 
% input : 2 images
% output : translation , rotation , scaling


%% �摜����
AI = rgb2gray(imread('luna1_1.png'));

%% �T�C�Y����
[height, width ] = size(AI);
 cy = height/2;
 cx = width/2;

 % Translation, rotation and scaling
BI = imtranslate(AI,[5.2, -19.6]);
BI = ImageRotateScale(BI,-100,1.2,height,width);




%% ���֐��̏��� �i�摜�[�̉e�����邽�߁j
% hannig window and root of hanning window
han_win = zeros(width);
Rhan_win = zeros(width);

% make window 
for i = 1 :height
    for j = 1:width
            han_win(i,j) = 0.25 * (1.0 + cos(pi*abs(cy- i) / height))*(1.0 + cos(pi*abs(cx - j) / width));
            % Root han_win
            Rhan_win(i,j)=abs(cos(pi*abs(cy - i) / height)*cos(pi*abs(cx - j) / width));
    end
end



%% ���֐��i�t�B���^�j��|����(convolute window) 
% IA = Rhan_win .* double(rgb2gray(AI));
IA = Rhan_win .* double((AI));
IB = Rhan_win .* double(BI);
 
%%�؂�o��
% IA = imcrop(AI,[ cx-width/2,cy-height/2,width-1,height-1]);
% IB = imcrop(BI,[ cx-width/2,cy-height/2,width-1,height-1]); 


%% 2DFFT
A=fft2(IA);
B=fft2(IB);

At = A./abs(A);
Bt = (conj(B))./abs(B);

%% �U�������̒��o�@���@�ΐ���
% As=cut_win .* fftshift(log(abs(A)+1));
% Bs=cut_win .* fftshift(log(abs(B)+1));
As= fftshift(log(abs(A)+1));
Bs= fftshift(log(abs(B)+1));





%% Log-Poler Transformation
% need bilinear interpolation

lpcA = zeros(height,width);
lpcB = zeros(height,width);
cx = width / 2;
cy = height / 2;

% cut off val of LPF 
LPmin_ = width*(1-log2(2*pi)/log2(width));
HPmax_ = width*(1-log2(4)/log2(width));
% marume
LPmin = floor(LPmin_)+1;
HPmax = floor(HPmax_);

M = floor (width / (HPmax-LPmin));
M = 4;

%% logplolar  Transformation with filter
for i= 0:width-1
        i_mag  = LPmin + i/M;
        r =power(width,(i_mag)/width);
    for j= 0:height-1
        x=r*cos(2*pi*j/height)+cx;
        y=r*sin(2*pi*j/height)+cy;
        if r < cx-1  % in the circle
             x0 = floor(x);
             y0 = floor(y);
             x1 = x0 + 1.0;
             y1 = y0 + 1.0;
            w0=x1-x;
            w1=x-x0;
            h0=y1-y;
            h1=y-y0;
            %�@Bilinear�⊮
            val=As(y0+1,x0+1)*w0*h0 + As(y0+1,x1+1)*w1*h0+ As(y1+1,x0+1)*w0*h1 + As(y1+1,x1+1)*w1*h1;
            lpcA(j+1,i+1)=val;
            val=Bs(y0+1,x0+1)*w0*h0 + Bs(y0+1,x1+1)*w1*h0+ Bs(y1+1,x0+1)*w0*h1 + Bs(y1+1,x1+1)*w1*h1;
            lpcB(j+1,i+1)=val;
        end
    end
end

%%%% end LogPoler %%%


PA = fft2(lpcA);
PB = fft2(lpcB);
Ap = PA./abs(PA);
Bp = (conj(PB))./abs(PB);
Pp = fftshift(ifft2(Ap.*Bp));


[mm,x]=max(Pp);
[mx,y]=max(mm);
px=y;
py=x(y);

%% Bilinear���
sum = Pp(py-1,px-1)+Pp(py,px-1)+Pp(py+1,px-1)+Pp(py-1,px)+Pp(py,px)+Pp(py+1,px)+Pp(py-1,px+1)+Pp(py,px+1)+Pp(py+1,px+1);

pxx = ( Pp(py-1,px-1)+Pp(py,px-1)+Pp(py+1,px-1) ) * (px-1) + ( Pp(py-1,px)+Pp(py,px)+Pp(py+1,px) ) * px + ( Pp(py-1,px+1)+Pp(py,px+1)+Pp(py+1,px+1) )* (px+1);
pxx = pxx/sum;

pyy = ( Pp(py-1,px-1)+Pp(py-1,px)+Pp(py-1,px+1) ) * (py-1) + ( Pp(py,px-1)+Pp(py,px)+Pp(py,px+1) ) * (py) + ( Pp(py+1,px-1)+Pp(py+1,px)+Pp(py+1,px+1) ) * (py+1);
pyy= pyy/sum;

dx = width/2 - pxx + 1;
dy = height/2 - pyy + 1;


%% Scale�ʂ̕␳
dx = dx/M;


%% ��]�ʂɂ�2�̃s�[�N���o������
theta1 = 360 * dy / height;
theta2 = theta1 + 180;
scale = power(width,dx/width)
figure;
mesh(Pp);
ylabel('rotation axis')
xlabel('scaling axis')
zlabel('correlation value')

%% ��]�E�g��k���� ��␳
% �ʓ|�����p�x�ɂ�2�̃p�^�[��������c
IB_recover1 = ImageRotateScale(IB, theta1,1/scale,width,height);
IB_recover2 = ImageRotateScale(IB, theta2,1/scale,width,height);

%% ���s�ړ��ʌ��o �� ��]�ʌ���

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

%% 2��ނ̉�]�ʂɂ���POC��s���C�s�[�N���o����C�l���傫���ق���^�l�Ƃ���
if mx1 > mx2
theta = theta1
% bilinear
sum1 = Pp1(py1-1,px1-1)+Pp1(py1,px1-1)+Pp1(py1+1,px1-1)+Pp1(py1-1,px1)+Pp1(py1,px1)+Pp1(py1+1,px1)+Pp1(py1-1,px1+1)+Pp1(py1,px1+1)+Pp1(py1+1,px1+1);

pxx1 = ( Pp1(py1-1,px1-1)+Pp1(py1,px1-1)+Pp1(py1+1,px1-1) ) * (px1-1) + ( Pp1(py1-1,px1)+Pp1(py1,px1)+Pp1(py1+1,px1) ) * px1 + ( Pp1(py1-1,px1+1)+Pp1(py1,px1+1)+Pp1(py1+1,px1+1) )* (px1+1);
pxx1 = pxx1/sum1;
pyy1 = ( Pp1(py1-1,px1-1)+Pp1(py1-1,px1)+Pp1(py1-1,px1+1) ) * (py1-1) + ( Pp1(py1,px1-1)+Pp1(py1,px1)+Pp1(py1,px1+1) ) * (py1) + ( Pp1(py1+1,px1-1)+Pp1(py1+1,px1)+Pp1(py1+1,px1+1) ) * (py1+1);
pyy1= pyy1/sum1;

% get translation from center
dx = width/2 - pxx1 + 1
dy = height/2 - pyy1 + 1

% show result
f1 = figure;
result = imtranslate(IB_recover1,[-dx, -dy]);
imshow(abs(double(IA)-result),[0 255]);
SaveFigPDF(f1,'sabun_bibun');
f2 = figure;
imshow(result,[0 255]);
SaveFigPDF(f2,'compared_moved');
f3 = figure;
imshow(IA,[0 255]);
SaveFigPDF(f3,'ref_bibun');
f4 = figure;
imshow(IB,[0 255]);
SaveFigPDF(f4,'compared');


else
theta = theta2
sum2 = Pp2(py2-1,px2-1)+Pp2(py2,px2-1)+Pp2(py2+1,px2-1)+Pp2(py2-1,px2)+Pp2(py2,px2)+Pp2(py2+1,px2)+Pp2(py2-1,px2+1)+Pp2(py2,px2+1)+Pp2(py2+1,px2+1);

pxx2 = ( Pp2(py2-1,px2-1)+Pp2(py2,px2-1)+Pp2(py2+1,px2-1) ) * (px2-1) + ( Pp2(py2-1,px2)+Pp2(py2,px2)+Pp2(py2+1,px2) ) * px2 + ( Pp2(py2-1,px2+1)+Pp2(py2,px2+1)+Pp2(py2+1,px2+1) )* (px2+1);
pxx2 = pxx2/sum2;

pyy2 = ( Pp2(py2-1,px2-1)+Pp2(py2-1,px2)+Pp2(py2-1,px2+1) ) * (py2-1) + ( Pp2(py2,px2-1)+Pp2(py2,px2)+Pp2(py2,px2+1) ) * (py2) + ( Pp2(py2+1,px2-1)+Pp2(py2+1,px2)+Pp2(py2+1,px2+1) ) * (py2+1);
pyy2= pyy2/sum2;

dx = width/2 - pxx2 + 1
dy = height/2 - pyy2 + 1

% show result 
f1 = figure;
result = imtranslate(IB_recover2,[-dx, -dy]);
imshow(abs(double(IA)-result),[0 255]);
SaveFigPDF(f1,'sabun_bibun');
f2 = figure;
imshow(result,[0 255]);
SaveFigPDF(f2,'compared_moved');
f3 = figure;
imshow(IA,[0 255]);
SaveFigPDF(f3,'ref_bibun');
f4 = figure;
imshow(IB,[0 255]);
SaveFigPDF(f4,'compared');
end

