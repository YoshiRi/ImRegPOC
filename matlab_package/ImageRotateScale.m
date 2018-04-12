function resizeimage = ImageRotateScale(Image,theta,scale,width,height)

 cx = width/2;
 cy = height/2;
 cx = width/2 + 1/2;
 cy = height/2 + 1/2;

ct = cosd(theta);
st = sind(theta);

% tform = affine2d([cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1]);

resizeimage = zeros(width,height);

for i= 0:width-1
    for j= 0:height-1
        x=(ct*(i-cx)-st*(j-cy))*scale+cx;
        y=(ct*(j-cy)+st*(i-cx))*scale+cy;
        if 0 < x && x < width - 1 && 0 < y && y < height - 1
             x0 = floor(x);
             y0 = floor(y);
             x1 = x0 + 1.0;
             y1 = y0 + 1.0;
            w0=x1-x;
            w1=x-x0;
            h0=y1-y;
            h1=y-y0;
            %@linear•âŠ®
            val=Image(y0+1,x0+1)*w0*h0 + Image(y0+1,x1+1)*w1*h0+ Image(y1+1,x0+1)*w0*h1 + Image(y1+1,x1+1)*w1*h1;
                resizeimage(j+1,i+1)=val;
            end
    end
end


end
