% Visualize the digit data
% Assumes D is an NxM matrix where N is the number of images and M is the
% number of pixels per image.
function plotDigits(D)

N=size(D,1); % number of images
im_m=ceil(sqrt(N));
im_n=ceil(N/im_m);

% correct for size with actual x, y dims and create output matrix
m=16
n=16;
im=zeros(im_m*m,im_n*n,'uint8');

for i=1:im_m
    for j=1:im_n
        if ((i-1)*im_n+j<=N)
            im((i-1)*m+1:i*m,(j-1)*n+1:j*n)=reshape(D((i-1)*im_n+j,:),n,m)';
        end
    end
end

colormap gray;
imagesc(im);



