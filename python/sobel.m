function edges = sobel(img)

kernel_x = [
    -1,0,1;
    -2,0,2;
    -1,0,1];

kernel_y = [
    -1,-2,-1;
    0,0,0;
    1,2,1];

Gx = zeros(size(img));
Gy = zeros(size(img));

%% Slow forloop way
% for i = 2:size(img,1)-1
%     for j = 2:size(img,2)-1
%         Gx(i,j) = sum(sum(img(i-1:i+1,j-1:j+1).*kernel_x));
%         Gy(i,j) = sum(sum(img(i-1:i+1,j-1:j+1).*kernel_y));
%     end
% end
%% faster bsxfun way
% todo
%% Use optimized conv2 
Gx = conv2(img, kernel_x, 'valid');
Gy = conv2(img, kernel_y, 'valid');


edges = abs(Gx)+ abs(Gy);
%edges = sqrt(Gx.^2 + Gy.^2);
end