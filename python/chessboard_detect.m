clf
% Trying to detect a chessboard from a photo manually (no libary help)
warning('off','images:initSize:adjustingMag');
%% 1 - Load image
filename = 'chessboard7.jpg';
original_img = imread(filename);
% Resize to max width of 512
img = imresize(original_img, 256/size(original_img,1));
fprintf('Image size: (%dx%dx%d)\n',size(img));
subplot(221);
imshow(img);
title('Original');

%% Intensity space
gray = im2double(rgb2gray(img));
subplot(222);
imshow(gray);
title('Gray');

%% 2 - Sobel edge detection on image
% Blur image a little first to keep only strong edges
 gray_blur = conv2(gray, ones(3)/9,'same');

edges = sobel(gray_blur) > 0.99;

% Canny edges
%edges = edge(gray,'canny');

subplot(223);
imshow(edges);
title('Edges')
%%
[H,T,R] = hough(edges);
subplot(224);
imshow(H,[],'XData',T,'YData',R,...
            'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

P  = houghpeaks(H,15,'threshold',ceil(0.3*max(H(:))), 'NHoodSize', [31,31]);
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'s','color','white');
hold off;

subplot(222);
lines = houghlines(edges,T,R,P,'FillGap',25,'MinLength',17);
imshow(gray), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
hold off;


%% 3 - Hough transform
theta_len = 180; % 4 degree bins
radius_len = 200;
[hough_space, thetas, radii] = hough_transform(edges > 0.99, theta_len, radius_len);
subplot(224);
imagesc(hough_space);
title('Hough')
%
threshold_hough_value = max(hough_space(:))*0.7;
thresholded_hough = hough_space > threshold_hough_value;
line_count = sum(sum(thresholded_hough));
if line_count < 100
    [rows,cols] = find(hough_space > threshold_hough_value);
    % rows = radii, cols = 'thetas
    subplot(224);
    imagesc(thetas*180/pi, radii, hough_space);
    xlabel('\theta'), ylabel('\rho');
    title('Hough')
    hold on;
    plot(thetas(cols)*180/pi, radii(rows),'rs');
    hold off;
    
    % Plot lines over image
    subplot(222);
    imshow(gray);
    title('Gray');
    hold on;
    for i = 1:line_count
        x = [0, size(img,2)];
        r = radii(rows(i));
        theta = thetas(cols(i));
        % r = x*cos(t) + y*sin(t);
        % y = (r - x*cos(t)) / sin(t); (unless sin(t) = 0, then line is
        % horizontal
        if (sin(theta) == 0)
            x = [r, r];
            y = [0, size(img,1)];
        else
            y = (r - x*cos(theta)) / sin(theta);
%             y(1) = max(min(y(1), size(img,1)), 0);
%             y(2) = max(min(y(2), size(img,1)), 0);
        end
        plot(x, y);
    end
    hold off;
else
    fprintf('Too many lines at this threshold (%d)\n', line_count);
end

%% 4 - Choose chessboard lines only
%% 5 - Rectify (affine warp) chessboard square
%% 6 - Pull out tiles
%% 7 - Check if empty, light or dark piece is on it (based on others)