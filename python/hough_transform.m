function [hough_space, thetas, radii] = hough_transform(edges, theta_len, radius_len)
r_max = max(size(edges));
thetas = linspace(-pi/2,pi/2, theta_len); % 0 - 180
radii = linspace(-r_max,r_max, radius_len);


hough_space = zeros(radius_len, theta_len);

% For each pixel
for y = 1:size(edges,1)
    for x = 1:size(edges,2)
        % If is an edge pixel
        if (edges(y,x) > 0)
            % For each theta bin
            for t_idx = 1:theta_len
                t = thetas(t_idx);
                r = x*cos(t) + y*sin(t);
                if r >= -r_max && r < r_max
                    % Find correct r bin
                    r_idx = 1;
                    while r > radii(r_idx)
                        r_idx = r_idx + 1;
                    end
                    
                    % Accumulate
                    hough_space(r_idx,t_idx) = hough_space(r_idx,t_idx) + 1;
                end
            end
        end
    end
end


end