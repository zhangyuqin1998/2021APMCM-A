function p = circleGrid(x,y,radius2,xCenter,yCenter,dx,dy)
numPixels = size(x, 1);
if (numPixels > 0)
    for n=1:numPixels
        grid = (x(n)+dx-xCenter).^2 + (y(n)+dy-yCenter).^2 < radius2;
        p(n) = mean(mean(grid));
    end
else
    p = [];
end
end

