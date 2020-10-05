clear;clc;close all
X = -4:0.1:4;
plot(X, sigmoid(X));
axis equal
axis([-inf, inf, -inf, inf]);
hold on

yy = -0.5:0.1:1.5;
xx = zeros(21,1);
plot(xx,yy,'K' );
y = zeros(81,1);
yyy = ones(81,1);
plot(X,y,'K' );
plot(X,yyy,'K--');
title('Image of \sigma(x)');