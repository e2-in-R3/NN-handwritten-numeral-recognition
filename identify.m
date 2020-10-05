close all

load('Theta.mat');                %引入已訓練好的Theta
I = imread('1.png');              %引入圖片(改檔名)
J = imresize(I, [20, 20]);        %將圖片壓成20*20像素
K = rgb2gray(J);                  %將彩色圖片改成灰階圖片
L = 1 - im2double(K);             %將黑白色倒過來
M = L(:)';                        %將20*20拉成1*40
displayData(M);                   %印出圖片
h1 = sigmoid([1, M] * Theta1);    %將圖片透過Theta1,Theta2送到1*10的結果
h2 = sigmoid([1, h1] * Theta2);
[dummy, p] = max(h2, [], 2);      %找出10個中最大的
fprintf('%d\n', p);               %印出結果